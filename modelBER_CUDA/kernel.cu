#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <device_atomic_functions.h>
#include <curand.h>
#include <curand_kernel.h>
#include <vector_functions.h>

#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>


#include <stdio.h>
#include <time.h>

#include <stdio.h>
#include <stdlib.h>



#include "modelBER_params.cuh"


__constant__ unsigned long long int kRandomSeed;		/// <summary>乱数のシード値</summary>



#define CUDA_SAFE_CALL(func) \
{ \
     cudaError_t err = (func); \
     if (err != cudaSuccess) \
	 { \
         fprintf(stderr, "[Error] %s (error code: %d) at %s line %d\n", cudaGetErrorString(err), err, __FILE__, __LINE__); \
         exit(err); \
	 } \
}


#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 600
#else

__device__ double atomicAdd(double* address, double val)
{
	unsigned long long int* address_as_ull = (unsigned long long int*)address;
	unsigned long long int old = *address_as_ull, assumed;
	do {
		assumed = old;
		old = atomicCAS(address_as_ull, assumed, __double_as_longlong(val + __longlong_as_double(assumed)));
	} while (assumed != old);
	return __longlong_as_double(old);
}

#endif

__device__ inline double calcCurieFromCu(double cu)
{
	return (2 * J_FE_FE * 4 * (1 - cu) * (G_FE - 1)  * (G_FE - 1) * S_FE * (S_FE + 1)) / (3 * K_B);
}

__device__ inline double calcCuFromCurie(double temp_curie)
{
	return -temp_curie * (3 * K_B) / (2 * J_FE_FE * 4 * (G_FE - 1) *(G_FE - 1) * S_FE * (S_FE + 1)) + 1;
}

__device__ inline double calcBrillouin(double s, double x)
{
	double o1 = (2 * s + 1) / (2 * s);
	double o2 = 1 / (2 * s);
	return o1 / tanh(o1 * x) - o2 / tanh(o2 * x);
}

__device__ double calc_sFe_Mean(double temp, double cu)
{
	double vl = 0.0;
	double vr = S_FE_MEAN_MAX;
	double v;

	double dxdv = (S_FE * 2 * J_FE_FE * 4 * (1 - cu) * (G_FE - 1)*(G_FE - 1)) / (K_B * temp);
	do
	{
		v = (vl + vr) / 2.0;
		double x = dxdv * v;
		double f = v - S_FE * calcBrillouin(S_FE, x);

		if (f < 0)
			vl = v;
		else
			vr = v;


	} while (fabs((vl - vr) / v) > S_FE_MEAN_ERROR);

	return v;
}

__device__ double calcKb(double temp, double hw, double cu)
{

	double temp_curie = calcCurieFromCu(cu);
	if (temp >= temp_curie) return 0;

	double dFeFe = BULK_D_FE_FE * KU_KBULK;

	double s = calc_sFe_Mean(temp, cu);

	double total_atom_number = 1 / ((FE  * (1 - cu) * V_FE) + ((1 - FE)*(1 - cu) * V_PT) + cu * V_CU);
	double ku = total_atom_number * FE * (1 - cu)* (4 * (1 - cu))  * dFeFe  * s * s;
	double hc = 2 * ((4 * (1 - cu))  * dFeFe  * s) / (M_B * G_FE);
	if (hw < 0 && hc <= fabs(hw)) return 0;

	double kb = (ku * GRAIN_VOLUME) / K_B / temp * (1 + hw / hc) * (1 + hw / hc);
	return kb;
}

__device__ void calcKb(double temp, double hw, double cu, double &kbp, double &kbm)
{

	double temp_curie = calcCurieFromCu(cu);
	kbm = kbp = 0;
	if (temp >= temp_curie) return;

	double dFeFe = BULK_D_FE_FE * KU_KBULK;

	double s = calc_sFe_Mean(temp, cu);

	double total_atom_number = 1 / ((FE  * (1 - cu) * V_FE) + ((1 - FE)*(1 - cu) * V_PT) + cu * V_CU);
	double ku = total_atom_number * FE * (1 - cu)* (4 * (1 - cu))  * dFeFe  * s * s;
	double hc = 2 * ((4 * (1 - cu))  * dFeFe  * s) / (M_B * G_FE);
	kbp = (ku * GRAIN_VOLUME) / K_B / temp * (1 + hw / hc) * (1 + hw / hc);
	kbm = hc <= hw ? 0 : (ku * GRAIN_VOLUME) / K_B / temp * (1 - hw / hc) * (1 - hw / hc);
}

__global__
void calcKbListKernel(double *kb_list, int kb_list_size, double hw)
{
	double cu = calcCuFromCurie(TEMP_CURIE_MEAN);
	for (int i = 0; i < kb_list_size; i++)
	{
		double temp = TEMP_CURIE_MEAN - THERMAL_GRADIENT * LINER_VELOCITY * TAU_AP * 1.0e+9 * i;
		if (temp < TEMP_AMBIENT)
			temp = TEMP_AMBIENT;

		kb_list[i] = calcKb(temp, hw, cu);
	}

}


#if BER_ALGORITHM == 0

__global__ void calcContinusBitErrorRateKernel(int *ber_list, int ber_list_count, double hw)
{
	int thread_number = threadIdx.x + blockIdx.x * blockDim.x;
	double grain_tcs[GRAIN_COUNT];			// グレインごとのTc
	double grain_cu[GRAIN_COUNT];			// グレインごとのCu組成
	double grain_area[GRAIN_COUNT];			// グレインごとの面積
	//double grain_ku_kum[GRAIN_COUNT];			// グレインごとのKu/Kum
	int grain_dir[GRAIN_COUNT];				// グレインの磁化の向き (1 = 逆方向、-1 = 順方向)
	double grain_size_mu = log((GRAIN_MEAN * GRAIN_MEAN) / sqrt(GRAIN_SD * GRAIN_SD + GRAIN_MEAN * GRAIN_MEAN));  					  // グレインサイズ分散のμ
	double grain_size_sigma = (sqrt(log((GRAIN_SD * GRAIN_SD) / (GRAIN_MEAN * GRAIN_MEAN) + 1)));									  // グレインサイズ分散のσ
	const int hw_switch_ap = (int)(BIT_PITCH / LINER_VELOCITY *1.0e-9 * F0_AP);														  // 書込磁界順方向終了タイミング
	const int attempt_offset = (int)(TEMP_CURIE_MEAN * TEMP_CURIE_SD * 2 / (THERMAL_GRADIENT * LINER_VELOCITY * TAU_AP * 1.0e+9));	  // Tcm以前の予備シミュレーション


	curandStateMRG32k3a rand_stat;			// 乱数ステータス	
	curand_init(kRandomSeed, thread_number, ber_list_count * GRAIN_COUNT, &rand_stat);


	for (int i = 0; i < GRAIN_COUNT; i++)
	{
		grain_tcs[i] = curand_normal_double(&rand_stat) * TEMP_CURIE_SD + TEMP_CURIE_MEAN;
		grain_cu[i] = calcCuFromCurie(grain_tcs[i]);
		double a = curand_log_normal_double(&rand_stat, grain_size_mu, grain_size_sigma);
		grain_area[i] = a * a;
		//grain_ku_kum[i] = 1;
		grain_dir[i] = 1;
	}

	double signed_hw = hw;
	for (int i = -attempt_offset; i < ber_list_count; i++)
	{
		double signal_power = 0;

		double temp = TEMP_CURIE_MEAN - THERMAL_GRADIENT * LINER_VELOCITY * TAU_AP * 1.0e+9 * i;
		if (temp < TEMP_AMBIENT)
			temp = TEMP_AMBIENT;


		if (i == 0 || i == hw_switch_ap)
			signed_hw = -signed_hw;


		for (int k = 0; k < GRAIN_COUNT; k++)
		{
			// 磁化反転する確率
			// hw = -1 = 順方向
			// grain_dir = 1 逆方向
			// hw * grain_dir = -1 hw方向への反転確率
			if (temp > grain_tcs[k]) continue;

			double rev_prob = exp(-calcKb(temp, signed_hw * grain_dir[k], grain_cu[k]) * grain_area[k]);
			double dice = curand_uniform(&rand_stat);
			if (rev_prob > dice)
				grain_dir[k] = -grain_dir[k];
			if (grain_dir[k] < 0)
				signal_power += grain_area[k];
		}

		if (READABLE_THRETH > signal_power && 0 <= i && i < ber_list_count)
			atomicAdd(&ber_list[i], 1);
	}
}

void calcContinusBitErrorRateHost(double *bER_list, int bER_list_count, double hw)
{
	int *dev_be_list;
	int *be_list = (int*)malloc(sizeof(int) * bER_list_count);
	unsigned long long int random_seed = (unsigned long long int)(time(NULL));

	for (int i = 0; i < bER_list_count; i++)
		be_list[i] = 0;

	cudaSetDevice(CUDA_DEVICE_NUM);
	cudaMalloc((void**)&dev_be_list, sizeof(int) * bER_list_count);
	cudaMemcpy(dev_be_list, be_list, sizeof(int) * bER_list_count, cudaMemcpyHostToDevice);

	cudaMemcpy(&kRandomSeed, &random_seed, sizeof(unsigned long long int), cudaMemcpyHostToDevice);


	calcContinusBitErrorRateKernel << <CUDA_BLOCK_COUNT, CUDA_THREAD_COUNT >> >(dev_be_list, bER_list_count, hw);

	auto err = cudaGetLastError();


	cudaDeviceSynchronize();
	cudaMemcpy(be_list, dev_be_list, sizeof(int) * bER_list_count, cudaMemcpyDeviceToHost);

	for (int i = 0; i < bER_list_count; i++)
	{
		bER_list[i] = (double)be_list[i] / BIT_COUNT;
	}
	cudaFree(dev_be_list);
	free(be_list);
}

__global__ void calcMidLastBitErrorRateKernel(int *mid_be_list, int *last_be_list, double hw)
{
	int thread_number = threadIdx.x + blockIdx.x * blockDim.x;
	double grain_tcs[GRAIN_COUNT];			// グレインごとのTc
	double grain_cu[GRAIN_COUNT];			// グレインごとのCu組成
	double grain_area[GRAIN_COUNT];			// グレインごとの面積
	//double grain_ku_kum[GRAIN_COUNT];			// グレインごとのKu/Kum
	int grain_dir[GRAIN_COUNT];				// グレインの磁化の向き (1 = 逆方向、-1 = 順方向)
	double grain_size_mu = log((GRAIN_MEAN * GRAIN_MEAN) / sqrt(GRAIN_SD * GRAIN_SD + GRAIN_MEAN * GRAIN_MEAN));  					  // グレインサイズ分散のμ
	double grain_size_sigma = (sqrt(log((GRAIN_SD * GRAIN_SD) / (GRAIN_MEAN * GRAIN_MEAN) + 1)));									  // グレインサイズ分散のσ
	const int hw_switch_ap = (int)(BIT_PITCH / LINER_VELOCITY *1.0e-9 * F0_AP);														  // 書込磁界順方向終了タイミング
	const int attempt_offset = (int)(TEMP_CURIE_MEAN * TEMP_CURIE_SD * 2 / (THERMAL_GRADIENT * LINER_VELOCITY * TAU_AP * 1.0e+9));	  // Tcm以前の予備シミュレーション
	const int last_attempt = hw_switch_ap * 2 + attempt_offset;



	curandStateMRG32k3a rand_stat;			// 乱数ステータス	
	curand_init(kRandomSeed, thread_number, last_attempt * GRAIN_COUNT, &rand_stat);


	for (int i = 0; i < GRAIN_COUNT; i++)
	{
		grain_tcs[i] = curand_normal_double(&rand_stat) * TEMP_CURIE_SD + TEMP_CURIE_MEAN;
		grain_cu[i] = calcCuFromCurie(grain_tcs[i]);
		double a = curand_log_normal_double(&rand_stat, grain_size_mu, grain_size_sigma);
		grain_area[i] = a * a;
		//grain_ku_kum[i] = 1;
		grain_dir[i] = 1;
	}
	mid_be_list[thread_number] = 0;
	last_be_list[thread_number] = 0;

	double signed_hw = hw;
	for (int i = -attempt_offset; i < last_attempt; i++)
	{
		double signal_power = 0;

		double temp = TEMP_CURIE_MEAN - THERMAL_GRADIENT * LINER_VELOCITY * TAU_AP * 1.0e+9 * i;
		if (temp < TEMP_AMBIENT)
			temp = TEMP_AMBIENT;


		if (i == 0 || i == hw_switch_ap)
			signed_hw = -signed_hw;


		for (int k = 0; k < GRAIN_COUNT; k++)
		{
			// 磁化反転する確率
			// hw = -1 = 順方向
			// grain_dir = 1 逆方向
			// hw * grain_dir = -1 hw方向への反転確率
			if (temp > grain_tcs[k]) continue;

			double rev_prob = exp(-calcKb(temp, signed_hw * grain_dir[k], grain_cu[k]) * grain_area[k]);
			double dice = curand_uniform(&rand_stat);
			if (rev_prob > dice)
				grain_dir[k] = -grain_dir[k];
			if (grain_dir[k] < 0)
				signal_power += grain_area[k];
		}

		if (i == hw_switch_ap - 1 && READABLE_THRETH > signal_power)
			mid_be_list[thread_number] = 1;
		if (i == last_attempt - 1 && READABLE_THRETH > signal_power)
			last_be_list[thread_number] = 1;
	}

}

void calcMidLastBitErrorRateHost(double *mid_bER, double *last_bER, double hw)
{
	const int list_size = BIT_COUNT;
	int *mid_be_list = (int*)malloc(sizeof(int) * list_size);
	int *last_be_list = (int*)malloc(sizeof(int) * list_size);
	int *dev_mid_be_list = NULL;
	int *dev_last_be_list = NULL;

	cudaSetDevice(CUDA_DEVICE_NUM);
	cudaMalloc((void**)&dev_mid_be_list, sizeof(int) * list_size);
	cudaMalloc((void**)&dev_last_be_list, sizeof(int) * list_size);

	cudaMemcpy(dev_mid_be_list, mid_be_list, sizeof(int) * list_size, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_last_be_list, last_be_list, sizeof(int) * list_size, cudaMemcpyHostToDevice);

	calcMidLastBitErrorRateKernel << <CUDA_BLOCK_COUNT, CUDA_THREAD_COUNT >> >(dev_mid_be_list, dev_last_be_list, hw);
	cudaDeviceSynchronize();
	cudaMemcpy(mid_be_list, dev_mid_be_list, sizeof(int) * list_size, cudaMemcpyDeviceToHost);
	cudaMemcpy(last_be_list, dev_last_be_list, sizeof(int) * list_size, cudaMemcpyDeviceToHost);

	double temp_mid_bER = 0;
	double temp_last_bER = 0;
	for (int i = 0; i < list_size; i++)
	{
		temp_mid_bER += mid_be_list[i];
		temp_last_bER += last_be_list[i];
	}
	temp_mid_bER /= list_size;
	temp_last_bER /= list_size;

	*mid_bER = temp_mid_bER;
	*last_bER = temp_last_bER;

	cudaFree(dev_mid_be_list);
	cudaFree(dev_last_be_list);
	free(mid_be_list);
	free(last_be_list);

}

void calcHwList(FILE *fp)
{
	fprintf(fp, "Hw[kOe]\tbER\tbER(WE)\tbER(EAW)\n");
	for (int i = 0; i < HW_LIST_SIZE; i++)
	{
		printf("%d / %d \r", i, HW_LIST_SIZE);

		double hw = HW_MAX * i / HW_LIST_SIZE;
		double mid_bER = 0;
		double last_bER = 0;
		calcMidLastBitErrorRateHost(&mid_bER, &last_bER, hw);

		fprintf(fp, "%f\t%.10e\t%.10e\t%.10e\n", hw * 1e-3, last_bER, mid_bER, last_bER - mid_bER);
	}

}

#else

__device__ inline double calcPattern(double *grain_area, double *grain_prov)
{
	double bit_error_rate = 0;
#define GAX(n) (0)
#define GAO(n) (grain_area[n])
#define GPX(n) (grain_prov[n])
#define GPO(n) (1 - grain_area[n])

#if GRAIN_COUNT == 1
	if (grain_area[0] < READABLE_THRETH) bit_error_rate = grain_prov[0];

#elif GRAIN_COUNT == 4
	if (GAX(0) + GAX(1) + GAX(2) + GAX(3) < READABLE_THRETH) bit_error_rate += GPX(0) * GPX(1) * GPX(2) * GPX(3);  //  0
	if (GAO(0) + GAX(1) + GAX(2) + GAX(3) < READABLE_THRETH) bit_error_rate += GPO(0) * GPX(1) * GPX(2) * GPX(1);  //  1
	if (GAX(0) + GAO(1) + GAX(2) + GAX(3) < READABLE_THRETH) bit_error_rate += GPX(0) * GPO(1) * GPX(2) * GPX(1);  //  2
	if (GAO(0) + GAO(1) + GAX(2) + GAX(3) < READABLE_THRETH) bit_error_rate += GPO(0) * GPO(1) * GPX(2) * GPX(1);  //  3
	if (GAX(0) + GAX(1) + GAX(2) + GAX(3) < READABLE_THRETH) bit_error_rate += GPX(0) * GPX(1) * GPX(2) * GPX(1);  //  4
	if (GAO(0) + GAX(1) + GAO(2) + GAX(3) < READABLE_THRETH) bit_error_rate += GPO(0) * GPX(1) * GPO(2) * GPX(1);  //  5
	if (GAX(0) + GAO(1) + GAO(2) + GAX(3) < READABLE_THRETH) bit_error_rate += GPX(0) * GPO(1) * GPO(2) * GPX(1);  //  6
	if (GAO(0) + GAO(1) + GAO(2) + GAX(3) < READABLE_THRETH) bit_error_rate += GPO(0) * GPO(1) * GPO(2) * GPX(1);  //  7
	if (GAX(0) + GAX(1) + GAO(2) + GAX(3) < READABLE_THRETH) bit_error_rate += GPX(0) * GPX(1) * GPO(2) * GPX(1);  //  8
	if (GAO(0) + GAX(1) + GAX(2) + GAO(3) < READABLE_THRETH) bit_error_rate += GPO(0) * GPX(1) * GPX(2) * GPO(1);  //  9
	if (GAX(0) + GAO(1) + GAX(2) + GAO(3) < READABLE_THRETH) bit_error_rate += GPX(0) * GPO(1) * GPX(2) * GPO(1);  // 10
	if (GAO(0) + GAO(1) + GAX(2) + GAO(3) < READABLE_THRETH) bit_error_rate += GPO(0) * GPO(1) * GPX(2) * GPO(1);  // 11
	if (GAX(0) + GAX(1) + GAO(2) + GAO(3) < READABLE_THRETH) bit_error_rate += GPX(0) * GPX(1) * GPO(2) * GPO(1);  // 12
	if (GAO(0) + GAX(1) + GAO(2) + GAO(3) < READABLE_THRETH) bit_error_rate += GPO(0) * GPX(1) * GPO(2) * GPO(1);  // 13
	if (GAX(0) + GAO(1) + GAO(2) + GAO(3) < READABLE_THRETH) bit_error_rate += GPX(0) * GPO(1) * GPO(2) * GPO(1);  // 14
	if (GAO(0) + GAO(1) + GAO(2) + GAO(3) < READABLE_THRETH) bit_error_rate += GPO(0) * GPO(1) * GPO(2) * GPO(1);  // 15
#else
#error Not implement for this GRAIN_COUNT pattern
#endif
	/*
	#undef GAX(n)
	#undef GAO(n)
	#undef GPX(n)
	#undef GPO(n)
	*/

	return bit_error_rate;
}

__global__ void calcContinusBitErrorRateKernel(double *ber_list, double ber_list_count, double hw)
{
	int thread_number = threadIdx.x + blockIdx.x * blockDim.x;
	double grain_prov[GRAIN_COUNT];			// グレインの逆方向に向いている確率
	double grain_tcs[GRAIN_COUNT];			// グレインごとのTc
	double grain_cu[GRAIN_COUNT];			// グレインごとのCu組成
	double grain_area[GRAIN_COUNT];			// グレインごとの面積
	//double grain_ku_kum[GRAIN_COUNT];			// グレインごとのKu/Kum
	double grain_size_mu = log((GRAIN_MEAN * GRAIN_MEAN) / sqrt(GRAIN_SD * GRAIN_SD + GRAIN_MEAN * GRAIN_MEAN));  					  // グレインサイズ分散のμ
	double grain_size_sigma = (sqrt(log((GRAIN_SD * GRAIN_SD) / (GRAIN_MEAN * GRAIN_MEAN) + 1)));									  // グレインサイズ分散のσ
	const int hw_switch_ap = (int)(BIT_PITCH / LINER_VELOCITY *1.0e-9 * F0_AP);														  // 書込磁界順方向終了タイミング
	const int attempt_offset = (int)(TEMP_CURIE_MEAN * TEMP_CURIE_SD * 2 / (THERMAL_GRADIENT * LINER_VELOCITY * TAU_AP * 1.0e+9));	  // Tcm以前の予備シミュレーション


	curandStateMRG32k3a rand_stat;			// 乱数ステータス	
	curand_init(kRandomSeed, thread_number, ber_list_count * GRAIN_COUNT, &rand_stat);


	for (int i = 0; i < GRAIN_COUNT; i++)
	{
		grain_tcs[i] = curand_normal_double(&rand_stat) * TEMP_CURIE_SD + TEMP_CURIE_MEAN;
		grain_cu[i] = calcCuFromCurie(grain_tcs[i]);
		double a = curand_log_normal_double(&rand_stat, grain_size_mu, grain_size_sigma);
		grain_area[i] = a * a;
		//grain_ku_kum[i] = 1;
		grain_prov[i] = 1;
	}

	for (int i = -attempt_offset; i < ber_list_count; i++)
	{

		double temp = TEMP_CURIE_MEAN - THERMAL_GRADIENT * LINER_VELOCITY * TAU_AP * 1.0e+9 * i;
		if (temp < TEMP_AMBIENT)
			temp = TEMP_AMBIENT;



		for (int k = 0; k < GRAIN_COUNT; k++)
		{

			if (temp > grain_tcs[k]) continue;

			double kbm, kbp;
			calcKb(temp, hw, grain_cu[k], kbp, kbm);

			double prov_neg = 0 <= i && i < hw_switch_ap ? exp(-kbp * grain_area[k]) : exp(-kbm * grain_area[k]);
			double prov_pog = 0 <= i && i < hw_switch_ap ? exp(-kbm * grain_area[k]) : exp(-kbp * grain_area[k]);
			grain_prov[k] = prov_neg * (1 - grain_prov[k]) + (1 - prov_pog) * grain_prov[k];
		}



		double ber = calcPattern(grain_area, grain_prov);
		atomicAdd(&ber_list[i], ber);
	}
}

void calcContinusBitErrorRateHost(double *bER_list, int bER_list_count, double hw)
{
	double *dev_ber_list;
	unsigned long long int random_seed = (unsigned long long int)(time(NULL));

	for (int i = 0; i < bER_list_count; i++)
		bER_list[i] = 0;

	cudaSetDevice(CUDA_DEVICE_NUM);
	cudaMalloc((void**)&dev_ber_list, sizeof(int) * bER_list_count);
	cudaMemcpy(dev_ber_list, bER_list, sizeof(int) * bER_list_count, cudaMemcpyHostToDevice);

	cudaMemcpy(&kRandomSeed, &random_seed, sizeof(unsigned long long int), cudaMemcpyHostToDevice);


	calcContinusBitErrorRateKernel << <CUDA_BLOCK_COUNT, CUDA_THREAD_COUNT >> >(dev_ber_list, bER_list_count, hw);

	auto err = cudaGetLastError();


	cudaDeviceSynchronize();
	cudaMemcpy(bER_list, dev_ber_list, sizeof(int) * bER_list_count, cudaMemcpyDeviceToHost);

	for (int i = 0; i < bER_list_count; i++)
	{
		bER_list[i] = bER_list[i] / BIT_COUNT;
	}
	cudaFree(dev_ber_list);
}

__global__ void calcMidLastBitErrorRateKernel(double *mid_be_list, double *last_be_list, double hw)
{
	int thread_number = threadIdx.x + blockIdx.x * blockDim.x;
	double grain_prov[GRAIN_COUNT];			// グレインの逆方向に向いている確率
	double grain_tcs[GRAIN_COUNT];			// グレインごとのTc
	double grain_cu[GRAIN_COUNT];			// グレインごとのCu組成
	double grain_area[GRAIN_COUNT];			// グレインごとの面積
	//double grain_ku_kum[GRAIN_COUNT];			// グレインごとのKu/Kum
	double grain_size_mu = log((GRAIN_MEAN * GRAIN_MEAN) / sqrt(GRAIN_SD * GRAIN_SD + GRAIN_MEAN * GRAIN_MEAN));  					  // グレインサイズ分散のμ
	double grain_size_sigma = (sqrt(log((GRAIN_SD * GRAIN_SD) / (GRAIN_MEAN * GRAIN_MEAN) + 1)));									  // グレインサイズ分散のσ
	const int hw_switch_ap = (int)(BIT_PITCH / LINER_VELOCITY *1.0e-9 * F0_AP);														  // 書込磁界順方向終了タイミング
	const int attempt_offset = (int)(TEMP_CURIE_MEAN * TEMP_CURIE_SD * 2 / (THERMAL_GRADIENT * LINER_VELOCITY * TAU_AP * 1.0e+9));	  // Tcm以前の予備シミュレーション
	const int last_attempt = hw_switch_ap * 2 + attempt_offset;



	curandStateMRG32k3a rand_stat;			// 乱数ステータス	
	curand_init(kRandomSeed, thread_number, last_attempt * GRAIN_COUNT, &rand_stat);


	for (int i = 0; i < GRAIN_COUNT; i++)
	{
		grain_tcs[i] = curand_normal_double(&rand_stat) * TEMP_CURIE_SD + TEMP_CURIE_MEAN;
		grain_cu[i] = calcCuFromCurie(grain_tcs[i]);
		double a = curand_log_normal_double(&rand_stat, grain_size_mu, grain_size_sigma);
		grain_area[i] = a * a;
		//grain_ku_kum[i] = 1;
	}
	mid_be_list[thread_number] = 0;
	last_be_list[thread_number] = 0;

	for (int i = -attempt_offset; i < last_attempt; i++)
	{
		double signal_power = 0;

		double temp = TEMP_CURIE_MEAN - THERMAL_GRADIENT * LINER_VELOCITY * TAU_AP * 1.0e+9 * i;
		if (temp < TEMP_AMBIENT)
			temp = TEMP_AMBIENT;



		for (int k = 0; k < GRAIN_COUNT; k++)
		{
			if (temp > grain_tcs[k]) continue;

			double kbm, kbp;
			calcKb(temp, hw, grain_cu[k], kbp, kbm);

			double prov_neg = 0 <= i && i < hw_switch_ap ? exp(-kbp * grain_area[k]) : exp(-kbm * grain_area[k]);
			double prov_pog = 0 <= i && i < hw_switch_ap ? exp(-kbm * grain_area[k]) : exp(-kbp * grain_area[k]);
			grain_prov[k] = prov_neg * (1 - grain_prov[k]) + (1 - prov_pog) * grain_prov[k];
		}

		if (i == hw_switch_ap - 1 && READABLE_THRETH > signal_power)
			mid_be_list[thread_number] = calcPattern(grain_area, grain_prov);;
		if (i == last_attempt - 1 && READABLE_THRETH > signal_power)
			last_be_list[thread_number] = calcPattern(grain_area, grain_prov);;
	}

}

void calcMidLastBitErrorRateHost(double *mid_bER, double *last_bER, double hw)
{
	const int list_size = BIT_COUNT;
	double *mid_be_list = (double*)malloc(sizeof(double) * list_size);
	double *last_be_list = (double*)malloc(sizeof(double) * list_size);
	double *dev_mid_be_list = NULL;
	double *dev_last_be_list = NULL;

	cudaSetDevice(CUDA_DEVICE_NUM);
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_mid_be_list, sizeof(double) * list_size));
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_last_be_list, sizeof(double) * list_size));

	CUDA_SAFE_CALL(cudaMemcpy(dev_mid_be_list, mid_be_list, sizeof(double) * list_size, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(dev_last_be_list, last_be_list, sizeof(double) * list_size, cudaMemcpyHostToDevice));

	calcMidLastBitErrorRateKernel << <CUDA_BLOCK_COUNT, CUDA_THREAD_COUNT >> >(dev_mid_be_list, dev_last_be_list, hw);
	CUDA_SAFE_CALL(cudaGetLastError());
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	CUDA_SAFE_CALL(cudaMemcpy(mid_be_list, dev_mid_be_list, sizeof(double) * list_size, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(last_be_list, dev_last_be_list, sizeof(double) * list_size, cudaMemcpyDeviceToHost));

	double temp_mid_bER = 0;
	double temp_last_bER = 0;




	
	for (int i = 0; i < list_size; i++)
	{
		temp_mid_bER += mid_be_list[i];
		temp_last_bER += last_be_list[i];
	}

	temp_mid_bER /= list_size;
	temp_last_bER /= list_size;

	*mid_bER = temp_mid_bER;
	*last_bER = temp_last_bER;

	cudaFree(dev_mid_be_list);
	cudaFree(dev_last_be_list);
	free(mid_be_list);
	free(last_be_list);

}

void calcHwList(FILE *fp)
{
	fprintf(fp, "Hw[kOe]\tbER\tbER(WE)\tbER(EAW)\n");
	for (int i = 0; i < HW_LIST_SIZE; i++)
	{
		printf("%d / %d \r", i, HW_LIST_SIZE);

		double hw = HW_MAX * i / HW_LIST_SIZE;
		double mid_bER = 0;
		double last_bER = 0;
		calcMidLastBitErrorRateHost(&mid_bER, &last_bER, hw);

		fprintf(fp, "%f\t%.10e\t%.10e\t%.10e\n", hw * 1e-3, last_bER, mid_bER, last_bER - mid_bER);
	}

}




#endif


int main()
{
	//showParameters();
	/*
	const int kbListSize = 150;
	double kbList[kbListSize];

	calcKbListHost(kbList, kbListSize,10e+3);
	for (int i = 0; i < kbListSize; i++)
		printf("%f\n", kbList[i]);
	*/

	
	/*
	double bER_list[200];
	double bER_list_count = 200;

	calcContinusBitErrorRateHost(bER_list, bER_list_count, CBER_HW);
	for (int i = 0; i < bER_list_count; i++)
	{
		printf("%f \n", bER_list[i]);
	}
	*/

	system("time /t");
	FILE *fp = fopen("hw_list.txt", "w");
	calcHwList(fp);
	fclose(fp);
	system("time /t");

	system("pause");
    return 0;
}
