#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <device_atomic_functions.h>
#include <curand.h>
#include <curand_kernel.h>
#include <cuda_profiler_api.h>
#include <vector_functions.h>
#include <thrust/reduce.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <stdio.h>
#include <time.h>

#include <stdio.h>
#include <stdlib.h>
#include <chrono>


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

__device__ inline double convertTempFromAP(int ap_count)
{
	//return fmax(TEMP_AMBIENT, TEMP_CURIE_MEAN - THERMAL_GRADIENT * LINER_VELOCITY * TAU_AP * 1.0e+9 * ap_count);
	double temp = TEMP_CURIE_MEAN - THERMAL_GRADIENT * LINER_VELOCITY * TAU_AP * 1.0e+9 * ap_count;
	if (temp < TEMP_AMBIENT)
		return TEMP_AMBIENT;
	else
		return temp;
}


__device__ inline double calcBrillouin(double s, double x)
{

	double o2 = 1 / (2 * S_FE);
	double o1 = (2 * S_FE + 1) * o2;
	return o1 / tanh(o1 * x) - o2 / tanh(o2 * x);
}

__device__ double calc_sFe_Mean(double temp, double cu)
{
	double vl = S_FE_MEAN_ERROR;
	double vr = S_FE_MEAN_MAX;
	double v;
	double dxdv = (S_FE * 2 * J_FE_FE * 4 * (1 - cu) * (G_FE - 1)*(G_FE - 1)) / (K_B * temp);
	do
	{
		v = (vl + vr) / 2.0;
		double br = S_FE *(2 * S_FE + 1) / (2 * S_FE) / tanh((2 * S_FE + 1) / (2 * S_FE) * dxdv * v) - S_FE  / (2 * S_FE) / tanh(1 / (2 * S_FE) * dxdv * v);
		double f = v - br;
		//double f = v - S_FE * calcBrillouin(S_FE, dxdv * v);
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

__device__ void calcKb(double temp, double hw, double cu, double tc, double &kbp, double &kbm)
{
	kbm = kbp = 0;
	if (tc <= temp) return;

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
		double temp = convertTempFromAP(i);

		kb_list[i] = calcKb(temp, hw, cu);
	}

}




#if BER_ALGORITHM == 0


// ###
// ###  単純モンテカルロ法
// ###
// #########################################################################################################################


__global__ void calcContinusBitErrorRateKernel(int *ber_list, int ber_list_count, double hw)
{
	int thread_number = threadIdx.x + blockIdx.x * blockDim.x;
	double grain_tc[GRAIN_COUNT];			// グレインごとのTc
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
		grain_tc[i] = curand_normal_double(&rand_stat) * TEMP_CURIE_SD * TEMP_CURIE_MEAN + TEMP_CURIE_MEAN;
		grain_cu[i] = calcCuFromCurie(grain_tc[i]);
		double a = curand_log_normal_double(&rand_stat, grain_size_mu, grain_size_sigma);
		grain_area[i] = a * a;
		//grain_ku_kum[i] = 1;
		grain_dir[i] = 1;
	}

	double signed_hw = hw;
	for (int i = -attempt_offset; i < ber_list_count; i++)
	{
		double signal_power = 0;

		double temp = convertTempFromAP(i);

		if (i == 0 || i == hw_switch_ap)
			signed_hw = -signed_hw;


		for (int k = 0; k < GRAIN_COUNT; k++)
		{
			// 磁化反転する確率
			// hw = -1 = 順方向
			// grain_dir = 1 逆方向
			// hw * grain_dir = -1 hw方向への反転確率
			if (temp > grain_tc[k]) continue;

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

	CUDA_SAFE_CALL(cudaSetDevice(CUDA_DEVICE_NUM));
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_be_list, sizeof(int) * bER_list_count));
	CUDA_SAFE_CALL(cudaMemcpy(dev_be_list, be_list, sizeof(int) * bER_list_count, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(&kRandomSeed, &random_seed, sizeof(unsigned long long int), cudaMemcpyHostToDevice));


	calcContinusBitErrorRateKernel << <CUDA_BLOCK_COUNT, CUDA_THREAD_COUNT >> >(dev_be_list, bER_list_count, hw);
	CUDA_SAFE_CALL(cudaGetLastError());
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	CUDA_SAFE_CALL(cudaMemcpy(be_list, dev_be_list, sizeof(int) * bER_list_count, cudaMemcpyDeviceToHost));

	for (int i = 0; i < bER_list_count; i++)
	{
		bER_list[i] = (double)be_list[i] / BIT_COUNT;
	}
	CUDA_SAFE_CALL(cudaFree(dev_be_list));
	free(be_list);
}

__global__ void calcMidLastBitErrorRateKernel(int *mid_be_list, int *last_be_list, double hw)
{
	int thread_number = threadIdx.x + blockIdx.x * blockDim.x;
	double grain_tc[GRAIN_COUNT];			// グレインごとのTc
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
		grain_tc[i] = curand_normal_double(&rand_stat) * TEMP_CURIE_SD * TEMP_CURIE_MEAN + TEMP_CURIE_MEAN;
		grain_cu[i] = calcCuFromCurie(grain_tc[i]);
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

		double temp = convertTempFromAP(i);

		if (i == 0 || i == hw_switch_ap)
			signed_hw = -signed_hw;

		for (int k = 0; k < GRAIN_COUNT; k++)
		{
			// 磁化反転する確率
			// hw = -1 = 順方向
			// grain_dir = 1 逆方向
			// hw * grain_dir = -1 hw方向への反転確率
			if (temp > grain_tc[k]) continue;

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

	CUDA_SAFE_CALL(cudaSetDevice(CUDA_DEVICE_NUM));
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_mid_be_list, sizeof(int) * list_size));
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_last_be_list, sizeof(int) * list_size));

	CUDA_SAFE_CALL(cudaMemcpy(dev_mid_be_list, mid_be_list, sizeof(int) * list_size, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(dev_last_be_list, last_be_list, sizeof(int) * list_size, cudaMemcpyHostToDevice));

	calcMidLastBitErrorRateKernel << <CUDA_BLOCK_COUNT, CUDA_THREAD_COUNT >> >(dev_mid_be_list, dev_last_be_list, hw);
	CUDA_SAFE_CALL(cudaGetLastError());
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	CUDA_SAFE_CALL(cudaMemcpy(mid_be_list, dev_mid_be_list, sizeof(int) * list_size, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(last_be_list, dev_last_be_list, sizeof(int) * list_size, cudaMemcpyDeviceToHost));

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

	CUDA_SAFE_CALL(cudaFree(dev_mid_be_list));
	CUDA_SAFE_CALL(cudaFree(dev_last_be_list));
	free(mid_be_list);
	free(last_be_list);

}



#else

// ###
// ###  確率　・　パターン法
// ###
// #########################################################################################################################

__device__ inline double calcPattern(double *grain_area, double *grain_prob)
{
	double bit_error_rate = 0;
#define GA_(n) (0)
#define GAM(n) (grain_area[n])
#define GP_(n) (grain_prob[n])
#define GPM(n) (1 - grain_prob[n])

#if GRAIN_COUNT == 1
	if (grain_area[0] < READABLE_THRETH) bit_error_rate = grain_prob[0];

#elif GRAIN_COUNT == 4
	if (GA_(0) + GA_(1) + GA_(2) + GA_(3) < READABLE_THRETH) bit_error_rate += GP_(0) * GP_(1) * GP_(2) * GP_(3);  //  0
	if (GAM(0) + GA_(1) + GA_(2) + GA_(3) < READABLE_THRETH) bit_error_rate += GPM(0) * GP_(1) * GP_(2) * GP_(3);  //  1
	if (GA_(0) + GAM(1) + GA_(2) + GA_(3) < READABLE_THRETH) bit_error_rate += GP_(0) * GPM(1) * GP_(2) * GP_(3);  //  2
	if (GAM(0) + GAM(1) + GA_(2) + GA_(3) < READABLE_THRETH) bit_error_rate += GPM(0) * GPM(1) * GP_(2) * GP_(3);  //  3
	if (GA_(0) + GA_(1) + GAM(2) + GA_(3) < READABLE_THRETH) bit_error_rate += GP_(0) * GP_(1) * GPM(2) * GP_(3);  //  4
	if (GAM(0) + GA_(1) + GAM(2) + GA_(3) < READABLE_THRETH) bit_error_rate += GPM(0) * GP_(1) * GPM(2) * GP_(3);  //  5
	if (GA_(0) + GAM(1) + GAM(2) + GA_(3) < READABLE_THRETH) bit_error_rate += GP_(0) * GPM(1) * GPM(2) * GP_(3);  //  6
	if (GAM(0) + GAM(1) + GAM(2) + GA_(3) < READABLE_THRETH) bit_error_rate += GPM(0) * GPM(1) * GPM(2) * GP_(3);  //  7
	if (GA_(0) + GA_(1) + GA_(2) + GAM(3) < READABLE_THRETH) bit_error_rate += GP_(0) * GP_(1) * GP_(2) * GPM(3);  //  8
	if (GAM(0) + GA_(1) + GA_(2) + GAM(3) < READABLE_THRETH) bit_error_rate += GPM(0) * GP_(1) * GP_(2) * GPM(3);  //  9
	if (GA_(0) + GAM(1) + GA_(2) + GAM(3) < READABLE_THRETH) bit_error_rate += GP_(0) * GPM(1) * GP_(2) * GPM(3);  // 10
	if (GAM(0) + GAM(1) + GA_(2) + GAM(3) < READABLE_THRETH) bit_error_rate += GPM(0) * GPM(1) * GP_(2) * GPM(3);  // 11
	if (GA_(0) + GA_(1) + GAM(2) + GAM(3) < READABLE_THRETH) bit_error_rate += GP_(0) * GP_(1) * GPM(2) * GPM(3);  // 12
	if (GAM(0) + GA_(1) + GAM(2) + GAM(3) < READABLE_THRETH) bit_error_rate += GPM(0) * GP_(1) * GPM(2) * GPM(3);  // 13
	if (GA_(0) + GAM(1) + GAM(2) + GAM(3) < READABLE_THRETH) bit_error_rate += GP_(0) * GPM(1) * GPM(2) * GPM(3);  // 14
	if (GAM(0) + GAM(1) + GAM(2) + GAM(3) < READABLE_THRETH) bit_error_rate += GPM(0) * GPM(1) * GPM(2) * GPM(3);  // 15
#else
#error Not implement for this GRAIN_COUNT pattern


#endif
	
#undef GA_
#undef GAM
#undef GP_
#undef GPM
	

	return bit_error_rate;
}

__global__ void calcContinusBitErrorRateKernel(double *ber_list, double ber_list_count, double hw)
{

	int thread_number = threadIdx.x + blockIdx.x * blockDim.x;
	double grain_prob[GRAIN_COUNT];				// グレインの逆方向に向いている確率
	double grain_tc[GRAIN_COUNT];				// グレインごとのTc
	double grain_cu[GRAIN_COUNT];				// グレインごとのCu組成
	double grain_area[GRAIN_COUNT];				// グレインごとの面積
	//double grain_ku_kum[GRAIN_COUNT];			// グレインごとのKu/Kum
	double grain_size_mu = log((GRAIN_MEAN * GRAIN_MEAN) / sqrt(GRAIN_SD * GRAIN_SD + GRAIN_MEAN * GRAIN_MEAN));  					  // グレインサイズ分散のμ
	double grain_size_sigma = (sqrt(log((GRAIN_SD * GRAIN_SD) / (GRAIN_MEAN * GRAIN_MEAN) + 1)));									  // グレインサイズ分散のσ
	const int hw_switch_ap = (int)(BIT_PITCH / LINER_VELOCITY *1.0e-9 * F0_AP);														  // 書込磁界順方向終了タイミング
	const int attempt_offset = (int)(TEMP_CURIE_MEAN * TEMP_CURIE_SD * 2 / (THERMAL_GRADIENT * LINER_VELOCITY * TAU_AP * 1.0e+9));	  // Tcm以前の予備シミュレーション


	curandStateMRG32k3a rand_stat;			// 乱数ステータス	
	curand_init(kRandomSeed, thread_number, ber_list_count * GRAIN_COUNT, &rand_stat);


	for (int i = 0; i < GRAIN_COUNT; i++)
	{
		grain_tc[i] = curand_normal_double(&rand_stat) * TEMP_CURIE_SD * TEMP_CURIE_MEAN + TEMP_CURIE_MEAN;
		grain_cu[i] = calcCuFromCurie(grain_tc[i]);
		double a = curand_log_normal_double(&rand_stat, grain_size_mu, grain_size_sigma);
		grain_area[i] = a * a;
		//grain_ku_kum[i] = 1;
		grain_prob[i] = INITIAL_MAG_PROB;
	}


	for (int i = -attempt_offset; i < ber_list_count; i++)
	{

		double temp = TEMP_CURIE_MEAN - THERMAL_GRADIENT * LINER_VELOCITY * TAU_AP * 1.0e+9 * i;
		if (temp < TEMP_AMBIENT)
			temp = TEMP_AMBIENT;



		for (int k = 0; k < GRAIN_COUNT; k++)
		{

			if (temp > grain_tc[k]) continue;

			double kbm, kbp;
			calcKb(temp, hw, grain_cu[k],grain_tc[k], kbp, kbm);

			double prob_neg = 0 <= i && i < hw_switch_ap ? exp(-kbp * grain_area[k]) : exp(-kbm * grain_area[k]);
			double prob_pog = 0 <= i && i < hw_switch_ap ? exp(-kbm * grain_area[k]) : exp(-kbp * grain_area[k]);
			grain_prob[k] = prob_neg * (1 - grain_prob[k]) + (1 - prob_pog) * grain_prob[k];
		}



		double ber = calcPattern(grain_area, grain_prob);
		atomicAdd(&ber_list[i], ber);
	}
}

void calcContinusBitErrorRateHost(double *bER_list, int bER_list_count, double hw)
{
	double *dev_ber_list;
	unsigned long long int random_seed = (unsigned long long int)(time(NULL));

	for (int i = 0; i < bER_list_count; i++)
		bER_list[i] = 0;

	CUDA_SAFE_CALL(cudaSetDevice(CUDA_DEVICE_NUM));
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_ber_list, sizeof(int) * bER_list_count));
	CUDA_SAFE_CALL(cudaMemcpy(dev_ber_list, bER_list, sizeof(int) * bER_list_count, cudaMemcpyHostToDevice));

	//CUDA_SAFE_CALL(cudaMemcpy(&kRandomSeed, &random_seed, sizeof(unsigned long long int), cudaMemcpyHostToDevice));
	
	calcContinusBitErrorRateKernel <<< CUDA_BLOCK_COUNT, CUDA_THREAD_COUNT >>>(dev_ber_list, bER_list_count, hw);

	CUDA_SAFE_CALL(cudaGetLastError());


	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	CUDA_SAFE_CALL(cudaMemcpy(bER_list, dev_ber_list, sizeof(int) * bER_list_count, cudaMemcpyDeviceToHost));

	for (int i = 0; i < bER_list_count; i++)
	{
		bER_list[i] = bER_list[i] / BIT_COUNT;
	}
	CUDA_SAFE_CALL(cudaFree(dev_ber_list));
}

__global__ void calcMidLastBitErrorRateKernel(double *mid_be_list, double *last_be_list, double hw)
{

	int thread_number = threadIdx.x + blockIdx.x * blockDim.x;
	double grain_prob[GRAIN_COUNT];			// グレインの逆方向に向いている確率
	double grain_tc[GRAIN_COUNT];			// グレインごとのTc
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
		grain_tc[i] = curand_normal_double(&rand_stat) * TEMP_CURIE_SD * TEMP_CURIE_MEAN + TEMP_CURIE_MEAN;
		grain_cu[i] = calcCuFromCurie(grain_tc[i]);
		double a = curand_log_normal_double(&rand_stat, grain_size_mu, grain_size_sigma);
		grain_area[i] = a * a;
		grain_prob[i] = INITIAL_MAG_PROB;
		//grain_ku_kum[i] = 1;
	}
	mid_be_list[thread_number] = 0;
	last_be_list[thread_number] = 0;

	for (int i = -attempt_offset; i < last_attempt; i++)
	{

		double temp = TEMP_CURIE_MEAN - THERMAL_GRADIENT * LINER_VELOCITY * TAU_AP * 1.0e+9 * i;
		if (temp < TEMP_AMBIENT)
			temp = TEMP_AMBIENT;

		for (int k = 0; k < GRAIN_COUNT; k++)
		{
			if (temp > grain_tc[k]) continue;

			double kbm, kbp;
			calcKb(temp, hw, grain_cu[k],grain_tc[k], kbp, kbm);
			//kbp = calcKb(temp, hw, grain_cu[k]);
			//kbm = calcKb(temp, -hw, grain_cu[k]);

			double prob_neg = 0 <= i && i < hw_switch_ap ? exp(-kbp * grain_area[k]) : exp(-kbm * grain_area[k]);
			double prob_pog = 0 <= i && i < hw_switch_ap ? exp(-kbm * grain_area[k]) : exp(-kbp * grain_area[k]);
			grain_prob[k] = prob_neg * (1 - grain_prob[k]) + (1 - prob_pog) * grain_prob[k];
		}

		if (i == hw_switch_ap - 1) // EAW用に中抜き
			mid_be_list[thread_number] = calcPattern(grain_area, grain_prob);
			
	}

	last_be_list[thread_number] = calcPattern(grain_area, grain_prob);

}

void calcMidLastBitErrorRateHost(double *mid_bER, double *last_bER, double hw)
{
	const int list_size = BIT_COUNT;
	double *mid_be_list = (double*)malloc(sizeof(double) * list_size);
	double *last_be_list = (double*)malloc(sizeof(double) * list_size);
	double *dev_mid_be_list = NULL;
	double *dev_last_be_list = NULL;
	unsigned long long int random_seed = (unsigned long long int)(time(NULL));

	CUDA_SAFE_CALL(cudaSetDevice(CUDA_DEVICE_NUM));
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_mid_be_list, sizeof(double) * list_size));
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_last_be_list, sizeof(double) * list_size));

	CUDA_SAFE_CALL(cudaMemcpy(dev_mid_be_list, mid_be_list, sizeof(double) * list_size, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(dev_last_be_list, last_be_list, sizeof(double) * list_size, cudaMemcpyHostToDevice));
	//CUDA_SAFE_CALL(cudaMemcpyToSymbol(&kRandomSeed, &random_seed, sizeof(unsigned long long int)));


	calcMidLastBitErrorRateKernel <<<CUDA_BLOCK_COUNT, CUDA_THREAD_COUNT>>>(dev_mid_be_list, dev_last_be_list, hw);
	CUDA_SAFE_CALL(cudaGetLastError());
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	CUDA_SAFE_CALL(cudaMemcpy(mid_be_list, dev_mid_be_list, sizeof(double) * list_size, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(last_be_list, dev_last_be_list, sizeof(double) * list_size, cudaMemcpyDeviceToHost));

	
	double temp_mid_bER = 0;
	double temp_last_bER = 0;

	/*// ニコイチリダクション、高精度？
	double half = list_size / 2;
	for (int i = 0; i < log2(list_size)-1; i++)
	{
		for (int k = 0; k < half-1; k++)
		{
			mid_be_list[k] = (mid_be_list[2 * k] + mid_be_list[2 * k + 1]) / 2.0;
			last_be_list[k] = (last_be_list[2 * k] + last_be_list[2 * k + 1]) / 2.0;
		}
		half /= 2.0;
	}

	temp_mid_bER = mid_be_list[0];
	temp_last_bER = last_be_list[0];
	*/


	
	// 単純リダクション
	for (int i = 0; i < list_size; i++)
	{
		temp_mid_bER += mid_be_list[i];
		temp_last_bER += last_be_list[i];
	}

	temp_mid_bER /= list_size;
	temp_last_bER /= list_size;
	
	

	/*
	// thrust GPUリダクション
	thrust::device_ptr<double> dev_mid_be_ptr(dev_mid_be_list);
	thrust::device_ptr<double> dev_last_be_ptr(dev_last_be_list);

	temp_mid_bER = thrust::reduce(dev_mid_be_ptr, dev_mid_be_ptr + list_size);
	temp_last_bER = thrust::reduce(dev_last_be_ptr, dev_last_be_ptr + list_size);
	temp_mid_bER /= list_size;
	temp_last_bER /= list_size;
	*/

	*mid_bER = temp_mid_bER;
	*last_bER = temp_last_bER;

	CUDA_SAFE_CALL(cudaFree(dev_mid_be_list));
	CUDA_SAFE_CALL(cudaFree(dev_last_be_list));
	free(mid_be_list);
	free(last_be_list);

}

#endif



void makeHwBerList(FILE *fp)
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

void makeContinusBerList(FILE *fp)
{
	fprintf(stderr, "not implement at line %d", __LINE__);
	exit(2);
}


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



	auto start = std::chrono::system_clock::now();
#if (BER_ALGORITHM == 1)

	FILE *fp = fopen("hw_list_prob.txt", "w");
#else
	FILE *fp = fopen("hw_list_pure.txt", "w");

#endif
	makeHwBerList(fp);
	fclose(fp);


	cudaProfilerStart();
	auto end = std::chrono::system_clock::now();
	auto dur = end - start;
	auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();

	std::cout <<"\n"<< msec << " milli sec \n";



	cudaProfilerStop();
	
	//system("pause");
    return 0;
}
