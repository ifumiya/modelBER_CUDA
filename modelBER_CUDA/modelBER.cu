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


__constant__ unsigned long long int kRandomSeed[1];		/// <summary>乱数のシード値</summary>


// ##########################################################################################
// # CUDA 用ユーティリティ


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

// ##########################################################################################
// # ロジック (アルゴリズム共通部分)


/// <summary>Cu組成からキュリー温度を算出します</summary>
/// <param name='cu'> Cu組成</param> 
/// <returns>キュリー温度 (K) </returns>
__device__ inline float calcCurieFromCu(float cu)
{
	return (2 * J_FE_FE * 4 * (1 - cu) * (G_FE - 1)  * (G_FE - 1) * S_FE * (S_FE + 1)) / (3 * K_B);
}

/// <summary>　キュリー温度からをCu組成算出します</summary>
/// <param name='temp_curie'> キュリー温度 (K)</param> 
/// <returns>Cu組成</returns>
__host__ __device__ inline float calcCuFromCurie(float temp_curie)
{
	return -temp_curie * (3 * K_B) / (2 * J_FE_FE * 4 * (G_FE - 1) *(G_FE - 1) * S_FE * (S_FE + 1)) + 1;
}

/// <summary>試行回数から温度を算出します</summary>
/// <param name='ap_count'> 試行回数 </param> 
/// <returns>温度</returns>
__host__ __device__ inline float convertTempFromAP(int ap_count)
{

	float temp = TEMP_CURIE_MEAN - THERMAL_GRADIENT * LINER_VELOCITY * TAU_AP * 1.0e+9F * ap_count;
	return temp < TEMP_AMBIENT ? TEMP_AMBIENT : temp;
}

__host__ __device__ inline float convertTimeFromTemp(float temp)
{
	return (TEMP_CURIE_MEAN - temp) / THERMAL_GRADIENT / LINER_VELOCITY;
}


__host__ __device__ inline float calcBrillouin(float s, float x)
{

	float o2 = 1 / (2 * S_FE);
	float o1 = (2 * S_FE + 1) * o2;
	return o1 / tanh(o1 * x) - o2 / tanh(o2 * x);
	
}

__host__ __device__ float calc_sFe_Mean(float temp, float cu)
{
	float vl = S_FE_MEAN_ERROR;
	float vr = S_FE_MEAN_MAX;
	float v;
	float dxdv = (S_FE * 2 * J_FE_FE * 4 * (1 - cu) * (G_FE - 1)*(G_FE - 1)) / (K_B * temp);
	do
	{
		v = (vl + vr) / 2.0F;
		float br = S_FE *(2 * S_FE + 1) / (2 * S_FE) / tanh((2 * S_FE + 1) / (2 * S_FE) * dxdv * v) - S_FE  / (2 * S_FE) / tanh(1 / (2 * S_FE) * dxdv * v);
		float f = v - br;
		//float f = v - S_FE * calcBrillouin(S_FE, dxdv * v);
		if (f < 0)
			vl = v;
		else
			vr = v;

	} while (fabs((vl - vr) / v) > S_FE_MEAN_ERROR);
	
	return v;
}

__device__ float calcKb(float temp, float hw, float cu)
{

	float temp_curie = calcCurieFromCu(cu);
	if (temp >= temp_curie) return 0;

	float dFeFe = BULK_D_FE_FE * KU_KBULK;

	float s = calc_sFe_Mean(temp, cu);

	float total_atom_number = 1 / ((FE  * (1 - cu) * V_FE) + ((1 - FE)*(1 - cu) * V_PT) + cu * V_CU);
	float ku = total_atom_number * FE * (1 - cu)* (4 * (1 - cu))  * dFeFe  * s * s;
	float hc = 2 * ((4 * (1 - cu))  * dFeFe  * s) / (M_B * G_FE);
	if (hw < 0 && hc <= fabs(hw)) return 0;

	float kb = (ku * GRAIN_VOLUME) / K_B / temp * (1 + hw / hc) * (1 + hw / hc);
	return kb;
}

__device__ void calcKb(float temp, float hw, float cu, float tc, float &kbp, float &kbm)
{
	kbm = kbp = 0;
	if (tc <= temp) return;

	const float dFeFe = BULK_D_FE_FE * KU_KBULK;

	float s = calc_sFe_Mean(temp, cu);
	float total_atom_number = 1 / ((FE  * (1 - cu) * V_FE) + ((1 - FE)*(1 - cu) * V_PT) + cu * V_CU);
	float ku = total_atom_number * FE * (1 - cu)* (4 * (1 - cu))  * dFeFe  * s * s;
	float hc = 2 * ((4 * (1 - cu))  * dFeFe  * s) / (M_B * G_FE);
	kbp = (ku * GRAIN_VOLUME) / K_B / temp * (1 + hw / hc) * (1 + hw / hc);
	kbm = hc <= hw ? 0 : (ku * GRAIN_VOLUME) / K_B / temp * (1 - hw / hc) * (1 - hw / hc);
}

__global__
void calcKbListKernel(float *kb_minus_list, float *kb_plus_list, int kb_list_size, int offset, float hw)
{
	int i = threadIdx.x + blockDim.x * blockIdx.x;
	if (i >= kb_list_size) return;

	float cu = calcCuFromCurie(TEMP_CURIE_MEAN);
	float temp = convertTempFromAP(i - offset);
	float kbp, kbm;
	calcKb(temp, hw, cu, TEMP_CURIE_MEAN, kbp, kbm);
	kb_plus_list[i] = kbp;
	kb_minus_list[i] = kbm;
}

__host__
void calcKbListHost(FILE *fp, float hw)
{
	int bit_ap = (int)(BIT_PITCH * 1e-9 / TAU_AP / LINER_VELOCITY);
	int kb_list_count = bit_ap * 4;
	int offset = bit_ap;

	int thread_count = (int)(fmax(sqrt(kb_list_count),THREAD_NUM));
	int block_count = (kb_list_count / thread_count + 1);

	CUDA_SAFE_CALL(cudaSetDevice(CUDA_DEVICE_NUM));

	thrust::host_vector<float> host_kb_m_list(kb_list_count);
	thrust::host_vector<float> host_kb_p_list(kb_list_count);
	thrust::device_vector<float> dev_kb_m_list(kb_list_count);
	thrust::device_vector<float> dev_kb_p_list(kb_list_count);

	calcKbListKernel << <thread_count, block_count >> >(
		thrust::raw_pointer_cast(dev_kb_m_list.data()),
		thrust::raw_pointer_cast(dev_kb_p_list.data()),
		kb_list_count,
		offset,
		hw);

	CUDA_SAFE_CALL(cudaGetLastError());
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	
	host_kb_m_list = dev_kb_m_list;
	host_kb_p_list = dev_kb_p_list;


	fprintf(fp , "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n",
		"AP Count",
		"Temp(K)",
		"Time(ns)",
		"Distance(nm)",
		"Kb+",
		"Kb-",
		"p+",
		"p-");
	for (int i = 0; i < kb_list_count; i++)
	{
		int ap = i - offset;
		fprintf(fp, "%d\t%e\t%e\t%e\t%e\t%e\t%e\t%e\n",
			ap,
			convertTempFromAP(i - offset),
			ap * TAU_AP * 1e+9,
			ap * TAU_AP * 1e+9 * LINER_VELOCITY,
			host_kb_m_list[i],
			host_kb_p_list[i],
			exp(-host_kb_m_list[i]),
			exp(-host_kb_p_list[i]));

	}

}

float calcRecordingTimeWindow(float hw)
{
	float dFeFe = BULK_D_FE_FE * KU_KBULK;
	float cu = calcCuFromCurie(TEMP_CURIE_MEAN);

	float temp_low = TEMP_AMBIENT;
	float temp_high = TEMP_CURIE_MEAN;
	float temp = (temp_low + temp_high) / 2.0F;

	// TODO : S_FE_MEAN_ERRORの誤用
	while (fabs((temp_high - temp_low) / temp) > S_FE_MEAN_ERROR)
	{
		
		float s = calc_sFe_Mean(temp, cu);
		float hc = 2 * ((4 * (1 - cu))  * dFeFe  * s) / (M_B * G_FE);

		if (hc < hw)
			temp_high = temp;
		else
			temp_low = temp;

		temp = (temp_high + temp_low) / 2.0F;
	}

	return convertTimeFromTemp(temp);
}




#if BER_ALGORITHM == 0


// ###
// ###  単純モンテカルロ法
// ###
// #########################################################################################################################


__global__ void calcContinusBitErrorRateKernel(int *ber_list, int ber_list_count, float hw)
{
	int thread_number = threadIdx.x + blockIdx.x * blockDim.x;
	float grain_tc[GRAIN_COUNT];			// グレインごとのTc
	float grain_cu[GRAIN_COUNT];			// グレインごとのCu組成
	float grain_area[GRAIN_COUNT];			// グレインごとの面積
	//float grain_ku_kum[GRAIN_COUNT];			// グレインごとのKu/Kum
	int grain_dir[GRAIN_COUNT];				// グレインの磁化の向き (1 = 逆方向、-1 = 順方向)
	float grain_size_mu = log((GRAIN_MEAN * GRAIN_MEAN) / sqrt(GRAIN_SD * GRAIN_SD + GRAIN_MEAN * GRAIN_MEAN));  					  // グレインサイズ分散のμ
	float grain_size_sigma = (sqrt(log((GRAIN_SD * GRAIN_SD) / (GRAIN_MEAN * GRAIN_MEAN) + 1)));									  // グレインサイズ分散のσ
	const int hw_switch_ap = (int)(BIT_PITCH / LINER_VELOCITY *1.0e-9 * F0_AP);														  // 書込磁界順方向終了タイミング
	const int attempt_offset = (int)(TEMP_CURIE_MEAN * TEMP_CURIE_SD * 2 / (THERMAL_GRADIENT * LINER_VELOCITY * TAU_AP * 1.0e+9));	  // Tcm以前の予備シミュレーション


	curandStateMRG32k3a rand_stat;			// 乱数ステータス	
	curand_init(*kRandomSeed, thread_number, ber_list_count * GRAIN_COUNT, &rand_stat);


	for (int i = 0; i < GRAIN_COUNT; i++)
	{
		grain_tc[i] = curand_normal_double(&rand_stat) * TEMP_CURIE_SD * TEMP_CURIE_MEAN + TEMP_CURIE_MEAN;
		grain_cu[i] = calcCuFromCurie(grain_tc[i]);
		float a = curand_log_normal_double(&rand_stat, grain_size_mu, grain_size_sigma);
		grain_area[i] = a * a;
		//grain_ku_kum[i] = 1;
		grain_dir[i] = 1;
	}

	float signed_hw = hw;
	for (int i = -attempt_offset; i < ber_list_count; i++)
	{
		float signal_power = 0;

		float temp = convertTempFromAP(i);

		if (i == 0 || i == hw_switch_ap)
			signed_hw = -signed_hw;


		for (int k = 0; k < GRAIN_COUNT; k++)
		{
			// 磁化反転する確率
			// hw = -1 = 順方向
			// grain_dir = 1 逆方向
			// hw * grain_dir = -1 hw方向への反転確率
			if (temp > grain_tc[k]) continue;

			float rev_prob = exp(-calcKb(temp, signed_hw * grain_dir[k], grain_cu[k]) * grain_area[k]);
			float dice = curand_uniform(&rand_stat);
			if (rev_prob > dice)
				grain_dir[k] = -grain_dir[k];
			if (grain_dir[k] < 0)
				signal_power += grain_area[k];
		}

		if (READABLE_THRETH > signal_power && 0 <= i && i < ber_list_count)
			atomicAdd(&ber_list[i], 1);
	}
}

void calcContinusBitErrorRateHost(float *bER_list, int bER_list_count, float hw)
{
	int *dev_be_list;
	int *be_list = (int*)malloc(sizeof(int) * bER_list_count);
	unsigned long long int random_seed = (unsigned long long int)(time(NULL));

	for (int i = 0; i < bER_list_count; i++)
		be_list[i] = 0;

	CUDA_SAFE_CALL(cudaSetDevice(CUDA_DEVICE_NUM));
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_be_list, sizeof(int) * bER_list_count));
	CUDA_SAFE_CALL(cudaMemcpy(dev_be_list, be_list, sizeof(int) * bER_list_count, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(*kRandomSeed, &random_seed, sizeof(unsigned long long int), cudaMemcpyHostToDevice));


	calcContinusBitErrorRateKernel << <CUDA_BLOCK_COUNT, CUDA_THREAD_COUNT >> >(dev_be_list, bER_list_count, hw);
	CUDA_SAFE_CALL(cudaGetLastError());
	CUDA_SAFE_CALL(cudaDeviceSynchronize());

	CUDA_SAFE_CALL(cudaMemcpy(be_list, dev_be_list, sizeof(int) * bER_list_count, cudaMemcpyDeviceToHost));

	for (int i = 0; i < bER_list_count; i++)
	{
		bER_list[i] = (float)be_list[i] / BIT_COUNT;
	}
	CUDA_SAFE_CALL(cudaFree(dev_be_list));
	free(be_list);
}

__global__ void calcMidLastBitErrorRateKernel(int *mid_be_list, int *last_be_list, float hw)
{
	int thread_number = threadIdx.x + blockIdx.x * blockDim.x;
	float grain_tc[GRAIN_COUNT];			// グレインごとのTc
	float grain_cu[GRAIN_COUNT];			// グレインごとのCu組成
	float grain_area[GRAIN_COUNT];			// グレインごとの面積
	//float grain_ku_kum[GRAIN_COUNT];			// グレインごとのKu/Kum
	int grain_dir[GRAIN_COUNT];				// グレインの磁化の向き (1 = 逆方向、-1 = 順方向)
	float grain_size_mu = log((GRAIN_MEAN * GRAIN_MEAN) / sqrt(GRAIN_SD * GRAIN_SD + GRAIN_MEAN * GRAIN_MEAN));  					  // グレインサイズ分散のμ
	float grain_size_sigma = (sqrt(log((GRAIN_SD * GRAIN_SD) / (GRAIN_MEAN * GRAIN_MEAN) + 1)));									  // グレインサイズ分散のσ
	const int hw_switch_ap = (int)(BIT_PITCH / LINER_VELOCITY *1.0e-9 * F0_AP);														  // 書込磁界順方向終了タイミング
	const int attempt_offset = (int)(TEMP_CURIE_MEAN * TEMP_CURIE_SD * 2 / (THERMAL_GRADIENT * LINER_VELOCITY * TAU_AP * 1.0e+9));	  // Tcm以前の予備シミュレーション
	const int last_attempt = hw_switch_ap * 2 + attempt_offset;



	curandStateMRG32k3a rand_stat;			// 乱数ステータス	
	curand_init(*kRandomSeed, thread_number, last_attempt * GRAIN_COUNT, &rand_stat);


	for (int i = 0; i < GRAIN_COUNT; i++)
	{
		grain_tc[i] = curand_normal_double(&rand_stat) * TEMP_CURIE_SD * TEMP_CURIE_MEAN + TEMP_CURIE_MEAN;
		grain_cu[i] = calcCuFromCurie(grain_tc[i]);
		float a = curand_log_normal_double(&rand_stat, grain_size_mu, grain_size_sigma);
		grain_area[i] = a * a;
		//grain_ku_kum[i] = 1;
		grain_dir[i] = 1;
	}
	mid_be_list[thread_number] = 0;
	last_be_list[thread_number] = 0;

	float signed_hw = hw;
	for (int i = -attempt_offset; i < last_attempt; i++)
	{
		float signal_power = 0;

		float temp = convertTempFromAP(i);

		if (i == 0 || i == hw_switch_ap)
			signed_hw = -signed_hw;

		for (int k = 0; k < GRAIN_COUNT; k++)
		{
			// 磁化反転する確率
			// hw = -1 = 順方向
			// grain_dir = 1 逆方向
			// hw * grain_dir = -1 hw方向への反転確率
			if (temp > grain_tc[k]) continue;

			float rev_prob = exp(-calcKb(temp, signed_hw * grain_dir[k], grain_cu[k]) * grain_area[k]);
			float dice = curand_uniform(&rand_stat);
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

void calcMidLastBitErrorRateHost(float *mid_bER, float *last_bER, float hw)
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

	float temp_mid_bER = 0;
	float temp_last_bER = 0;
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

__device__ inline float calcPattern(float *grain_area, float *grain_prob)
{
	float bit_error_rate = 0;
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

__global__ void calcContinusBitErrorRateKernel(float *ber_list, int ber_list_count, float hw)
{

	int thread_number = threadIdx.x + blockIdx.x * blockDim.x;
	float grain_prob[GRAIN_COUNT];				// グレインの逆方向に向いている確率
	float grain_tc[GRAIN_COUNT];				// グレインごとのTc
	float grain_cu[GRAIN_COUNT];				// グレインごとのCu組成
	float grain_area[GRAIN_COUNT];				// グレインごとの面積
	//float grain_ku_kum[GRAIN_COUNT];			// グレインごとのKu/Kum
	float grain_size_mu = log((GRAIN_MEAN * GRAIN_MEAN) / sqrt(GRAIN_SD * GRAIN_SD + GRAIN_MEAN * GRAIN_MEAN));  					  // グレインサイズ分散のμ
	float grain_size_sigma = (sqrt(log((GRAIN_SD * GRAIN_SD) / (GRAIN_MEAN * GRAIN_MEAN) + 1)));									  // グレインサイズ分散のσ
	const int hw_switch_ap = (int)(BIT_PITCH / LINER_VELOCITY *1.0e-9 * F0_AP);														  // 書込磁界順方向終了タイミング
	const int attempt_offset = (int)(TEMP_CURIE_MEAN * TEMP_CURIE_SD * 2 / (THERMAL_GRADIENT * LINER_VELOCITY * TAU_AP * 1.0e+9));	  // Tcm以前の予備シミュレーション


	curandStateMRG32k3a rand_stat;			// 乱数ステータス	
	curand_init(*kRandomSeed, thread_number, ber_list_count * GRAIN_COUNT, &rand_stat);


	for (int i = 0; i < GRAIN_COUNT; i++)
	{
		grain_tc[i] = curand_normal (&rand_stat) * TEMP_CURIE_SD * TEMP_CURIE_MEAN + TEMP_CURIE_MEAN;
		grain_cu[i] = calcCuFromCurie(grain_tc[i]);
		float a = curand_log_normal(&rand_stat, grain_size_mu, grain_size_sigma);
		grain_area[i] = a * a;
		//grain_ku_kum[i] = 1;
		grain_prob[i] = INITIAL_MAG_PROB;
	}


	for (int i = -attempt_offset; i < ber_list_count; i++)
	{

		float temp = TEMP_CURIE_MEAN - THERMAL_GRADIENT * LINER_VELOCITY * TAU_AP * 1.0e+9 * i;
		if (temp < TEMP_AMBIENT)
			temp = TEMP_AMBIENT;



		for (int k = 0; k < GRAIN_COUNT; k++)
		{

			if (temp > grain_tc[k]) continue;

			float kbm, kbp;
			calcKb(temp, hw, grain_cu[k],grain_tc[k], kbp, kbm);

			float prob_neg = 0 <= i && i < hw_switch_ap ? exp(-kbp * grain_area[k]) : exp(-kbm * grain_area[k]);
			float prob_pog = 0 <= i && i < hw_switch_ap ? exp(-kbm * grain_area[k]) : exp(-kbp * grain_area[k]);
			grain_prob[k] = prob_neg * (1 - grain_prob[k]) + (1 - prob_pog) * grain_prob[k];
		}



		float ber = calcPattern(grain_area, grain_prob);
		atomicAdd(&ber_list[i], ber);
	}
}

void calcContinusBitErrorRateHost(float *bER_list, int bER_list_count, float hw)
{
	float *dev_ber_list;
	unsigned long long int random_seed = (unsigned long long int)(time(NULL));
	unsigned long long int *random_seed_ptr = &random_seed;

	for (int i = 0; i < bER_list_count; i++)
		bER_list[i] = 0;

	CUDA_SAFE_CALL(cudaSetDevice(CUDA_DEVICE_NUM));
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_ber_list, sizeof(int) * bER_list_count));
	CUDA_SAFE_CALL(cudaMemcpy(dev_ber_list, bER_list, sizeof(int) * bER_list_count, cudaMemcpyHostToDevice));

	CUDA_SAFE_CALL(cudaMemcpyToSymbol(kRandomSeed, &random_seed, sizeof(unsigned long long int)));
	
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

__global__ void calcMidLastBitErrorRateKernel(float *mid_be_list, float *last_be_list, float hw)
{

	int thread_number = threadIdx.x + blockIdx.x * blockDim.x;
	float grain_prob[GRAIN_COUNT];			// グレインの逆方向に向いている確率
	float grain_tc[GRAIN_COUNT];			// グレインごとのTc
	float grain_cu[GRAIN_COUNT];			// グレインごとのCu組成
	float grain_area[GRAIN_COUNT];			// グレインごとの面積
	//float grain_ku_kum[GRAIN_COUNT];			// グレインごとのKu/Kum
	float grain_size_mu = log((GRAIN_MEAN * GRAIN_MEAN) / sqrt(GRAIN_SD * GRAIN_SD + GRAIN_MEAN * GRAIN_MEAN));  					  // グレインサイズ分散のμ
	float grain_size_sigma = (sqrt(log((GRAIN_SD * GRAIN_SD) / (GRAIN_MEAN * GRAIN_MEAN) + 1)));									  // グレインサイズ分散のσ
	const int hw_switch_ap = (int)(BIT_PITCH / LINER_VELOCITY *1.0e-9 * F0_AP);														  // 書込磁界順方向終了タイミング
	const int attempt_offset = (int)(TEMP_CURIE_MEAN * TEMP_CURIE_SD * 2 / (THERMAL_GRADIENT * LINER_VELOCITY * TAU_AP * 1.0e+9));	  // Tcm以前の予備シミュレーション
	const int last_attempt = hw_switch_ap * 2 + attempt_offset;


	curandStateMRG32k3a rand_stat;			// 乱数ステータス	
	curand_init(*kRandomSeed, thread_number, last_attempt * GRAIN_COUNT, &rand_stat);

	for (int i = 0; i < GRAIN_COUNT; i++)
	{
		grain_tc[i] = curand_normal(&rand_stat) * TEMP_CURIE_SD * TEMP_CURIE_MEAN + TEMP_CURIE_MEAN;
		grain_cu[i] = calcCuFromCurie(grain_tc[i]);
		float a = curand_log_normal(&rand_stat, grain_size_mu, grain_size_sigma);
		grain_area[i] = a * a;
		grain_prob[i] = INITIAL_MAG_PROB;
		//grain_ku_kum[i] = 1;
	}
	mid_be_list[thread_number] = 0;
	last_be_list[thread_number] = 0;

	for (int i = -attempt_offset; i < last_attempt; i++)
	{

		float temp = TEMP_CURIE_MEAN - THERMAL_GRADIENT * LINER_VELOCITY * TAU_AP * 1.0e+9F * i;
		if (temp < TEMP_AMBIENT)
			temp = TEMP_AMBIENT;

		for (int k = 0; k < GRAIN_COUNT; k++)
		{
			if (temp > grain_tc[k]) continue;

			float kbm, kbp;
			calcKb(temp, hw, grain_cu[k],grain_tc[k], kbp, kbm);
			//kbp = calcKb(temp, hw, grain_cu[k]);
			//kbm = calcKb(temp, -hw, grain_cu[k]);

			float prob_neg = 0 <= i && i < hw_switch_ap ? exp(-kbp * grain_area[k]) : exp(-kbm * grain_area[k]);
			float prob_pog = 0 <= i && i < hw_switch_ap ? exp(-kbm * grain_area[k]) : exp(-kbp * grain_area[k]);
			grain_prob[k] = prob_neg * (1 - grain_prob[k]) + (1 - prob_pog) * grain_prob[k];
		}

		if (i == hw_switch_ap - 1) // EAW用に中抜き
			mid_be_list[thread_number] = calcPattern(grain_area, grain_prob);
			
	}

	last_be_list[thread_number] = calcPattern(grain_area, grain_prob);

}

void calcMidLastBitErrorRateHost(float *mid_bER, float *last_bER, float hw)
{
	const int list_size = BIT_COUNT;
	float *mid_be_list = (float*)malloc(sizeof(float) * list_size);
	float *last_be_list = (float*)malloc(sizeof(float) * list_size);
	float *dev_mid_be_list = NULL;
	float *dev_last_be_list = NULL;
	unsigned long long int random_seed = (unsigned long long int)(time(NULL));

	CUDA_SAFE_CALL(cudaSetDevice(CUDA_DEVICE_NUM));
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_mid_be_list, sizeof(float) * list_size));
	CUDA_SAFE_CALL(cudaMalloc((void**)&dev_last_be_list, sizeof(float) * list_size));

	CUDA_SAFE_CALL(cudaMemcpy(dev_mid_be_list, mid_be_list, sizeof(float) * list_size, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpy(dev_last_be_list, last_be_list, sizeof(float) * list_size, cudaMemcpyHostToDevice));
	CUDA_SAFE_CALL(cudaMemcpyToSymbol(kRandomSeed, &random_seed, sizeof(unsigned long long int)));


	calcMidLastBitErrorRateKernel <<<CUDA_BLOCK_COUNT, CUDA_THREAD_COUNT>>>(dev_mid_be_list, dev_last_be_list, hw);
	CUDA_SAFE_CALL(cudaGetLastError());
	CUDA_SAFE_CALL(cudaDeviceSynchronize());
	CUDA_SAFE_CALL(cudaMemcpy(mid_be_list, dev_mid_be_list, sizeof(float) * list_size, cudaMemcpyDeviceToHost));
	CUDA_SAFE_CALL(cudaMemcpy(last_be_list, dev_last_be_list, sizeof(float) * list_size, cudaMemcpyDeviceToHost));

	
	float temp_mid_bER = 0;
	float temp_last_bER = 0;

	/*// ニコイチリダクション、高精度？
	float half = list_size / 2;
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
	thrust::device_ptr<float> dev_mid_be_ptr(dev_mid_be_list);
	thrust::device_ptr<float> dev_last_be_ptr(dev_last_be_list);

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


// ################################################################################################
// #



void makeHwBerList(FILE *fp)
{
	fprintf(fp, "Hw[kOe]\tTrw[ns]\tbER\tbER(WE)\tbER(EAW)\n");
	for (int i = 0; i < HW_LIST_SIZE; i++)
	{
		printf("%d / %d \r", i, HW_LIST_SIZE);

		float hw = (HW_LAST- HW_FIRST) * i / HW_LIST_SIZE + HW_FIRST;
		float trw = calcRecordingTimeWindow(hw);

		float mid_bER = 0;
		float last_bER = 0;
		calcMidLastBitErrorRateHost(&mid_bER, &last_bER, hw);

		fprintf(fp, "%f\t%e\t%.10e\t%.10e\t%.10e\n", hw * 1e-3,trw, last_bER, mid_bER, last_bER - mid_bER);
	}
}

void makeContinusBerList(FILE *fp)
{
	const int hw_switch_ap = (int)(BIT_PITCH / LINER_VELOCITY *1.0e-9 * F0_AP);														  // 書込磁界順方向終了タイミング
	const int attempt_count = hw_switch_ap * 3;
	float ber[attempt_count];

	calcContinusBitErrorRateHost(ber, attempt_count, CBER_HW);

	fprintf(fp, "Count\tTemp\tTime(ns)\tbER\n");
	for (int i = 0; i < attempt_count; i++)
	{
		fprintf(fp, "%d\t%f\t%f\t%e\n", i,
			convertTempFromAP(i),
			i * TAU_AP * 1e+9,
			ber[i]);
	}
}

void writeParamsEx(FILE *fp)
{
	fprintf(fp, "CUDA_THREAD_COUNT\t%e", CUDA_THREAD_COUNT);
	fprintf(fp, "CUDA_BLOCK_COUNT\t%e", CUDA_BLOCK_COUNT);
	fprintf(fp, "BIT_COUNT\t%e", BIT_COUNT);
	fprintf(fp, "CUDA_DEVICE_NUM\t%e", CUDA_DEVICE_NUM);
	fprintf(fp, "S_FE_MEAN_ERROR\t%e", S_FE_MEAN_ERROR);
	fprintf(fp, "K_B\t%e", K_B);
	fprintf(fp, "M_B\t%e", M_B);
	fprintf(fp, "S_FE_MEAN_MAX\t%e", S_FE_MEAN_MAX);
	fprintf(fp, "F0_AP\t%e", F0_AP);
	fprintf(fp, "TAU_AP\t%e", TAU_AP);
	fprintf(fp, "HW_FIRST\t%e", HW_FIRST);
	fprintf(fp, "HW_LAST\t%e", HW_LAST);
	fprintf(fp, "HW_LIST_SIZE\t%e", HW_LIST_SIZE);
	fprintf(fp, "READABLE_THRETH_PER_GRAIN\t%e", READABLE_THRETH_PER_GRAIN);
	fprintf(fp, "READABLE_THRETH\t%e", READABLE_THRETH);
	fprintf(fp, "G_FE\t%e", G_FE);
	fprintf(fp, "S_FE\t%e", S_FE);
	fprintf(fp, "V_FE\t%e", V_FE);
	fprintf(fp, "V_PT\t%e", V_PT);
	fprintf(fp, "V_CU\t%e", V_CU);
	fprintf(fp, "KU_KBULK\t%e", KU_KBULK);
	fprintf(fp, "J_FE_FE\t%e", J_FE_FE);
	fprintf(fp, "BULK_D_FE_FE\t%e", BULK_D_FE_FE);
	fprintf(fp, "THERMAL_GRADIENT\t%e", THERMAL_GRADIENT);
	fprintf(fp, "TEMP_AMBIENT\t%e", TEMP_AMBIENT);
	fprintf(fp, "BIT_AREA\t%e", BIT_AREA);
	fprintf(fp, "GRAIN_COUNT\t%e", GRAIN_COUNT);
	fprintf(fp, "S_DELTA\t%e", S_DELTA);
	fprintf(fp, "THICKNESS\t%e", THICKNESS);
	fprintf(fp, "GRAIN_VOLUME\t%e", GRAIN_VOLUME);
	fprintf(fp, "BIT_PITCH\t%e", BIT_PITCH);
	fprintf(fp, "LINER_VELOCITY\t%e", LINER_VELOCITY);
	fprintf(fp, "GRAIN_SD\t%e", GRAIN_SD);
	fprintf(fp, "GRAIN_MEAN\t%e", GRAIN_MEAN);
	fprintf(fp, "TEMP_CURIE_SD\t%e", TEMP_CURIE_SD);
	fprintf(fp, "TEMP_CURIE_MEAN\t%e", TEMP_CURIE_MEAN);
	fprintf(fp, "FE\t%e", FE);
	fprintf(fp, "HW_SW_OFFSET\t%e", HW_SW_OFFSET);
	fprintf(fp, "CBER_HW\t%e", CBER_HW);
	fprintf(fp, "BER_ALGORITHM\t%e", BER_ALGORITHM);
	fprintf(fp, "INITIAL_MAG_PROB\t%e", INITIAL_MAG_PROB);
	fprintf(fp, "PROGRAM_MODE\t%e", PROGRAM_MODE);
	fprintf(fp, "SIM_TITLE\t%e", SIM_TITLE);
	fprintf(fp, "SIM_COMMENT\t%e", SIM_COMMENT);
}

void getFilename(char *buf, char *prefix)
{
	time_t now = time(NULL);
	tm* t = localtime(&now);
	sprintf(buf, "%s_%s__%04d%02d%02d%_%02d%02d.txt", SIM_TITLE, prefix, t->tm_year+1900, t->tm_mon, t->tm_mday, t->tm_hour, t->tm_min);
}


/*************************************
*
*/


void subWriteParam(char* base_filename)
{
	char param_filename[256];
	sprintf(param_filename, "%s.param.txt", base_filename);
	FILE *fp = fopen(param_filename, "w");
	writeParamsEx(fp);
	fclose(fp);
}


void subKbList()
{
	char filename[256];
	getFilename(filename, "kb");
	sprintf(filename, "%s_kb_list.txt", SIM_TITLE);

	subWriteParam(filename);
	FILE *fp = fopen(filename, "w");
	calcKbListHost(fp, 10e+3);
	fclose(fp);
}

void subHwBER()
{
	char filename[256];
	getFilename(filename, "hw");
	sprintf(filename, "%s_hw_list.txt", SIM_TITLE);

	subWriteParam(filename);
	FILE *fp = fopen(filename, "w");
	makeHwBerList(fp);
	fclose(fp);
}


void subContinusBER()
{
	char filename[256];
	char prefix[256];
	sprintf(prefix, "cber_%e", CBER_HW / 1e+3);
	getFilename(filename, prefix);

	subWriteParam(filename);
	FILE *fp = fopen(filename, "w");
	makeContinusBerList(fp);
	fclose(fp);
}



int main()
{
	
	auto start = std::chrono::system_clock::now();
	cudaProfilerStart();

	subKbList();
	subContinusBER();
	subHwBER();


	auto end = std::chrono::system_clock::now();
	auto dur = end - start;
	auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(dur).count();

	std::cout << "\n" << msec << " milli sec \n";
	cudaProfilerStop();
    return 0;

}
