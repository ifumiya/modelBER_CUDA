

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <device_functions.h>
#include <device_atomic_functions.h>
#include <curand.h>
#include <curand_kernel.h>
#include <vector_functions.h>

#include <stdio.h>
#include <math.h>
#include <time.h>

const double kTc500_KuKbulk = 0.677;
const double kTc600_KuKbulk = 0.3949;
const double kTc700_KuKbulk = 0.269;


#define CUDA_THREAD_COUNT			512					/// <summary>並列実行するCUDAスレッド数</summary>
#define CUDA_BLOCK_COUNT			2048					/// <summary>並列実行するCUDAブロック数</summary>
#define BIT_COUNT					(CUDA_THREAD_COUNT*CUDA_BLOCK_COUNT)
#define CUDA_DEVICE_NUM				1					/// <summary>計算に用いるデバイス (0:Auto 1:Quadro 2,3:Tesla) </summary>
#define S_FE_MEAN_ERROR				1.0e-8				/// <summary>平均分芝の計算精度									 </summary>
#define K_B							1.38065E-16			/// <summary>ボルツマン定数　(erg/K) 					 		 </summary>
#define M_B							9.27401E-21			/// <summary>ボーア磁子 (emu)								  	 </summary>
#define S_FE_MEAN_MAX				20.0				/// <summary>分子場の最大値								     	 </summary>
#define F0_AP						1.0e+11				/// <summary>試行頻度(1/s)										 </summary>
#define TAU_AP						(1/ F0_AP)			/// <summary>試行間隔(s)									  	 </summary>
#define HW_MAX						20.0e+3				/// <summary>KuKbulk-Hwグラフ作成時に計算を途中で放棄するHw		 </summary>
#define READABLE_THRETH_PER_GRAIN	0.35				/// <summary>読み取りエラーの出力面積割合でのしきい値    		 </summary>
#define READABLE_THRETH				(0.35*GRAIN_COUNT)  /// <summary>読み取りエラーの出力面積でのしきい値
#define G_FE						2.0					/// <summary>Feのg-factor 								   		 </summary>
#define S_FE						1.504746866			/// <summary>											    	 </summary>
#define V_FE						1.180000e-23		/// <summary>Feの原子容積(cm^3)							   		 </summary>
#define V_PT						1.510000e-23		/// <summary>Ptの原子容積(cm^3)							   		 </summary>
#define V_CU						1.180000e-23		/// <summary>Cuの原子容積(cm^3)									 </summary>
#define KU_KBULK					0.8					/// <summary>異方性定数比[-]								 	 </summary>
#define J_FE_FE						1.058006613e-14		/// <summary>交換積分									    	 </summary>
#define BULK_D_FE_FE				2.238102360e-16		/// <summary>異方性定数の初期値							   		 </summary>
#define THERMAL_GRADIENT			6.1					/// <summary>温度勾配											 </summary>
#define TEMP_AMBIENT				330.0				/// <summary>動作温度(K)								    	 </summary>
#define BIT_AREA					140.0				/// <summary>ビット面積									   		 </summary>
#define GRAIN_COUNT					4					/// <summary>グレイン数									   		 </summary>
#define S_DELTA						1.0					/// <summary>非磁性領域幅(nm)							 		 </summary>
#define THICKNESS					8.0e-7				/// <summary>膜厚											 	 </summary>
#define GRAIN_VOLUME				1.93342723470e-19	/// <summary>グレインの平均面積(cm^3)							 </summary>   
#define BIT_PITCH					6.8					/// <summary>ビット幅(nm)								    	 </summary>
#define LINER_VELOCITY				10					/// <summary>線速度(m/s	nm/ns)						     		 </summary>
#define GRAIN_SD					0.05				/// <summary>グレインサイズ分散							   		 </summary>
#define GRAIN_MEAN					1.0					/// <summary>グレインサイズ平均							   		 </summary>
#define TEMP_CURIE_SD				0.01				/// <summary>Tc分散										  		 </summary>
#define TEMP_CURIE_MEAN				700.0				/// <summary>Tc平均										  		 </summary>
#define FE							0.5					/// <summary>Feの含有割合										 </summary>
#define HW_SW_OFFSET				 0					/// <summary>磁界を反転させるタイミング(ap count)				 </summary>
#define CBER_HW						15e+3				/// <summary>磁界を反転させるタイミング(ap count)				 </summary>


__constant__ unsigned long long int kRandomSeed;		/// <summary>乱数のシード値</summary>



__device__
inline double calcCurieFromCu(double cu)
{
	return (2 * J_FE_FE * 4 * (1 - cu) * (G_FE - 1)  * (G_FE - 1) * S_FE * (S_FE + 1)) / (3 * K_B);
}

__device__
inline double calcCuFromCurie(double temp_curie)
{
	return -temp_curie * (3 * K_B) / (2 * J_FE_FE * 4 * (G_FE - 1) *(G_FE - 1) * S_FE * (S_FE + 1)) + 1;
}

__device__
inline double calcBrillouin(double s, double x)
{
	double o1 = (2 * s + 1) / (2 * s);
	double o2 = 1 / (2 * s);
	return o1 / tanh(o1 * x) - o2 / tanh(o2 * x);
}

__device__
double calc_sFe_Mean(double temp, double cu)
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

__device__
double calcKb(double temp, double hw, double cu)
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

__global__
void calcContinusBitErrorRateKernel(int *ber_list, int ber_list_count, double hw)
{
	int thread_number = threadIdx.x + blockIdx.x * blockDim.x;	
	double grain_tcs[GRAIN_COUNT];			// グレインごとのTc
	double grain_cu[GRAIN_COUNT];			// グレインごとのCu組成
	double grain_area[GRAIN_COUNT];			// グレインごとの面積
	double grain_ku[GRAIN_COUNT];			// グレインごとのKu/Kum
	int grain_dir[GRAIN_COUNT];				// グレインの磁化の向き (1 = 逆方向、-1 = 順方向)
	double grain_size_mu = log((GRAIN_MEAN * GRAIN_MEAN) / sqrt(GRAIN_SD * GRAIN_SD + GRAIN_MEAN * GRAIN_MEAN));  					  // グレインサイズ分散のμ
	double grain_size_sigma = (sqrt(log((GRAIN_SD * GRAIN_SD) / (GRAIN_MEAN * GRAIN_MEAN) + 1)));									  // グレインサイズ分散のσ
	const int hw_switch_ap = (int)(BIT_PITCH / LINER_VELOCITY *1.0e-9 * F0_AP);														  // 書込磁界順方向終了タイミング
	const int attempt_offset = (int)(TEMP_CURIE_MEAN * TEMP_CURIE_SD * 2 / (THERMAL_GRADIENT * LINER_VELOCITY * TAU_AP * 1.0e+9));	  // Tc以前の予備シミュレーション


	curandStateMRG32k3a rand_stat;			// 乱数ステータス	
	curand_init(kRandomSeed, thread_number, ber_list_count * GRAIN_COUNT, &rand_stat);


	for (int i = 0; i < GRAIN_COUNT; i++)
	{
		grain_tcs[i] = curand_normal_double(&rand_stat) * TEMP_CURIE_SD + TEMP_CURIE_MEAN;
		grain_cu[i] = calcCuFromCurie(grain_tcs[i]);
		double a = curand_log_normal_double(&rand_stat, grain_size_mu, grain_size_sigma);
		grain_area[i] = a * a;
		grain_ku[i] = 1;
		grain_dir[i] = 1;
	}

	double signed_hw = hw;
	for (int i = -attempt_offset; i < ber_list_count; i++)
	{
		double signal_power = 0;
		
		double temp =TEMP_CURIE_MEAN -  THERMAL_GRADIENT * LINER_VELOCITY * TAU_AP * 1.0e+9 * i;
		if (temp < TEMP_AMBIENT)
			temp = TEMP_AMBIENT;

		
		if (i == 0 || i == hw_switch_ap)
			signed_hw = -signed_hw;


		for (int k = 0; k < GRAIN_COUNT;k++)
		{
			// 磁化反転する確率
			// hw = -1 = 順方向
			// grain_dir = 1 逆方向
			// hw * grain_dir = -1 hw方向への反転確率
			
			double rev_prob = exp(-calcKb(temp, signed_hw * grain_dir[k], grain_cu[k]) * grain_area[k]);
			double dice = curand_uniform(&rand_stat);
			if (rev_prob > dice)
				grain_dir[k] = -grain_dir[k];
			if (grain_dir[k] < 0 && temp < grain_tcs[k])
				signal_power += grain_area[k];
		}

		if (READABLE_THRETH > signal_power && 0 <= i && i < ber_list_count)
			atomicAdd(&ber_list[i], 1);
	}
}

void calcContinusBitErrorRateHost(double *bER_list, int bER_list_count,double hw)
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


	calcContinusBitErrorRateKernel <<<CUDA_BLOCK_COUNT, CUDA_THREAD_COUNT >>>(dev_be_list, bER_list_count, hw);
	cudaDeviceSynchronize();
	cudaMemcpy(be_list, dev_be_list, sizeof(int) * bER_list_count, cudaMemcpyDeviceToHost);

	for (int i = 0; i < bER_list_count; i++)
	{
		bER_list[i] = (double)be_list[i] / BIT_COUNT;
		fprintf(stderr, "c %d \n", be_list[i]);
	}
	cudaFree(dev_be_list);
	free(be_list);
}


__global__
void calcKbListKernel(double *kb_list, int kb_list_size,double hw)
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

void calcKbListHost(double *kb_list, int kb_list_size,double hw)
{
	double *dev_kb_list = NULL;
	cudaSetDevice(CUDA_DEVICE_NUM);
	cudaMalloc((void**)&dev_kb_list, sizeof(double) * kb_list_size);
	cudaMemcpy(dev_kb_list, kb_list, sizeof(double) * kb_list_size, cudaMemcpyHostToDevice);

	calcKbListKernel << <1, 1 >> >(dev_kb_list, kb_list_size, hw);
	cudaDeviceSynchronize();
	cudaMemcpy(kb_list, dev_kb_list, sizeof(double) * kb_list_size, cudaMemcpyDeviceToHost);
}

/*
void showParameters()
{
	printf("%f\tCUDA_THREAD_COUNT \t並列実行するCUDAスレッド数", CUDA_THREAD_COUNT );
	printf("%f\tCUDA_BLOCK_COUNT \t並列実行するCUDAブロック数", CUDA_BLOCK_COUNT);
	printf("%f\tCUDA_DEVICE_NUM \t計算に用いるデバイス", CUDA_DEVICE_NUM);
	printf("%f\tS_FE_MEAN_ERROR \t平均分芝の計算精度", S_FE_MEAN_ERROR);
	printf("%f\tK_B \tボルツマン定数　(erg/K)", K_B);
	printf("%f\tM_B \tボーア磁子 (emu)", M_B);
	printf("%f\tS_FE_MEAN_MAX \t分子場の最大値", S_FE_MEAN_MAX);
	printf("%f\tF0_AP \t試行頻度(1/s)", F0_AP);
	printf("%f\tTAU_AP \t試行間隔(s)", TAU_AP);
	printf("%f\tHW_MAX \tKuKbulk-Hwグラフ作成時に計算を途中で放棄するHw", HW_MAX);
	printf("%f\tREADABLE_THRETH_PER_GRAIN \t読み取りエラーの出力面積割合でのしきい値", READABLE_THRETH_PER_GRAIN);
	printf("%f\tREADABLE_THRETH \t読み取りエラーの出力面積でのしきい値", READABLE_THRETH);
	printf("%f\tG_FE \tFeのg-factor", G_FE);
	printf("%f\tS_FE \t", S_FE);
	printf("%f\tV_FE \tFeの原子容積(cm^3)", V_FE);
	printf("%f\tV_PT \tPtの原子容積(cm^3)", V_PT);
	printf("%f\tV_CU \tCuの原子容積(cm^3)", V_CU);
	printf("%f\tKU_KBULK \t異方性定数比[-]", KU_KBULK);
	printf("%f\tJ_FE_FE \t交換積分", J_FE_FE);
	printf("%f\tBULK_D_FE_FE \t異方性定数の初期値", BULK_D_FE_FE);
	printf("%f\tTHERMAL_GRADIENT \t温度勾配", THERMAL_GRADIENT);
	printf("%f\tTEMP_AMBIENT \t動作温度(K)", TEMP_AMBIENT);
	printf("%f\tBIT_AREA \tビット面積", BIT_AREA);
	printf("%f\tGRAIN_COUNT \tグレイン数", GRAIN_COUNT);
	printf("%f\tS_DELTA \t非磁性領域幅(nm)", S_DELTA);
	printf("%f\tTHICKNESS \t膜厚", THICKNESS);
	printf("%f\tGRAIN_VOLUME \tグレインの平均面積(cm^3)", GRAIN_VOLUME);
	printf("%f\tBIT_PITCH \tビット幅(nm)", BIT_PITCH);
	printf("%f\tLINER_VELOCITY \t線速度(m/s	nm/ns)", LINER_VELOCITY);
	printf("%f\tGRAIN_SD \tグレインサイズ分散", GRAIN_SD);
	printf("%f\tGRAIN_MEAN \tグレインサイズ平均", GRAIN_MEAN);
	printf("%f\tTEMP_CURIE_SD \tTc分散", TEMP_CURIE_SD);
	printf("%f\tTEMP_CURIE_MEAN \tTc平均", TEMP_CURIE_MEAN);
	printf("%f\tFE \tFeの含有割合", FE);
	printf("%f\tHW_SW_OFFSET \t磁界を反転させるタイミング(ap count)", HW_SW_OFFSET);
}
*/

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

	
	double bER_list[200];
	double bER_list_count = 200;

	calcContinusBitErrorRateHost(bER_list, bER_list_count, CBER_HW);
	for (int i = 0; i < bER_list_count; i++)
	{
		printf("%f \n", bER_list[i]);
	}
	

    return 0;
}
