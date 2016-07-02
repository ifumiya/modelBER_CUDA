

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

const double kTc500_KuKbulk = 0.677;
const double kTc600_KuKbulk = 0.3949;
const double kTc700_KuKbulk = 0.269;


#define CUDA_THREAD_COUNT			512					/// <summary>並列実行するCUDAスレッド数</summary>
#define CUDA_BLOCK_COUNT			2048				/// <summary>並列実行するCUDAブロック数</summary>
#define BIT_COUNT					(CUDA_THREAD_COUNT*CUDA_BLOCKCOUNT)
#define CUDA_DEVICE_NUM				1					/// <summary>計算に用いるデバイス</summary>
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
#define KU_KBULK					1.0					/// <summary>異方性定数比[-]								 	 </summary>
#define J_FE_FE						1.058006613e-14		/// <summary>交換積分									    	 </summary>
#define BULK_D_FE_FE				2.238102360e-16		/// <summary>異方性定数の初期値							   		 </summary>
#define THERMAL_GRADIENT			15.1				/// <summary>温度勾配											 </summary>
#define TEMP_AMBIENT				330.0				/// <summary>動作温度(K)								    	 </summary>
#define BIT_AREA					140.0				/// <summary>ビット面積									   		 </summary>
#define GRAIN_COUNT					4					/// <summary>グレイン数									   		 </summary>
#define S_DELTA						1.0					/// <summary>非磁性領域幅(nm)							 		 </summary>
#define THICKNESS					8.0e-7				/// <summary>膜厚											 	 </summary>
#define GRAIN_VOLUME				1.93342723470e-19	/// <summary>グレインの平均面積(cm^3)							 </summary>   
#define BIT_PITCH					6.8					/// <summary>ビット幅(nm)								    	 </summary>
#define LINER_VELOCITY				10					/// <summary>線速度(m/s	nm/ns)						     		 </summary>
#define GRAIN_SD					0.10				/// <summary>グレインサイズ分散							   		 </summary>
#define GRAIN_MEAN					1.0					/// <summary>グレインサイズ平均							   		 </summary>
#define TEMP_CURIE_SD				0.0					/// <summary>Tc分散										  		 </summary>
#define TEMP_CURIE_MEAN				700.0				/// <summary>Tc平均										  		 </summary>
#define FE							0.5					/// <summary>Feの含有割合										 </summary>
#define HW_SW_OFFSET				 0					/// <summary>磁界を反転させるタイミング(ap count)				 </summary>
__constant__ unsigned long long int kRandomSeed;		// aaaa



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
void calcContinusBitErrorRate(int *ber_list, int ber_list_count, double hw)
{
	int thread_number = threadIdx.x + blockIdx.x * blockDim.x;

	double grain_tcs[GRAIN_COUNT];			// グレインごとのTc
	double grain_cu[GRAIN_COUNT];			// グレインごとのCu組成
	double grain_area[GRAIN_COUNT];			// グレインごとの面積
	double grain_ku[GRAIN_COUNT];			// グレインごとのKu/Kum
	int grain_dir[GRAIN_COUNT];				// グレインの磁化の向き (1 = 逆方向、-1 = 順方向)
	curandStateMRG32k3a rand_stat;			// 乱数ステータス
	
	curand_init(kRandomSeed, thread_number, ber_list_count * GRAIN_COUNT, &rand_stat);


	for (int i = 0; i < GRAIN_COUNT; i++)
	{
		grain_tcs[i] = curand_normal_double(&rand_stat) * TEMP_CURIE_SD + TEMP_CURIE_MEAN;
		grain_cu[i] = calcCuFromCurie(grain_tcs[i]);
		double a = curand_log_normal_double(&rand_stat, GRAIN_MEAN, GRAIN_SD);
		grain_area[i] = a * a;
		grain_ku[i] = 1;
		grain_dir[i] = 1;
	}

	int hw_switch_ap = BIT_PITCH / LINER_VELOCITY *1.0e-9 * F0_AP ;
	
	for (int i = 0; i < ber_list_count; i++)
	{
		double s = 0;
		
		double temp = THERMAL_GRADIENT * LINER_VELOCITY * TAU_AP * 1.0e+9 * i;
		if (temp < TEMP_AMBIENT)
			temp = TEMP_AMBIENT;

		
		if (i == 0 || i == hw_switch_ap)
			hw = -hw;


		for (int j = 0; j < GRAIN_COUNT; j++)
		{
			// 磁化反転する確率
			// hw = -1 = 順方向
			// grain_dir = 1 逆方向
			// hw * grain_dir = -1 hw方向への反転確率
			
			double rev_prob = exp(-calcKb(temp, hw * grain_dir[i], grain_cu[i]) * grain_area[i] );
			
			if (rev_prob < curand_uniform(&rand_stat))
				grain_dir[i] = -grain_dir[i];
			if (grain_dir[i] < 0 && temp < grain_tcs[i])
				s += grain_area[i];
		}

		if (READABLE_THRETH < s)
			atomicAdd(&ber_list[i], 1);
	}
}

void calcContinusBitErrorRateHost(double *bER_list, int bER_list_count)
{
	int *dev_be_list;
	int *be_list = (int*)malloc(sizeof(int) * bER_list_count);

	for (int i = 0; i < bER_list_count; i++)
		be_list[i] = 0;

	cuda



}


int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize] = { 0 };

    // Add vectors in parallel.
    cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addWithCuda failed!");
        return 1;
    }

    printf("{1,2,3,4,5} + {10,20,30,40,50} = {%d,%d,%d,%d,%d}\n",
        c[0], c[1], c[2], c[3], c[4]);

    // cudaDeviceReset must be called before exiting in order for profiling and
    // tracing tools such as Nsight and Visual Profiler to show complete traces.
    cudaStatus = cudaDeviceReset();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceReset failed!");
        return 1;
    }

    return 0;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    int *dev_a = 0;
    int *dev_b = 0;
    int *dev_c = 0;
    cudaError_t cudaStatus;

    // Choose which GPU to run on, change this on a multi-GPU system.
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    // Allocate GPU buffers for three vectors (two input, one output)    .
    cudaStatus = cudaMalloc((void**)&dev_c, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_a, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_b, size * sizeof(int));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    // Copy input vectors from host memory to GPU buffers.
    cudaStatus = cudaMemcpy(dev_a, a, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_b, b, size * sizeof(int), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(dev_c, dev_a, dev_b);

    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "addKernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }
    
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching addKernel!\n", cudaStatus);
        goto Error;
    }

    // Copy output vector from GPU buffer to host memory.
    cudaStatus = cudaMemcpy(c, dev_c, size * sizeof(int), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_c);
    cudaFree(dev_a);
    cudaFree(dev_b);
    
    return cudaStatus;
}
