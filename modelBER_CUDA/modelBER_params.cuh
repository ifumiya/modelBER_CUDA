#define CUDA_THREAD_COUNT			100				/// <summary>並列実行するCUDAスレッド数</summary>
#define CUDA_BLOCK_COUNT			1000				/// <summary>並列実行するCUDAブロック数</summary>
#define BIT_COUNT					(CUDA_THREAD_COUNT*CUDA_BLOCK_COUNT)
#define CUDA_DEVICE_NUM				0					/// <summary>計算に用いるデバイス (1:Quadro 0,1:Tesla) </summary>
#define S_FE_MEAN_ERROR				1.0e-8				/// <summary>平均分芝の計算精度									 </summary>
#define K_B							1.38065E-16			/// <summary>ボルツマン定数　(erg/K) 					 		 </summary>
#define M_B							9.27401E-21			/// <summary>ボーア磁子 (emu)								  	 </summary>
#define S_FE_MEAN_MAX				20.0				/// <summary>分子場の最大値								     	 </summary>
#define F0_AP						1.0e+11				/// <summary>試行頻度(1/s)										 </summary>
#define TAU_AP						(1/ F0_AP)			/// <summary>試行間隔(s)									  	 </summary>
#define HW_MAX						20.0e+3				/// <summary>KuKbulk-Hwグラフ作成時に計算を途中で放棄するHw		 </summary>
#define HW_LIST_SIZE				20					/// <summary>KuKbulk-Hwグラフ作成時に計算を途中で放棄するHw		 </summary>

#define READABLE_THRETH_PER_GRAIN	0.35				/// <summary>読み取りエラーの出力面積割合でのしきい値    		 </summary>
#define READABLE_THRETH				(0.35*GRAIN_COUNT)  /// <summary>読み取りエラーの出力面積でのしきい値			     </summary>
#define G_FE						2.0					/// <summary>Feのg-factor 								   		 </summary>
#define S_FE						1.504746866			/// <summary>											    	 </summary>
#define V_FE						1.180000e-23		/// <summary>Feの原子容積(cm^3)							   		 </summary>
#define V_PT						1.510000e-23		/// <summary>Ptの原子容積(cm^3)							   		 </summary>
#define V_CU						1.180000e-23		/// <summary>Cuの原子容積(cm^3)									 </summary>
#define KU_KBULK					0.4					/// <summary>異方性定数比[-]								 	 </summary>
#define J_FE_FE						1.058006613e-14		/// <summary>交換積分									    	 </summary>
#define BULK_D_FE_FE				2.238102360e-16		/// <summary>異方性定数の初期値							   		 </summary>
#define THERMAL_GRADIENT			15.0649				/// <summary>温度勾配											 </summary>
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
#define TEMP_CURIE_SD				0.05				/// <summary>Tc分散										  		 </summary>
#define TEMP_CURIE_MEAN				700.0				/// <summary>Tc平均										  		 </summary>
#define FE							0.5					/// <summary>Feの含有割合										 </summary>
#define HW_SW_OFFSET				0					/// <summary>磁界を反転させるタイミング(ap count)				 </summary>
#define CBER_HW						10e+3				/// <summary>磁界を反転させるタイミング(ap count)				 </summary>

#define BER_ALGORITHM				1					/// <summary> bER算出アルゴリズム( 0: 純モンテカルロ　1: 確率+パターン </summary>
#define INITIAL_MAG_PROV			1					/// <summary>確率パターンにおいて、磁化確率の初期値 </summary>