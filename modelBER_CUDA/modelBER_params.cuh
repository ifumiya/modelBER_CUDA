#define CUDA_THREAD_COUNT			100				/// <summary>������s����CUDA�X���b�h��</summary>
#define CUDA_BLOCK_COUNT			1000				/// <summary>������s����CUDA�u���b�N��</summary>
#define BIT_COUNT					(CUDA_THREAD_COUNT*CUDA_BLOCK_COUNT)
#define CUDA_DEVICE_NUM				0					/// <summary>�v�Z�ɗp����f�o�C�X (1:Quadro 0,1:Tesla) </summary>
#define S_FE_MEAN_ERROR				1.0e-8				/// <summary>���ϕ��ł̌v�Z���x									 </summary>
#define K_B							1.38065E-16			/// <summary>�{���c�}���萔�@(erg/K) 					 		 </summary>
#define M_B							9.27401E-21			/// <summary>�{�[�A���q (emu)								  	 </summary>
#define S_FE_MEAN_MAX				20.0				/// <summary>���q��̍ő�l								     	 </summary>
#define F0_AP						1.0e+11				/// <summary>���s�p�x(1/s)										 </summary>
#define TAU_AP						(1/ F0_AP)			/// <summary>���s�Ԋu(s)									  	 </summary>
#define HW_MAX						20.0e+3				/// <summary>KuKbulk-Hw�O���t�쐬���Ɍv�Z��r���ŕ�������Hw		 </summary>
#define HW_LIST_SIZE				20					/// <summary>KuKbulk-Hw�O���t�쐬���Ɍv�Z��r���ŕ�������Hw		 </summary>

#define READABLE_THRETH_PER_GRAIN	0.35				/// <summary>�ǂݎ��G���[�̏o�͖ʐϊ����ł̂������l    		 </summary>
#define READABLE_THRETH				(0.35*GRAIN_COUNT)  /// <summary>�ǂݎ��G���[�̏o�͖ʐςł̂������l			     </summary>
#define G_FE						2.0					/// <summary>Fe��g-factor 								   		 </summary>
#define S_FE						1.504746866			/// <summary>											    	 </summary>
#define V_FE						1.180000e-23		/// <summary>Fe�̌��q�e��(cm^3)							   		 </summary>
#define V_PT						1.510000e-23		/// <summary>Pt�̌��q�e��(cm^3)							   		 </summary>
#define V_CU						1.180000e-23		/// <summary>Cu�̌��q�e��(cm^3)									 </summary>
#define KU_KBULK					0.4					/// <summary>�ٕ����萔��[-]								 	 </summary>
#define J_FE_FE						1.058006613e-14		/// <summary>�����ϕ�									    	 </summary>
#define BULK_D_FE_FE				2.238102360e-16		/// <summary>�ٕ����萔�̏����l							   		 </summary>
#define THERMAL_GRADIENT			15.0649				/// <summary>���x���z											 </summary>
#define TEMP_AMBIENT				330.0				/// <summary>���쉷�x(K)								    	 </summary>
#define BIT_AREA					140.0				/// <summary>�r�b�g�ʐ�									   		 </summary>
#define GRAIN_COUNT					4					/// <summary>�O���C����									   		 </summary>
#define S_DELTA						1.0					/// <summary>�񎥐��̈敝(nm)							 		 </summary>
#define THICKNESS					8.0e-7				/// <summary>����											 	 </summary>
#define GRAIN_VOLUME				1.93342723470e-19	/// <summary>�O���C���̕��ϖʐ�(cm^3)							 </summary>   
#define BIT_PITCH					6.8					/// <summary>�r�b�g��(nm)								    	 </summary>
#define LINER_VELOCITY				10					/// <summary>�����x(m/s	nm/ns)						     		 </summary>
#define GRAIN_SD					0.10				/// <summary>�O���C���T�C�Y���U							   		 </summary>
#define GRAIN_MEAN					1.0					/// <summary>�O���C���T�C�Y����							   		 </summary>
#define TEMP_CURIE_SD				0.05				/// <summary>Tc���U										  		 </summary>
#define TEMP_CURIE_MEAN				700.0				/// <summary>Tc����										  		 </summary>
#define FE							0.5					/// <summary>Fe�̊ܗL����										 </summary>
#define HW_SW_OFFSET				0					/// <summary>���E�𔽓]������^�C�~���O(ap count)				 </summary>
#define CBER_HW						10e+3				/// <summary>���E�𔽓]������^�C�~���O(ap count)				 </summary>

#define BER_ALGORITHM				1					/// <summary> bER�Z�o�A���S���Y��( 0: �������e�J�����@1: �m��+�p�^�[�� </summary>
#define INITIAL_MAG_PROV			1					/// <summary>�m���p�^�[���ɂ����āA�����m���̏����l </summary>