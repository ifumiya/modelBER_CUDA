#ifndef IG_MODEL_BER_PARAMS
#define IG_MODEL_BER_PARAMS

#define CUDA_THREAD_COUNT           500                  /// <summary>並列実行するCUDAスレッド数                                </summary>
#define CUDA_BLOCK_COUNT            4000                 /// <summary>並列実行するCUDAブロック数                                </summary>
#define BIT_COUNT                   (CUDA_THREAD_COUNT*CUDA_BLOCK_COUNT)
                                                         /// <summary>平均bERの計算に用いるbit数                                </summary>       
#define CUDA_DEVICE_NUM             0                    /// <summary>計算に用いるデバイス (1:Quadro 0,2:Tesla)                 </summary>
#define S_FE_MEAN_ERROR             1.0e-5F              /// <summary>平均分芝の計算精度                                        </summary>
#define K_B                         1.38065E-16F         /// <summary>ボルツマン定数　(erg/K)                                   </summary>
#define M_B                         9.27401E-21F         /// <summary>ボーア磁子 (emu)                                          </summary>
#define S_FE_MEAN_MAX               20.0F                /// <summary>分子場の最大値                                            </summary>
#define F_AP                        (400e+9F * 2.0F)     /// <summary>マイクロマグによる基本試行頻度(V200 Tc700 Ku/Kbulk0.4)    </summary>
#define ALPHA_AP                    0.1F                 /// <summary> 損失定数                                                 </summary>
//#define F0_AP                       1.0e+11F           /// <summary>試行頻度(1/s)                                             </summary>
#define F0_AP                       (ALPHA_AP/(1+ALPHA_AP*ALPHA_AP) * F_AP)
                                                         /// <summary>試行頻度(1/s)                                             </summary>
#define TAU_AP                      (1/ F0_AP)           /// <summary>試行間隔(s)                                               </summary>
#define HW_FIRST                    0.0F                 /// <summary>Hw-berグラフ作成時に計算を始めるHw                        </summary>
#define HW_LAST                     20.0e+3F             /// <summary>Hw-berグラフ作成時に計算を終わるHw                        </summary>
#define HW_LIST_SIZE                20                   /// <summary>Hw-last_bER グラフ作成時のプロット数                      </summary>
#define READABLE_THRETH_PER_GRAIN   0.35F                /// <summary>読み取りエラーの出力面積割合でのしきい値                  </summary>
#define READABLE_THRETH             (0.35F*GRAIN_COUNT)  /// <summary>読み取りエラーの出力面積でのしきい値                      </summary>
#define G_FE                        2.0F                 /// <summary>Feのg-factor                                              </summary>
#define S_FE                        1.504746866F         /// <summary>Feのスピン角運動量                                        </summary>
#define V_FE                        1.180000e-23F        /// <summary>Feの原子容積(cm^3)                                        </summary>
#define V_PT                        1.510000e-23F        /// <summary>Ptの原子容積(cm^3)                                        </summary>
#define V_CU                        1.180000e-23F        /// <summary>Cuの原子容積(cm^3)                                        </summary>
#define KU_KBULK                    0.4F                 /// <summary>異方性定数比[-]                                           </summary>
#define J_FE_FE                     1.058006613e-14F     /// <summary>交換積分                                                  </summary>
#define BULK_D_FE_FE                2.238102360e-16F     /// <summary>薄膜整形前の異方性定数                                    </summary>
#define THERMAL_GRADIENT            5.0F                 /// <summary>温度勾配(T/nm)                                            </summary>
#define TEMP_AMBIENT                330.0F               /// <summary>動作温度(K)                                               </summary>
#define BIT_AREA                    140.0F               /// <summary>ビット面積(nm^2)                                          </summary>
#define GRAIN_COUNT                 4                    /// <summary>グレイン数                                                </summary>
#define S_DELTA                     1.0F                 /// <summary>非磁性領域幅(nm)                                          </summary>
#define THICKNESS                   8.0e-7F              /// <summary>膜厚                                                      </summary>
#define GRAIN_VOLUME                1.93342723470e-19F   /// <summary>グレインの平均面積(cm^3)                                  </summary>
#define BIT_PITCH                   6.8F                 /// <summary>ビット幅(nm)                                              </summary>
#define LINER_VELOCITY              10                   /// <summary>線速度(m/s nm/ns)                                         </summary>
#define GRAIN_SD                    0.10F                /// <summary>グレインサイズ標準偏差                                    </summary>
#define GRAIN_MEAN                  1.0F                 /// <summary>グレインサイズ平均(計算上の)                              </summary>
#define TEMP_CURIE_SD               0.05F                /// <summary>Tc標準偏差                                                </summary>
#define TEMP_CURIE_MEAN             700.0F               /// <summary>Tc平均                                                    </summary>
#define KU_SD						0.0F				 /// <summary>Ku標準偏差												</summary>
#define KU_MEAN						1.0F				 /// <summary>Ku平均(計算上/係数として)									</summary>
#define FE                          0.5F                 /// <summary>Feの含有割合                                              </summary>
#define TAU_SFIT_TAU_STC            0                    /// <summary>磁界を反転させるタイミング(τshift/τσTc)                </summary>
#define CBER_HW                     2.0e+3F              /// <summary>試行周期ごとのbERを算出する際の書込磁界(Oe)               </summary>
#define BER_ALGORITHM               1                    /// <summary>bER算出アルゴリズム( 0: 純モンテカルロ　1: 確率+パターン  </summary>
#define INITIAL_MAG_PROB            0.5                  /// <summary>確率パターンにおいて、(記録磁界と反平行)磁化確率の初期値  </summary>
#define ENABLE_KB_CALC              0                    /// <summary>Kb/Pデータの作成を有効化する(1:有効 0:無効)               </summary>
#define ENABLE_KB_TC_CALC           0                    /// <summary>Kb/Pデータの作成を有効化する(1:有効 0:無効)               </summary>
#define ENABLE_CBER_CALC            0                    /// <summary>試行周期ごとのbERを計算する (1:有効 0:無効)               </summary>
#define ENABLE_HW_BER_CALC          1                    /// <summary>Hwごとのbit終端のbER計算    (1:有効 0:無効)               </summary>
#define SIM_TITLE                   "RESULT"                     
                                                         /// <summary> 出力ファイル名に追加されるコメント                       </summary>
#define SIM_COMMENT                    ""                /// <summary> パラメータ一覧に出力されるコメント                       </summary>


#endif