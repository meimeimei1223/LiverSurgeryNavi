#pragma once
#ifndef CMAES_REFINE_V3R_H
#define CMAES_REFINE_V3R_H
/*
 * CmaesRefineV3R.h
 * ----------------------------------------------------------------------
 * V3-R: Region-aware fork of CmaesRefineV3 for the Ctrl+G entry point.
 *
 * Differences from V3 (Shift+G):
 *   1. Inner-loop KDTree is built on a SUBSET of liver vertices, selected
 *      by a 4-quadrant bitmask (ant_R, ant_L, pos_R, pos_L). The subset
 *      is computed in voxel-after-downsample index space via NN lookup
 *      fro
[V3RS/cov-diag]   |                                 +##########################                                                            |
[V3RS/cov-diag]   |                                :#############################+                                                         |
[V3RS/cov-diag]   |                                ###############################+                                                        |
[V3RS/cov-diag]   |                              :+###################################+:                                                   |
[V3RS/cov-diag]   |                              +#####################################++                                                  |
[V3RS/cov-diag]   |                              ########################################++                                                |
[V3RS/cov-diag]   |                              ##########################################++: +#######+                                   |
[V3RS/cov-diag]   |                             +#########################################################++                               |
[V3RS/cov-diag]   |                             ############################################################++                             |
[V3RS/cov-diag]   |                             ##############################################################+:                           |
[V3RS/cov-diag]   |                            +#################################################################+++::::::                 |
[V3RS/cov-diag]   |                         ::+#####################################################################::::::                 |
[V3RS/cov-diag]   |                         :+#######################################################################+++++                 |
[V3RS/cov-diag]   |                         :+############################################################################+:               |
[V3RS/cov-diag]   |                         :#############################################################################++               |
[V3RS/cov-diag]   |                         :##############################################################################+               |
[V3RS/cov-diag]   |                        :###############################################################################+               |
[V3RS/cov-diag]   |                        :################################################################################++::           |
[V3RS/cov-diag]   |                        +##################################################################################+:           |
[V3RS/cov-diag]   |                        ####################################################################################+           |
[V3RS/cov-diag]   |                       +######################################################################################++        |
[V3RS/cov-diag]   |                      :#########################################################################################+:      |
[V3RS/cov-diag]   |                      :##########################################################################################:      |
[V3RS/cov-diag]   |                      +#######################+##########+++++####################################################      |
[V3RS/cov-diag]   |                      ###############################+#######+####+###############################################+     |
[V3RS/cov-diag]   |                      ###############################+############+################################################     |
[V3RS/cov-diag]   |                     +###############################++############################################################     |
[V3RS/cov-diag]   |                     ################################+#############################################################     |
[V3RS/cov-diag]   |                     ##############################################################################################     |
[V3RS/cov-diag]   |                     ##############################################################################################     |
[V3RS/cov-diag]   |                     #####################################++#######################################################     |
[V3RS/cov-diag]   |                    :###############################+#######################++#############+#######################     |
[V3RS/cov-diag]   |                    +####################################################################++:#######################     |
[V3RS/cov-diag]   |                    #########################################################++#########++:.+#####################+     |
[V3RS/cov-diag]   |                    ###################################################################+++  ++####################:     |
[V3RS/cov-diag]   |                   +######################+:##########+###############################+:    .+###################+      |
[V3RS/cov-diag]   |                   +#####################+. ########################################+++:     :+#################+:      |
[V3RS/cov-diag]   |                   ####################+:   +#####################################++:.        +++##############+.       |
[V3RS/cov-diag]   |                   ##################++.     :####################################++:.        .:::+++#########+         |
[V3RS/cov-diag]   |                   ################+.         ####################################+.            ..:+++#######+:         |
[V3RS/cov-diag]   |                   ##############+:.          ######################+#####+######++.                .:++#####:          |
[V3RS/cov-diag]   |                   #############++            +#################################+:.                                     |
[V3RS/cov-diag]   |                   #############+:             ###################+#+##########++:.                                     |
[V3RS/cov-diag]   |                   +###########++:             +#######++#####################++::.                                     |
[V3RS/cov-diag]   |                    ###########++:              +###++..+######+++++#########+::                                        |
[V3RS/cov-diag]   |                    ##########+++.                      +##################+:.                                          |
[V3RS/cov-diag]   |                    ##########+++                       +################++                                             |
[V3RS/cov-diag]   |                    ##########+..                       +###############+::                                             |
[V3RS/cov-diag]   |                    #########+                          :+############+::                                               |
[V3RS/cov-diag]   |                    +######++:                           :+##########++                                                 |
[V3RS/cov-diag]   |                    +######++.                            :++#######+:                                                  |
[V3RS/cov-diag]   |                     #####+                                  :++####+.                                                  |
[V3RS/cov-diag]   |                     +###+                                                                                              |
[V3RS/cov-diag]   |                                                                                                                        |
[V3RS/cov-diag]   |                                                                                                                        |
[V3RS/cov-diag]   |                                                                                                                        |
[V3RS/cov-diag]   |                                                                                                                        |
[V3RS/cov-diag]   |                                                                                                                        |
[V3RS/cov-diag]   |                                                                                                                        |
[V3RS/cov-diag]   |                                                                                                                        |
[V3RS/cov-diag]   |                                                                                                                        |
[V3RS/cov-diag]   |                                                                                                                        |
[V3RS/cov-diag]   |                                                                                                                        |
[V3RS/cov-diag]   |                                                                                                                        |
[V3RS/cov-diag]   |                                                                                                                        |
[V3RS/cov-diag]   |                                                                                                                        |
[V3RS] Gen    0  best=0.06224  sigma=0.2832
[V3RS] Gen  100  best=0.06211  sigma=0.0097
[V3RS] === Run 1 Time Breakdown (total 282 ms, 1350 evals) ===
[V3RS]   evaluate_one (sum) : 259 ms (91%)
[V3RS]   rebuild_corr (sum) : 21 ms (7%)  [14 calls]
[V3RS]   cmaes/log/other    : 1 ms (0%)
[V3RS]   build_eval_ctx     : 1 ms   compute_full_rmse  : 169 ms
[V3RS]   --- per-eval breakdown (1350 evals, 1350 with IoU) ---
[V3RS]     rmse_w (V3R-W)    : 20 ms (7%)  avg=15.1 us/eval
[V3RS]     sil_total (raster): 237 ms (91%)  avg=175.9 us/eval
[V3RS]       step1 proj      : 58 ms  avg=43.5 us/eval
[V3RS]       step2 splat(bbox): 148 ms  avg=110.2 us/eval
[V3RS]       step3 iou       : 28 ms  avg=21.4 us/eval
[V3RS]     other (M_srt+pen) : 0 ms  avg=0.3 us/eval
[V3RS] Run 1  best_inner=0.0617937  best_full=0.0525598  stop=MaxGenerations  [-]
[V3RS/sil] Run 1  iou_evals=1350  avg_iou2d=0.7037  avg_sil_loss=0.2963  lambda=4.0000
[V3RS] Run 2/10  Global  sigma0=0.5365  cma_seed=20262561
[V3R] build_eval_context  src=5595  subset=2829  tgt=1430  session_init_matched=940471  post_jitter_matched=1192  matched_min=357  post_jitter_rmse=0.115901  max_dist=0.227649m  scene_diag=1.6755
[V3RS] Run 2  Global  sigma0=0.536453  cma_seed=20262561
[V3RS] Deterministic seed: 20262561
[V3RS] Gen    0  best=0.11590  sigma=0.4672
[V3RS] Gen  100  best=0.10810  sigma=0.0068
[V3RS] === Run 2 Time Breakdown (total 304 ms, 1350 evals) ===
[V3RS]   evaluate_one (sum) : 274 ms (90%)
[V3RS]   rebuild_corr (sum) : 28 ms (9%)  [14 calls]
[V3RS]   cmaes/log/other    : 2 ms (0%)
[V3RS]   build_eval_ctx     : 2 ms   compute_full_rmse  : 363 ms
[V3RS]   --- per-eval breakdown (1350 evals, 1350 with IoU) ---
[V3RS]     rmse_w (V3R-W)    : 18 ms (6%)  avg=13.4 us/eval
[V3RS]     sil_total (raster): 255 ms (92%)  avg=189.0 us/eval
[V3RS]       step1 proj      : 66 ms  avg=49.1 us/eval
[V3RS]       step2 splat(bbox): 158 ms  avg=117.3 us/eval
[V3RS]       step3 iou       : 29 ms  avg=21.7 us/eval
[V3RS]     other (M_srt+pen) : 0 ms  avg=0.3 us/eval
[V3RS] Run 2  best_inner=0.108098  best_full=0.100672  stop=MaxGenerations  [-]
[V3RS/sil] Run 2  iou_evals=1350  avg_iou2d=0.5137  avg_sil_loss=0.4863  lambda=4.0000
[V3RS] Run 3/10  Local   sigma0=0.4285  cma_seed=20262562
[V3R] build_eval_context  src=5595  subset=2829  tgt=1430  session_init_matched=940471  post_jitter_matched=1425  matched_min=427  post_jitter_rmse=0.090509  max_dist=0.227649m  scene_diag=1.6755
[V3RS] Run 3  Local  sigma0=0.428499  cma_seed=20262562
[V3RS] Deterministic seed: 20262562
[V3RS] Gen    0  best=0.09051  sigma=0.4052
[V3RS] Gen  100  best=0.06875  sigma=0.0444
[V3RS] === Run 3 Time Breakdown (total 416 ms, 1350 evals) ===
[V3RS]   evaluate_one (sum) : 390 ms (93%)
[V3RS]   rebuild_corr (sum) : 23 ms (5%)  [14 calls]
[V3RS]   cmaes/log/other    : 2 ms (0%)
[V3RS]   build_eval_ctx     : 0 ms   compute_full_rmse  : 212 ms
[V3RS]   --- per-eval breakdown (1350 evals, 1350 with IoU) ---
[V3RS]     rmse_w (V3R-W)    : 18 ms (4%)  avg=13.5 us/eval
[V3RS]     sil_total (raster): 371 ms (94%)  avg=275.0 us/eval
[V3RS]       step1 proj      : 116 ms  avg=86.3 us/eval
[V3RS]       step2 splat(bbox): 223 ms  avg=165.2 us/eval
[V3RS]       step3 iou       : 30 ms  avg=22.6 us/eval
[V3RS]     other (M_srt+pen) : 0 ms  avg=0.3 us/eval
[V3RS] Run 3  best_inner=0.0663677  best_full=0.0572173  stop=MaxGenerations  [-]
[V3RS/sil] Run 3  iou_evals=1350  avg_iou2d=0.6564  avg_sil_loss=0.3436  lambda=4.0000
[V3RS] Run 4/10  Global  sigma0=0.8223  cma_seed=20262563
[V3R] build_eval_context  src=5595  subset=2829  tgt=1430  session_init_matched=940471  post_jitter_matched=1082  matched_min=324  post_jitter_rmse=0.136054  max_dist=0.227649m  scene_diag=1.6755
[V3RS] Run 4  Global  sigma0=0.822304  cma_seed=20262563
[V3RS] Deterministic seed: 20262563
[V3RS] Gen    0  best=0.13605  sigma=0.7285
[V3RS] Gen  100  best=0.09182  sigma=0.0411
[V3RS] === Run 4 Time Breakdown (total 339 ms, 1350 evals) ===
[V3RS]   evaluate_one (sum) : 312 ms (92%)
[V3RS]   rebuild_corr (sum) : 24 ms (7%)  [14 calls]
[V3RS]   cmaes/log/other    : 1 ms (0%)
[V3RS]   build_eval_ctx     : 1 ms   compute_full_rmse  : 238 ms
[V3RS]   --- per-eval breakdown (1350 evals, 1350 with IoU) ---
[V3RS]     rmse_w (V3R-W)    : 18 ms (5%)  avg=13.5 us/eval
[V3RS]     sil_total (raster): 290 ms (92%)  avg=214.9 us/eval
[V3RS]       step1 proj      : 109 ms  avg=80.8 us/eval
[V3RS]       step2 splat(bbox): 151 ms  avg=112.2 us/eval
[V3RS]       step3 iou       : 28 ms  avg=21.1 us/eval
[V3RS]     other (M_srt+pen) : 3 ms  avg=2.4 us/eval
[V3RS] Run 4  best_inner=0.0906143  best_full=0.0810756  stop=MaxGenerations  [-]
[V3RS/sil] Run 4  iou_evals=1350  avg_iou2d=0.6641  avg_sil_loss=0.3359  lambda=4.0000
[V3RS] Run 5/10  Local   sigma0=0.6159  cma_seed=20262564
[V3R] build_eval_context  src=5595  subset=2829  tgt=1430  session_init_matched=940471  post_jitter_matched=1430  matched_min=429  post_jitter_rmse=0.0723021  max_dist=0.227649m  scene_diag=1.6755
[V3RS] Run 5  Local  sigma0=0.61593  cma_seed=20262564
[V3RS] Deterministic seed: 20262564
[V3RS] Gen    0  best=0.07230  sigma=0.5446
[V3RS] Gen  100  best=0.06722  sigma=0.0187
[V3RS] === Run 5 Time Breakdown (total 374 ms, 1350 evals) ===
[V3RS]   evaluate_one (sum) : 352 ms (93%)
[V3RS]   rebuild_corr (sum) : 20 ms (5%)  [14 calls]
[V3RS]   cmaes/log/other    : 1 ms (0%)
[V3RS]   build_eval_ctx     : 1 ms   compute_full_rmse  : 190 ms
[V3RS]   --- per-eval breakdown (1350 evals, 1350 with IoU) ---
[V3RS]     rmse_w (V3R-W)    : 17 ms (5%)  avg=13.2 us/eval
[V3RS]     sil_total (raster): 332 ms (94%)  avg=246.6 us/eval
[V3RS]       step1 proj      : 101 ms  avg=74.9 us/eval
[V3RS]       step2 splat(bbox): 201 ms  avg=149.2 us/eval
[V3RS]       step3 iou       : 29 ms  avg=21.7 us/eval
[V3RS]     other (M_srt+pen) : 0 ms  avg=0.3 us/eval
[V3RS] Run 5  best_inner=0.0650512  best_full=0.05571  stop=MaxGenerations  [-]
[V3RS/sil] Run 5  iou_evals=1350  avg_iou2d=0.7093  avg_sil_loss=0.2907  lambda=4.0000
[V3RS] Run 6/10  Global  sigma0=0.7986  cma_seed=20262565
[V3R] build_eval_context  src=5595  subset=2829  tgt=1430  session_init_matched=940471  post_jitter_matched=1328  matched_min=398  post_jitter_rmse=0.121706  max_dist=0.227649m  scene_diag=1.6755
[V3RS] Run 6  Global  sigma0=0.798631  cma_seed=20262565
[V3RS] Deterministic seed: 20262565
[V3RS] Gen    0  best=0.12171  sigma=0.7324
[V3RS] Gen  100  best=0.09617  sigma=0.0424
[V3RS] === Run 6 Time Breakdown (total 481 ms, 1350 evals) ===
[V3RS]   evaluate_one (sum) : 452 ms (94%)
[V3RS]   rebuild_corr (sum) : 26 ms (5%)  [14 calls]
[V3RS]   cmaes/log/other    : 2 ms (0%)
[V3RS]   build_eval_ctx     : 1 ms   compute_full_rmse  : 308 ms
[V3RS]   --- per-eval breakdown (1350 evals, 1350 with IoU) ---
[V3RS]     rmse_w (V3R-W)    : 19 ms (4%)  avg=14.4 us/eval
[V3RS]     sil_total (raster): 431 ms (95%)  avg=319.8 us/eval
[V3RS]       step1 proj      : 128 ms  avg=95.1 us/eval
[V3RS]       step2 splat(bbox): 271 ms  avg=200.8 us/eval
[V3RS]       step3 iou       : 31 ms  avg=23.0 us/eval
[V3RS]     other (M_srt+pen) : 0 ms  avg=0.3 us/eval
[V3RS] Run 6  best_inner=0.0933821  best_full=0.0907331  stop=MaxGenerations  [-]
[V3RS/sil] Run 6  iou_evals=1350  avg_iou2d=0.5331  avg_sil_loss=0.4669  lambda=4.0000
[V3RS] Run 7/10  Local   sigma0=0.6961  cma_seed=20262566
[V3R] build_eval_context  src=5595  subset=2829  tgt=1430  session_init_matched=940471  post_jitter_matched=1366  matched_min=409  post_jitter_rmse=0.0948144  max_dist=0.227649m  scene_diag=1.6755
[V3RS] Run 7  Local  sigma0=0.696137  cma_seed=20262566
[V3RS] Deterministic seed: 20262566
[V3RS] Gen    0  best=0.09481  sigma=0.6339
[V3RS] Gen  100  best=0.06662  sigma=0.0452
[V3RS] === Run 7 Time Breakdown (total 476 ms, 1350 evals) ===
[V3RS]   evaluate_one (sum) : 449 ms (94%)
[V3RS]   rebuild_corr (sum) : 24 ms (5%)  [14 calls]
[V3RS]   cmaes/log/other    : 2 ms (0%)
[V3RS]   build_eval_ctx     : 1 ms   compute_full_rmse  : 210 ms
[V3RS]   --- per-eval breakdown (1350 evals, 1350 with IoU) ---
[V3RS]     rmse_w (V3R-W)    : 18 ms (4%)  avg=14.0 us/eval
[V3RS]     sil_total (raster): 429 ms (95%)  avg=318.0 us/eval
[V3RS]       step1 proj      : 124 ms  avg=92.3 us/eval
[V3RS]       step2 splat(bbox): 272 ms  avg=201.8 us/eval
[V3RS]       step3 iou       : 31 ms  avg=23.0 us/eval
[V3RS]     other (M_srt+pen) : 0 ms  avg=0.3 us/eval
[V3RS] Run 7  best_inner=0.0659542  best_full=0.0564047  stop=MaxGenerations  [-]
[V3RS/sil] Run 7  iou_evals=1350  avg_iou2d=0.6452  avg_sil_loss=0.3548  lambda=4.0000
[V3RS] Run 8/10  Global  sigma0=0.9615  cma_seed=20262567
[V3R] build_eval_context  src=5595  subset=2829  tgt=1430  session_init_matched=940471  post_jitter_matched=1269  matched_min=380  post_jitter_rmse=0.126106  max_dist=0.227649m  scene_diag=1.6755
[V3RS] Run 8  Global  sigma0=0.961471  cma_seed=20262567
[V3RS] Deterministic seed: 20262567
[V3RS] Gen    0  best=0.12611  sigma=0.8828
[V3RS] Gen  100  best=0.10634  sigma=0.0558
[V3RS] === Run 8 Time Breakdown (total 363 ms, 1350 evals) ===
[V3RS]   evaluate_one (sum) : 333 ms (91%)
[V3RS]   rebuild_corr (sum) : 27 ms (7%)  [14 calls]
[V3RS]   cmaes/log/other    : 2 ms (0%)
[V3RS]   build_eval_ctx     : 1 ms   compute_full_rmse  : 349 ms
[V3RS]   --- per-eval breakdown (1350 evals, 1350 with IoU) ---
[V3RS]     rmse_w (V3R-W)    : 18 ms (5%)  avg=13.7 us/eval
[V3RS]     sil_total (raster): 313 ms (94%)  avg=232.3 us/eval
[V3RS]       step1 proj      : 94 ms  avg=70.3 us/eval
[V3RS]       step2 splat(bbox): 188 ms  avg=139.9 us/eval
[V3RS]       step3 iou       : 28 ms  avg=21.2 us/eval
[V3RS]     other (M_srt+pen) : 0 ms  avg=0.3 us/eval
[V3RS] Run 8  best_inner=0.106338  best_full=0.107394  stop=MaxGenerations  [-]
[V3RS/sil] Run 8  iou_evals=1350  avg_iou2d=0.3615  avg_sil_loss=0.6385  lambda=4.0000
[V3RS] Run 9/10  Local   sigma0=0.4783  cma_seed=20262568
[V3R] build_eval_context  src=5595  subset=2829  tgt=1430  session_init_matched=940471  post_jitter_matched=1430  matched_min=429  post_jitter_rmse=0.0903593  max_dist=0.227649m  scene_diag=1.6755
[V3RS] Run 9  Local  sigma0=0.478329  cma_seed=20262568
[V3RS] Deterministic seed: 20262568
[V3RS] Gen    0  best=0.09036  sigma=0.4433
[V3RS] Gen  100  best=0.06480  sigma=0.0284
[V3RS] === Run 9 Time Breakdown (total 292 ms, 1350 evals) ===
[V3RS]   evaluate_one (sum) : 269 ms (91%)
[V3RS]   rebuild_corr (sum) : 21 ms (7%)  [14 calls]
[V3RS]   cmaes/log/other    : 1 ms (0%)
[V3RS]   build_eval_ctx     : 1 ms   compute_full_rmse  : 189 ms
[V3RS]   --- per-eval breakdown (1350 evals, 1350 with IoU) ---
[V3RS]     rmse_w (V3R-W)    : 17 ms (6%)  avg=13.3 us/eval
[V3RS]     sil_total (raster): 250 ms (92%)  avg=185.2 us/eval
[V3RS]       step1 proj      : 63 ms  avg=46.9 us/eval
[V3RS]       step2 splat(bbox): 156 ms  avg=115.7 us/eval
[V3RS]       step3 iou       : 29 ms  avg=21.7 us/eval
[V3RS]     other (M_srt+pen) : 0 ms  avg=0.3 us/eval
[V3RS] Run 9  best_inner=0.0633561  best_full=0.0545517  stop=MaxGenerations  [-]
[V3RS/sil] Run 9  iou_evals=1350  avg_iou2d=0.6463  avg_sil_loss=0.3537  lambda=4.0000
[V3RS] Run 10/10  Global  sigma0=0.6873  cma_seed=20262569
[V3R] build_eval_context  src=5595  subset=2829  tgt=1430  session_init_matched=940471  post_jitter_matched=0  matched_min=10  post_jitter_rmse=0  max_dist=0.227649m  scene_diag=1.6755
[V3RS] jitter retry 1/3: sigma_factor=0.5  post_jitter_matched=0 < 141  (Q=Q:AR+AL)
[V3R] build_eval_context  src=5595  subset=2829  tgt=1430  session_init_matched=940471  post_jitter_matched=1090  matched_min=327  post_jitter_rmse=0.162735  max_dist=0.227649m  scene_diag=1.6755
[V3RS] Run 10  Global  sigma0=0.687308  cma_seed=20262569
[V3RS] Deterministic seed: 20262569
[V3RS] Gen    0  best=0.16274  sigma=0.6309
[V3RS] Gen  100  best=0.07973  sigma=0.0333
[V3RS] === Run 10 Time Breakdown (total 393 ms, 1350 evals) ===
[V3RS]   evaluate_one (sum) : 369 ms (93%)
[V3RS]   rebuild_corr (sum) : 22 ms (5%)  [14 calls]
[V3RS]   cmaes/log/other    : 2 ms (0%)
[V3RS]   build_eval_ctx     : 3 ms   compute_full_rmse  : 177 ms
[V3RS]   --- per-eval breakdown (1350 evals, 1350 with IoU) ---
[V3RS]     rmse_w (V3R-W)    : 18 ms (4%)  avg=13.4 us/eval
[V3RS]     sil_total (raster): 349 ms (94%)  avg=259.1 us/eval
[V3RS]       step1 proj      : 106 ms  avg=79.0 us/eval
[V3RS]       step2 splat(bbox): 208 ms  avg=154.4 us/eval
[V3RS]       step3 iou       : 33 ms  avg=24.8 us/eval
[V3RS]     other (M_srt+pen) : 0 ms  avg=0.3 us/eval
[V3RS] Run 10  best_inner=0.0720385  best_full=0.0636451  stop=MaxGenerations  [-]
[V3RS/sil] Run 10  iou_evals=1350  avg_iou2d=0.5532  avg_sil_loss=0.4468  lambda=4.0000
[V3RS session/Time] runs loop wall-clock : 6152.1 ms  (sum of per-run outer = 6152.0 ms)
[V3RS/Selector] Run 1: full=0.0526  IoU2D=0.6568  scale=1.0000  jitter_scale=1.0000
[V3RS/Selector] Run 2: full=0.1007  IoU2D=0.4444  scale=1.0002  jitter_scale=1.0089
[V3RS/Selector] Run 3: full=0.0572  IoU2D=0.6210  scale=1.0000  jitter_scale=0.9666
[V3RS/Selector] Run 4: full=0.0811  IoU2D=0.5862  scale=1.0000  jitter_scale=0.9201
[V3RS/Selector] Run 5: full=0.0557  IoU2D=0.6616  scale=1.0000  jitter_scale=1.0220
[V3RS/Selector] Run 6: full=0.0907  IoU2D=0.3629  scale=1.0000  jitter_scale=1.0559
[V3RS/Selector] Run 7: full=0.0564  IoU2D=0.6180  scale=1.0000  jitter_scale=0.9514
[V3RS/Selector] Run 8: full=0.1074  IoU2D=0.3139  scale=0.9998  jitter_scale=0.9193
[V3RS/Selector] Run 9: full=0.0546  IoU2D=0.6184  scale=1.0000  jitter_scale=0.9861
[V3RS/Selector] Run 10: full=0.0636  IoU2D=0.6078  scale=1.0000  jitter_scale=1.0242
[V3RS/Selector] best by full  : Run 1  (full=0.0526  IoU=0.6568)
[V3RS/Selector] best by IoU   : Run 5  (full=0.0557  IoU=0.6616)   <-- differs from full!
[V3RS/Selector] best by combo : Run 1  (combo=0.2241, gamma=0.5000)
[V3RS/Selector] DECISION: no Run improved on rmse_before=0.0525 (argmin full = Run 1)
[V3RS/Selector] selector cost : 25.9 ms  (10 IoU2D evals, step=8)
[V3RS] === BIPOP-CMA-ES V3R DONE ===  best_run=none  RMSE: 0.0524606 -> 0.0524606  delta=0  [NO CHANGE]  total_gens=1500  quadrant=Q:AR+AL
[V3RS session/Time] DRIVER TOTAL : 6206.5 ms
[CtrlGS/Time] D. CMA-ES V3R driver (10 runs + voxel + subset) : 6208.9 ms  (cumulative 6397.7 ms)
[CtrlGS/Time]   (no improvement, skipped apply)
[CtrlGS/Time] E. apply_matrix + setUp x6 : 0.0 ms  (cumulative 6397.7 ms)
[CtrlGS/Time] E5. capture sil projection (viz) : 0.0 ms  (cumulative 6397.7 ms)
[CtrlGS/Time] E6. final-pose hitmap dump (viz) : 1.3 ms  (cumulative 6399.0 ms)
  [Cached] returning injected target cloud (940471 points, boundaryDist: YES)
[Boundary3D] accepted=42184 rejected_by_instrument=22559 interior=875728  (instPxTh=8e+01, instMask=YES)
[Metrics] matched=940471  RMSE=0.05  avg=0.04  max=0.2
[Hausdorff2D] IoU=0.7  H2D=240.0px  (cost=7.21ms, step=8)
[CtrlGS/Time] F. post_computeUnifiedMetrics : 223.5 ms  (cumulative 6622.5 ms)
[Ctrl+Shift+G] Best: 0.0524606 -> 0.0524606 (delta=0)  best_run=none  total_gens=1500  Q=Q:AR+AL  [NO CHANGE]
[CtrlGS/Time] === GRAND TOTAL: 6622.5 ms ===
  [Cached] returning injected target cloud (940471 points, boundaryDist: YES)
[Boundary3D] accepted=42184 rejected_by_instrument=22559 interior=875728  (instPxTh=8e+01, instMask=YES)
[Metrics] matched=940471  RMSE=0.05  avg=0.04  max=0.2
[Hausdorff2D] IoU=0.7  H2D=240.0px  (cost=7.51ms, step=8)
[Shift+Alt+G] --- running V3RSB (vtx squash lv0) ---

=== V3RS-B (Alt+G) vertex-squash rasterizer ===
[Alt+G] quadrant_mask = Q:AR+AL  (0x3)  subdiv_level=0
[Seed Alt+G] BIPOP outer=20262778  CMA-ES base=20262560  (trial=20260420, callIdx=14)
  [Cached] returning injected target cloud (940471 points, boundaryDist: YES)
[Boundary3D] accepted=42184 rejected_by_instrument=22559 interior=875728  (instPxTh=80, instMask=YES)
[Metrics] matched=940471  RMSE=0.0524606  avg=0.0448445  max=0.188657
[Hausdorff2D] IoU=0.660188  H2D=240.0px  (cost=6.96ms, step=8)
[Alt+G] start RMSE=0.0524606  init_matched=940471
  [Cached] returning injected target cloud (940471 points, boundaryDist: YES)
[Alt+G/sil] quad-filter tris: 9151 / 20000  (Q=Q:AR+AL)
[Alt+G/sil] ON  lambda_sil=4.0000  img=1920x1080  tris=9151  raster_step=16  subdiv_level=0  edge_thresh=2.0000cells
  [Custom] VoxelDownSample: 9992 -> 5595 (voxel=0.0251324)
  [Custom] VoxelDownSample: 940471 -> 1430 (voxel=0.0251324)
[V3RSB session/Time] voxel src (9992->5595) : 1.8 ms
[V3RSB session/Time] voxel tgt (940471->1430) : 22.5 ms
[V3RSB session/Time] voxel_to_orig (5595 NN lookups) : 3.2 ms
[V3RSB session/Time] subset_filter : 0.7 ms
[V3RSB] quadrant_mask=Q:AR+AL  arvis=OFF  caudal=OFF  combine=AND  subset_size=2829/5595 (voxel-space)
[V3RSB-W/sil] lambda_sil=4  voxel_total=5595  tris=9151  eval_interval=1  raster_step=16  dist_map=YES  active=ON
[V3RSB] === Starting BIPOP-CMA-ES V3R ===
[V3RSB] outer_seed=20262778  cma_base=20262560  rmse_before=0.05  init_matched=940471  scene_diag=2
[V3RSB] src: 9992 -> 5595 (voxel=0.03, ratio=0.01)
[V3RSB] tgt: 940471 -> 1430 (voxel=0.03, ratio=0.01)
[V3RSB-W/sil] tolfun override: 1e-06 -> 1e-05  (sil cost surface is at a different scale; default would early-stop)
[V3RSB] Run 1/10  Local   sigma0=0.3327  cma_seed=20262560
[V3R] build_eval_context  src=5595  subset=2829  tgt=1430  session_init_matched=940471  post_jitter_matched=1430  matched_min=429  post_jitter_rmse=0.0622367  max_dist=0.227649m  scene_diag=1.6755
[V3RSB] Run 1  Local  sigma0=0.332739  cma_seed=20262560
[V3RSB] Deterministic seed: 20262560
[V3RS/cov-diag] === Coverage diagnostic (Run 1, initial pose) ===
[V3RS/cov-diag]   grid=120x68  tris=9151  cells_on=3575/8160  cells_on(no-halo)=3238
[V3RS/cov-diag]   total bbox writes : 167165   avg coverage = 46.76 writes/on-cell
[V3RS/cov-diag]   cov==1 cells      : 26  (0.7% of on-cells)   [coverage-based, halo-fooled]
[V3RS/cov-diag]   cov>=2 cells      : 3549  (99.3% of on-cells)
[V3RS/cov-diag]   GEOMETRIC boundary: 274  (7.7% of on-cells)   [4-neighbour, halo-proof]
[V3RS/cov-diag]   GEOMETRIC interior: 3301  (92.3% of on-cells)
[V3RS/cov-diag]   coverage histogram (on-cells):
[V3RS/cov-diag]     cov==1 : 26
[V3RS/cov-diag]     cov==2 : 33
[V3RS/cov-diag]     cov==3 : 42
[V3RS/cov-diag]     cov==4 : 24
[V3RS/cov-diag]     cov==5 : 27
[V3RS/cov-diag]     cov 6-10 : 141
[V3RS/cov-diag]     cov 11+  : 3282   (max coverage = 227)
[V3RS/cov-diag]   --- per-row span structure ---
[V3RS/cov-diag]     rows with on-cells : 55
[V3RS/cov-diag]     single-run rows    : 36   (naive scanline OK)
[V3RS/cov-diag]     multi-run rows     : 19   (naive scanline would overfill)
[V3RS/cov-diag]     max runs in a row  : 3
[V3RS/cov-diag]     scanline overfill  : 319 cells would be wrongly filled
[V3RS/cov-diag]   flood-fill estimate: ideal writes = 3575  vs current = 167165  -> up to 46.76x fewer splat writes
[V3RS/cov-diag]   --- ASCII silhouette map (120x68, legend: ' '=off .=1 :=2-3 +=4-10 #=11+) ---
[V3RS/cov-diag]   |                                    :+####################+                                                             |
[V3RS/cov-diag]   |                                   +#######################                                                             |
[V3RS/cov-diag]   |                                 +##########################                                                            |
[V3RS/cov-diag]   |                                :#############################+                                                         |
[V3RS/cov-diag]   |                                ###############################+                                                        |
[V3RS/cov-diag]   |                              :+###################################+:                                                   |
[V3RS/cov-diag]   |                              +#####################################++                                                  |
[V3RS/cov-diag]   |                              ########################################++                                                |
[V3RS/cov-diag]   |                              ##########################################++: +#######+                                   |
[V3RS/cov-diag]   |                             +#########################################################++                               |
[V3RS/cov-diag]   |                             ############################################################++                             |
[V3RS/cov-diag]   |                             ##############################################################+:                           |
[V3RS/cov-diag]   |                            +#################################################################+++::::::                 |
[V3RS/cov-diag]   |                         ::+#####################################################################::::::                 |
[V3RS/cov-diag]   |                         :+#######################################################################+++++                 |
[V3RS/cov-diag]   |                         :+############################################################################+:               |
[V3RS/cov-diag]   |                         :#############################################################################++               |
[V3RS/cov-diag]   |                         :##############################################################################+               |
[V3RS/cov-diag]   |                        :###############################################################################+               |
[V3RS/cov-diag]   |                        :################################################################################++::           |
[V3RS/cov-diag]   |                        +##################################################################################+:           |
[V3RS/cov-diag]   |                        ####################################################################################+           |
[V3RS/cov-diag]   |                       +######################################################################################++        |
[V3RS/cov-diag]   |                      :#########################################################################################+:      |
[V3RS/cov-diag]   |                      :##########################################################################################:      |
[V3RS/cov-diag]   |                      +#######################+##########+++++####################################################      |
[V3RS/cov-diag]   |                      ###############################+#######+####+###############################################+     |
[V3RS/cov-diag]   |                      ###############################+############+################################################     |
[V3RS/cov-diag]   |                     +###############################++############################################################     |
[V3RS/cov-diag]   |                     ################################+#############################################################     |
[V3RS/cov-diag]   |                     ##############################################################################################     |
[V3RS/cov-diag]   |                     ##############################################################################################     |
[V3RS/cov-diag]   |                     #####################################++#######################################################     |
[V3RS/cov-diag]   |                    :###############################+#######################++#############+#######################     |
[V3RS/cov-diag]   |                    +####################################################################++:#######################     |
[V3RS/cov-diag]   |                    #########################################################++#########++:.+#####################+     |
[V3RS/cov-diag]   |                    ###################################################################+++  ++####################:     |
[V3RS/cov-diag]   |                   +######################+:##########+###############################+:    .+###################+      |
[V3RS/cov-diag]   |                   +#####################+. ########################################+++:     :+#################+:      |
[V3RS/cov-diag]   |                   ####################+:   +#####################################++:.        +++##############+.       |
[V3RS/cov-diag]   |                   ##################++.     :####################################++:.        .:::+++#########+         |
[V3RS/cov-diag]   |                   ################+.         ####################################+.            ..:+++#######+:         |
[V3RS/cov-diag]   |                   ##############+:.          ######################+#####+######++.                .:++#####:          |
[V3RS/cov-diag]   |                   #############++            +#################################+:.                                     |
[V3RS/cov-diag]   |                   #############+:             ###################+#+##########++:.                                     |
[V3RS/cov-diag]   |                   +###########++:             +#######++#####################++::.                                     |
[V3RS/cov-diag]   |                    ###########++:              +###++..+######+++++#########+::                                        |
[V3RS/cov-diag]   |                    ##########+++.                      +##################+:.                                          |
[V3RS/cov-diag]   |                    ##########+++                       +################++                                             |
[V3RS/cov-diag]   |                    ##########+..                       +###############+::                                             |
[V3RS/cov-diag]   |                    #########+                          :+############+::                                               |
[V3RS/cov-diag]   |                    +######++:                           :+##########++                                                 |
[V3RS/cov-diag]   |                    +######++.                            :++#######+:                                                  |
[V3RS/cov-diag]   |                     #####+                                  :++####+.                                                  |
[V3RS/cov-diag]   |                     +###+                                                                                              |
[V3RS/cov-diag]   |                                                                                                                        |
[V3RS/cov-diag]   |                                                                                                                        |
[V3RS/cov-diag]   |                                                                                                                        |
[V3RS/cov-diag]   |                                                                                                                        |
[V3RS/cov-diag]   |                                                                                                                        |
[V3RS/cov-diag]   |                                                                                                                        |
[V3RS/cov-diag]   |                                                                                                                        |
[V3RS/cov-diag]   |                                                                                                                        |
[V3RS/cov-diag]   |                                                                                                                        |
[V3RS/cov-diag]   |                                                                                                                        |
[V3RS/cov-diag]   |                                                                                                                        |
[V3RS/cov-diag]   |                                                                                                                        |
[V3RS/cov-diag]   |                                                                                                                        |
[V3RSB] Gen    0  best=0.06224  sigma=0.2832
[V3RSB] Gen  100  best=0.06224  sigma=0.0043
[V3RSB] === Run 1 Time Breakdown (total 494 ms, 1350 evals) ===
[V3RSB]   evaluate_one (sum) : 473 ms (95%)
[V3RSB]   rebuild_corr (sum) : 19 ms (3%)  [14 calls]
[V3RSB]   cmaes/log/other    : 1 ms (0%)
[V3RSB]   build_eval_ctx     : 1 ms   compute_full_rmse  : 187 ms
[V3RSB]   --- per-eval breakdown (1350 evals, 1350 with IoU) ---
[V3RSB]     rmse_w (V3R-W)    : 16 ms (3%)  avg=12.3 us/eval
[V3RSB]     sil_total (raster): 455 ms (96%)  avg=337.7 us/eval
[V3RSB]       step1 proj      : 204 ms  avg=151.2 us/eval
[V3RSB]       step2 splat(bbox): 228 ms  avg=169.0 us/eval
[V3RSB]       step3 iou       : 23 ms  avg=17.1 us/eval
[V3RSB]     other (M_srt+pen) : 0 ms  avg=0.3 us/eval
[V3RSB] Run 1  best_inner=0.0622367  best_full=0.0524606  stop=MaxGenerations  [-]
[V3RSB/sil] Run 1  iou_evals=1350  avg_iou2d=0.6810  avg_sil_loss=0.3190  lambda=4.0000
[V3RSB] Run 2/10  Global  sigma0=0.5365  cma_seed=20262561
[V3R] build_eval_context  src=5595  subset=2829  tgt=1430  session_init_matched=940471  post_jitter_matched=1192  matched_min=357  post_jitter_rmse=0.115901  max_dist=0.227649m  scene_diag=1.6755
[V3RSB] Run 2  Global  sigma0=0.536453  cma_seed=20262561
[V3RSB] Deterministic seed: 20262561
[V3RSB] Gen    0  best=0.11590  sigma=0.4672
[V3RSB] Gen  100  best=0.10799  sigma=0.0077
[V3RSB] === Run 2 Time Breakdown (total 378 ms, 1350 evals) ===
[V3RSB]   evaluate_one (sum) : 350 ms (92%)
[V3RSB]   rebuild_corr (sum) : 26 ms (7%)  [14 calls]
[V3RSB]   cmaes/log/other    : 1 ms (0%)
[V3RSB]   build_eval_ctx     : 2 ms   compute_full_rmse  : 358 ms
[V3RSB]   --- per-eval breakdown (1350 evals, 1350 with IoU) ---
[V3RSB]     rmse_w (V3R-W)    : 16 ms (4%)  avg=11.9 us/eval
[V3RSB]     sil_total (raster): 333 ms (95%)  avg=246.7 us/eval
[V3RSB]       step1 proj      : 94 ms  avg=69.7 us/eval
[V3RSB]       step2 splat(bbox): 213 ms  avg=158.1 us/eval
[V3RSB]       step3 iou       : 25 ms  avg=18.6 us/eval
[V3RSB]     other (M_srt+pen) : 0 ms  avg=0.3 us/eval
[V3RSB] Run 2  best_inner=0.107985  best_full=0.1026  stop=MaxGenerations  [-]
[V3RSB/sil] Run 2  iou_evals=1350  avg_iou2d=0.4923  avg_sil_loss=0.5077  lambda=4.0000
[V3RSB] Run 3/10  Local   sigma0=0.4285  cma_seed=20262562
[V3R] build_eval_context  src=5595  subset=2829  tgt=1430  session_init_matched=940471  post_jitter_matched=1425  matched_min=427  post_jitter_rmse=0.090509  max_dist=0.227649m  scene_diag=1.6755
[V3RSB] Run 3  Local  sigma0=0.428499  cma_seed=20262562
[V3RSB] Deterministic seed: 20262562
[V3RSB] Gen    0  best=0.09051  sigma=0.4052
[V3RSB] Gen  100  best=0.06898  sigma=0.0191
[V3RSB] === Run 3 Time Breakdown (total 436 ms, 1350 evals) ===
[V3RSB]   evaluate_one (sum) : 414 ms (94%)
[V3RSB]   rebuild_corr (sum) : 20 ms (4%)  [14 calls]
[V3RSB]   cmaes/log/other    : 1 ms (0%)
[V3RSB]   build_eval_ctx     : 1 ms   compute_full_rmse  : 179 ms
[V3RSB]   --- per-eval breakdown (1350 evals, 1350 with IoU) ---
[V3RSB]     rmse_w (V3R-W)    : 16 ms (4%)  avg=12.4 us/eval
[V3RSB]     sil_total (raster): 396 ms (95%)  avg=293.9 us/eval
[V3RSB]       step1 proj      : 139 ms  avg=103.5 us/eval
[V3RSB]       step2 splat(bbox): 232 ms  avg=172.3 us/eval
[V3RSB]       step3 iou       : 23 ms  avg=17.8 us/eval
[V3RSB]     other (M_srt+pen) : 0 ms  avg=0.3 us/eval
[V3RSB] Run 3  best_inner=0.0660882  best_full=0.0574897  stop=MaxGenerations  [-]
[V3RSB/sil] Run 3  iou_evals=1350  avg_iou2d=0.6299  avg_sil_loss=0.3701  lambda=4.0000
[V3RSB] Run 4/10  Global  sigma0=0.8223  cma_seed=20262563
[V3R] build_eval_context  src=5595  subset=2829  tgt=1430  session_init_matched=940471  post_jitter_matched=1082  matched_min=324  post_jitter_rmse=0.136054  max_dist=0.227649m  scene_diag=1.6755
[V3RSB] Run 4  Global  sigma0=0.822304  cma_seed=20262563
[V3RSB] Deterministic seed: 20262563
[V3RSB] Gen    0  best=0.13605  sigma=0.7363
[V3RSB] Gen  100  best=0.09449  sigma=0.0525
[V3RSB] === Run 4 Time Breakdown (total 389 ms, 1350 evals) ===
[V3RSB]   evaluate_one (sum) : 365 ms (93%)
[V3RSB]   rebuild_corr (sum) : 22 ms (5%)  [14 calls]
[V3RSB]   cmaes/log/other    : 1 ms (0%)
[V3RSB]   build_eval_ctx     : 1 ms   compute_full_rmse  : 229 ms
[V3RSB]   --- per-eval breakdown (1350 evals, 1350 with IoU) ---
[V3RSB]     rmse_w (V3R-W)    : 16 ms (4%)  avg=12.0 us/eval
[V3RSB]     sil_total (raster): 347 ms (95%)  avg=257.5 us/eval
[V3RSB]       step1 proj      : 113 ms  avg=83.9 us/eval
[V3RSB]       step2 splat(bbox): 210 ms  avg=156.0 us/eval
[V3RSB]       step3 iou       : 23 ms  avg=17.2 us/eval
[V3RSB]     other (M_srt+pen) : 0 ms  avg=0.3 us/eval
[V3RSB] Run 4  best_inner=0.0914621  best_full=0.0833391  stop=MaxGenerations  [-]
[V3RSB/sil] Run 4  iou_evals=1350  avg_iou2d=0.6626  avg_sil_loss=0.3374  lambda=4.0000
[V3RSB] Run 5/10  Local   sigma0=0.6159  cma_seed=20262564
[V3R] build_eval_context  src=5595  subset=2829  tgt=1430  session_init_matched=940471  post_jitter_matched=1430  matched_min=429  post_jitter_rmse=0.0723021  max_dist=0.227649m  scene_diag=1.6755
[V3RSB] Run 5  Local  sigma0=0.61593  cma_seed=20262564
[V3RSB] Deterministic seed: 20262564
[V3RSB] Gen    0  best=0.07230  sigma=0.5446
[V3RSB] Gen  100  best=0.06693  sigma=0.0132
[V3RSB] === Run 5 Time Breakdown (total 479 ms, 1350 evals) ===
[V3RSB]   evaluate_one (sum) : 459 ms (95%)
[V3RSB]   rebuild_corr (sum) : 19 ms (4%)  [14 calls]
[V3RSB]   cmaes/log/other    : 1 ms (0%)
[V3RSB]   build_eval_ctx     : 0 ms   compute_full_rmse  : 168 ms
[V3RSB]   --- per-eval breakdown (1350 evals, 1350 with IoU) ---
[V3RSB]     rmse_w (V3R-W)    : 16 ms (3%)  avg=12.3 us/eval
[V3RSB]     sil_total (raster): 441 ms (96%)  avg=326.9 us/eval
[V3RSB]       step1 proj      : 192 ms  avg=142.2 us/eval
[V3RSB]       step2 splat(bbox): 225 ms  avg=166.7 us/eval
[V3RSB]       step3 iou       : 23 ms  avg=17.6 us/eval
[V3RSB]     other (M_srt+pen) : 0 ms  avg=0.3 us/eval
[V3RSB] Run 5  best_inner=0.0651265  best_full=0.0562052  stop=MaxGenerations  [-]
[V3RSB/sil] Run 5  iou_evals=1350  avg_iou2d=0.6710  avg_sil_loss=0.3290  lambda=4.0000
[V3RSB] Run 6/10  Global  sigma0=0.7986  cma_seed=20262565
[V3R] build_eval_context  src=5595  subset=2829  tgt=1430  session_init_matched=940471  post_jitter_matched=1328  matched_min=398  post_jitter_rmse=0.121706  max_dist=0.227649m  scene_diag=1.6755
[V3RSB] Run 6  Global  sigma0=0.798631  cma_seed=20262565
[V3RSB] Deterministic seed: 20262565
[V3RSB] Gen    0  best=0.12171  sigma=0.7182
[V3RSB] Gen  100  best=0.09182  sigma=0.1588
[V3RSB] === Run 6 Time Breakdown (total 416 ms, 1350 evals) ===
[V3RSB]   evaluate_one (sum) : 392 ms (94%)
[V3RSB]   rebuild_corr (sum) : 21 ms (5%)  [14 calls]
[V3RSB]   cmaes/log/other    : 1 ms (0%)
[V3RSB]   build_eval_ctx     : 1 ms   compute_full_rmse  : 226 ms
[V3RSB]   --- per-eval breakdown (1350 evals, 1350 with IoU) ---
[V3RSB]     rmse_w (V3R-W)    : 16 ms (4%)  avg=12.3 us/eval
[V3RSB]     sil_total (raster): 374 ms (95%)  avg=277.7 us/eval
[V3RSB]       step1 proj      : 122 ms  avg=90.9 us/eval
[V3RSB]       step2 splat(bbox): 226 ms  avg=168.1 us/eval
[V3RSB]       step3 iou       : 24 ms  avg=18.3 us/eval
[V3RSB]     other (M_srt+pen) : 0 ms  avg=0.3 us/eval
[V3RSB] Run 6  best_inner=0.0905959  best_full=0.090088  stop=MaxGenerations  [-]
[V3RSB/sil] Run 6  iou_evals=1350  avg_iou2d=0.4753  avg_sil_loss=0.5247  lambda=4.0000
[V3RSB] Run 7/10  Local   sigma0=0.6961  cma_seed=20262566
[V3R] build_eval_context  src=5595  subset=2829  tgt=1430  session_init_matched=940471  post_jitter_matched=1366  matched_min=409  post_jitter_rmse=0.0948144  max_dist=0.227649m  scene_diag=1.6755
[V3RSB] Run 7  Local  sigma0=0.696137  cma_seed=20262566
[V3RSB] Deterministic seed: 20262566
[V3RSB] Gen    0  best=0.09481  sigma=0.6339
[V3RSB] Gen  100  best=0.06593  sigma=0.0144
[V3RSB] === Run 7 Time Breakdown (total 497 ms, 1350 evals) ===
[V3RSB]   evaluate_one (sum) : 474 ms (95%)
[V3RSB]   rebuild_corr (sum) : 21 ms (4%)  [14 calls]
[V3RSB]   cmaes/log/other    : 1 ms (0%)
[V3RSB]   build_eval_ctx     : 1 ms   compute_full_rmse  : 176 ms
[V3RSB]   --- per-eval breakdown (1350 evals, 1350 with IoU) ---
[V3RSB]     rmse_w (V3R-W)    : 16 ms (3%)  avg=12.5 us/eval
[V3RSB]     sil_total (raster): 456 ms (96%)  avg=338.1 us/eval
[V3RSB]       step1 proj      : 197 ms  avg=146.2 us/eval
[V3RSB]       step2 splat(bbox): 234 ms  avg=173.7 us/eval
[V3RSB]       step3 iou       : 23 ms  avg=17.8 us/eval
[V3RSB]     other (M_srt+pen) : 0 ms  avg=0.3 us/eval
[V3RSB] Run 7  best_inner=0.0657491  best_full=0.0562064  stop=MaxGenerations  [-]
[V3RSB/sil] Run 7  iou_evals=1350  avg_iou2d=0.6054  avg_sil_loss=0.3946  lambda=4.0000
[V3RSB] Run 8/10  Global  sigma0=0.9615  cma_seed=20262567
[V3R] build_eval_context  src=5595  subset=2829  tgt=1430  session_init_matched=940471  post_jitter_matched=1269  matched_min=380  post_jitter_rmse=0.126106  max_dist=0.227649m  scene_diag=1.6755
[V3RSB] Run 8  Global  sigma0=0.961471  cma_seed=20262567
[V3RSB] Deterministic seed: 20262567
[V3RSB] Gen    0  best=0.12611  sigma=0.8828
[V3RSB] Gen  100  best=0.10681  sigma=0.0554
[V3RSB] === Run 8 Time Breakdown (total 373 ms, 1350 evals) ===
[V3RSB]   evaluate_one (sum) : 347 ms (92%)
[V3RSB]   rebuild_corr (sum) : 24 ms (6%)  [14 calls]
[V3RSB]   cmaes/log/other    : 1 ms (0%)
[V3RSB]   build_eval_ctx     : 1 ms   compute_full_rmse  : 377 ms
[V3RSB]   --- per-eval breakdown (1350 evals, 1350 with IoU) ---
[V3RSB]     rmse_w (V3R-W)    : 16 ms (4%)  avg=12.2 us/eval
[V3RSB]     sil_total (raster): 329 ms (94%)  avg=244.1 us/eval
[V3RSB]       step1 proj      : 98 ms  avg=72.7 us/eval
[V3RSB]       step2 splat(bbox): 206 ms  avg=152.7 us/eval
[V3RSB]       step3 iou       : 24 ms  avg=18.3 us/eval
[V3RSB]     other (M_srt+pen) : 0 ms  avg=0.3 us/eval
[V3RSB] Run 8  best_inner=0.105778  best_full=0.107052  stop=MaxGenerations  [-]
[V3RSB/sil] Run 8  iou_evals=1350  avg_iou2d=0.3311  avg_sil_loss=0.6689  lambda=4.0000
[V3RSB] Run 9/10  Local   sigma0=0.4783  cma_seed=20262568
[V3R] build_eval_context  src=5595  subset=2829  tgt=1430  session_init_matched=940471  post_jitter_matched=1430  matched_min=429  post_jitter_rmse=0.0903593  max_dist=0.227649m  scene_diag=1.6755
[V3RSB] Run 9  Local  sigma0=0.478329  cma_seed=20262568
[V3RSB] Deterministic seed: 20262568
[V3RSB] Gen    0  best=0.09036  sigma=0.4487
[V3RSB] Gen  100  best=0.06452  sigma=0.0256
[V3RSB] === Run 9 Time Breakdown (total 523 ms, 1350 evals) ===
[V3RSB]   evaluate_one (sum) : 500 ms (95%)
[V3RSB]   rebuild_corr (sum) : 20 ms (4%)  [14 calls]
[V3RSB]   cmaes/log/other    : 1 ms (0%)
[V3RSB]   build_eval_ctx     : 1 ms   compute_full_rmse  : 207 ms
[V3RSB]   --- per-eval breakdown (1350 evals, 1350 with IoU) ---
[V3RSB]     rmse_w (V3R-W)    : 16 ms (3%)  avg=12.3 us/eval
[V3RSB]     sil_total (raster): 483 ms (96%)  avg=357.8 us/eval
[V3RSB]       step1 proj      : 232 ms  avg=172.2 us/eval
[V3RSB]       step2 splat(bbox): 225 ms  avg=167.4 us/eval
[V3RSB]       step3 iou       : 24 ms  avg=17.8 us/eval
[V3RSB]     other (M_srt+pen) : 0 ms  avg=0.3 us/eval
[V3RSB] Run 9  best_inner=0.0629501  best_full=0.0543242  stop=MaxGenerations  [-]
[V3RSB/sil] Run 9  iou_evals=1350  avg_iou2d=0.6256  avg_sil_loss=0.3744  lambda=4.0000
[V3RSB] Run 10/10  Global  sigma0=0.6873  cma_seed=20262569
[V3R] build_eval_context  src=5595  subset=2829  tgt=1430  session_init_matched=940471  post_jitter_matched=0  matched_min=10  post_jitter_rmse=0  max_dist=0.227649m  scene_diag=1.6755
[V3RSB] jitter retry 1/3: sigma_factor=0.5  post_jitter_matched=0 < 141  (Q=Q:AR+AL)
[V3R] build_eval_context  src=5595  subset=2829  tgt=1430  session_init_matched=940471  post_jitter_matched=1090  matched_min=327  post_jitter_rmse=0.162735  max_dist=0.227649m  scene_diag=1.6755
[V3RSB] Run 10  Global  sigma0=0.687308  cma_seed=20262569
[V3RSB] Deterministic seed: 20262569
[V3RSB] Gen    0  best=0.16274  sigma=0.6309
[V3RSB] Gen  100  best=0.07534  sigma=0.0168
[V3RSB] === Run 10 Time Breakdown (total 1426 ms, 1350 evals) ===
[V3RSB]   evaluate_one (sum) : 1404 ms (98%)
[V3RSB]   rebuild_corr (sum) : 20 ms (1%)  [14 calls]
[V3RSB]   cmaes/log/other    : 2 ms (0%)
[V3RSB]   build_eval_ctx     : 3 ms   compute_full_rmse  : 159 ms
[V3RSB]   --- per-eval breakdown (1350 evals, 1350 with IoU) ---
[V3RSB]     rmse_w (V3R-W)    : 16 ms (1%)  avg=12.5 us/eval
[V3RSB]     sil_total (raster): 1385 ms (98%)  avg=1026.4 us/eval
[V3RSB]       step1 proj      : 1120 ms  avg=829.7 us/eval
[V3RSB]       step2 splat(bbox): 240 ms  avg=178.2 us/eval
[V3RSB]       step3 iou       : 24 ms  avg=18.1 us/eval
[V3RSB]     other (M_srt+pen) : 0 ms  avg=0.3 us/eval
[V3RSB] Run 10  best_inner=0.0718519  best_full=0.0636701  stop=MaxGenerations  [-]
[V3RSB/sil] Run 10  iou_evals=1350  avg_iou2d=0.5820  avg_sil_loss=0.4180  lambda=4.0000
[V3RSB session/Time] runs loop wall-clock : 7703.2 ms  (sum of per-run outer = 7703.2 ms)
[V3RSB] === BIPOP-CMA-ES V3R DONE ===  best_run=none  RMSE: 0.0524606 -> 0.0524606  delta=0  [NO CHANGE]  total_gens=1500  quadrant=Q:AR+AL
[V3RSB session/Time] DRIVER TOTAL : 7731.6 ms
[Alt+G] No improvement; pose unchanged.  driver=7731 ms
[Alt+G] F8 overlay updated. Press F8 to view.
  [Cached] returning injected target cloud (940471 points, boundaryDist: YES)
[Boundary3D] accepted=42184 rejected_by_instrument=22559 interior=875728  (instPxTh=8e+01, instMask=YES)
[Metrics] matched=940471  RMSE=0.05  avg=0.04  max=0.2
[Hausdorff2D] IoU=0.7  H2D=240.0px  (cost=6.87ms, step=8)
  [Cached] returning injected target cloud (940471 points, boundaryDist: YES)
[Boundary3D] accepted=42184 rejected_by_instrument=22559 interior=875728  (instPxTh=80, instMask=YES)
[Metrics] matched=940471  RMSE=0.0524606  avg=0.0448445  max=0.188657
[Hausdorff2D] IoU=0.660188  H2D=240.0px  (cost=6.74ms, step=8)

[Shift+Alt+G] ===== COMPARISON (same seed / same pose) =====
[Shift+Alt+G]              RMSE_before  RMSE_after  delta     ms
[Shift+Alt+G]   V3RS       0.05246   0.05246   0.00000   6622
[Shift+Alt+G]   V3RSB(lv0)  0.05246   0.05246   0.00000   7927
[Shift+Alt+G]   speedup(V3RS/V3RSB) = 0.84x
[Shift+Alt+G] V3RS result applied to pose. F8 = V3RSB overlay.
11:39:01: /home/meidaikasai/Documents/MyGithubProject/AAA_LiverSurgeryNaviForOBJ/build/Desktop-Release/bin/LiverSurgeryNaviForOBJ は終了コード 0 で終了しました

11:47:13: /home/meidaikasai/Documents/MyGithubProject/AAA_LiverSurgeryNaviForOBJ/build/Desktop-Release/bin/LiverSurgeryNaviForOBJ を起動中...

========================================
Current working directory: "/home/meidaikasai/Documents/MyGithubProject/AAA_LiverSurgeryNaviForOBJ/build/Desktop-Release/bin"
========================================
[Path] MODEL_PATH (auto): model/
[Path] SHADERS_PATH (auto): shaders/
[Path] REG_MODEL_PATH (auto): ../../../registration_model/
[Path] INPUT_IMAGE_PATH (auto): ../../../input_image/
[Path] DEPTH_OUTPUT_PATH (auto): ../../../depth_output/
[Path] ONNX_MODELS_PATH (auto): ../../../medsam2_da3_lite/onnx_models/
[Path] DEPTH_EXE_PATH (auto): ./medsam2_da3_lite
========================================
Final paths:
  MODEL_PATH:       model/
  SHADERS_PATH:     shaders/
  REG_MODEL_PATH:   ../../../registration_model/
  INPUT_IMAGE_PATH: ../../../input_image/
  DEPTH_OUTPUT_PATH:../../../depth_output/
  ONNX_MODELS_PATH: ../../../medsam2_da3_lite/onnx_models/
  DEPTH_EXE_PATH:   ./medsam2_da3_lite
========================================

[ImGui] Font loaded: /usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf (18px)
=== Full Sphere Camera Info ===
Using Intrinsics: fx=918.234, fy=918.112, cx=640.152, cy=366.447
Equivalent FOV: 42.8212 degrees
Position: (0, 0, 0)
Target: (0, 0, 0)
Target Mode: 0
Radius: 11.35
Sensitivities: Mouse=0.005, Light=0.01, Zoom=-1, Scale=1.1
[ARBackground] GL initialized (program=9, VAO=1)
[ARBackground] loaded ../../../depth_output/original.jpg  (1920x1080, 3 channel)
[MaskPicker] GL initialized (program=12, VAO=2)
[OriginalDiag] liver=296.282 mm, tumor=39.0581 mm  (CT-mm reference for Shift+M scale restoration)
[main] Multiple existing OBJs found, picking most recent:
[main]   * ../../../depth_output/pc_metric_pinhole_masked_custom.obj
[main]     ../../../depth_output/pc_metric_pinhole_masked_k4a.obj
[main] Using existing OBJ: ../../../depth_output/pc_metric_pinhole_masked_custom.obj
[main] Intrinsics source aligned to OBJ tag: custom (g_intrinsicsSource=2)
[TargetCache] cleared
[BBox OBJ raw (meters)] X[-0.676566, 0.760797] Y[-0.438999, 0.409655] Z[0.952772, 1.10934]
[CameraIntrinsics] loaded custom  fx=1279.81 fy=1279.81 cx=955.173 cy=499.166  1920x1080  (../../../depth_output/intrinsics_custom.txt)
[CameraIntrinsics] distortion  k1=-0.270786 k2=0.167911 k3=-0.104785 k4=0 p1=-0.00163236 p2=0.00207261
[EdgeStats/OBJ raw] 5696679 edges:  min=0.00065  p5=0.00077  p50=0.00086  p95=0.00126  p99=0.00139  max=0.00832
[MeshCleanup] component sizes (top 8 of 11): 954133 192 108 99 96 96 96 96  threshold=95414 (=0.10000 x largest 954133)
[MeshCleanup] triangles 1898893 -> 1897163  (99.90889% kept),  vertices 955196 -> 954133
[MeshCleanup]   degenerate    : 0  (area<0.00000)
[MeshCleanup]   silhouette    : 0  (|cos(n,view)| < 0.02000)
[MeshCleanup]   isolated comp : 1730  (11 components found)
[BBox OBJ after cleanup] X[-0.67657, 0.76080] Y[-0.43696, 0.40966] Z[0.95277, 1.10934]
[OrbitCam] intrinsics -> custom (fx=1279.81006 fy=1279.81006 cx=955.17297 cy=499.16599 res=1920x1080)
[MaskCleanup] hole fill: 21732 interior pixels (specular gaps) filled
[MaskCleanup] components=3  kept=962133px  dropped=256px
[MaskCleanup] occupancy filter (R=25px, th=0.85000): dropped 0 interior pseudo-boundary pixels
[Boundary] Map computed: 1920x1080 boundary_pixels=6835  (after cleanup)
[InstrumentMask] Map computed: 1920x1080  instrument_pixels=283097  valid=YES
[Normals] computed for 954133 vertices
[extractTargetFromOBJ] kept=940471  z_filtered=0  out_of_frame=0  out_of_mask=13662  (total=954133, normals=YES, instDist=YES)
[TargetCache] injected target cloud (940471 points)
[mirrorMeshAndCloudX] flipped X axis (mesh verts=954133, tris=1897163, cloud points=940471)
[prealignSourceToTarget] s=0.00566  (srcDiag=296.28168 -> tgtDiag=1.67550)  srcCenter=(-46.24188,-64.16932,51.97478) tgtCenter=(-0.04212,-0.01365,1.03105)
[SceneDiag] 1.67550  (target AABB diagonal, used for parameter normalization)
[L-diagnostic] median NN distance of target cloud:
    points         : 940471
    median (= L)   : 0.00081
    mean           : 0.00081
    p05            : 0.00076
    p95            : 0.00086
    min            : 0.00065
    max            : 0.00170
[L-info] median NN distance and active params:
    L (median NN)          : 0.00081
    sceneDiag / L          : 2074.42114
    voxel (active)         : 0.06829    (= 84.55522 L)
    voxel (literature 5L)  : 0.00404    [not used: too fine for dense depth-anything clouds]
[applySceneScaleToCamera] r=0.22765  gRadius=2.58382  pan=0.00228  zoom=-0.22765  clamp=[0.45530,18.21192]
[BBox OBJ final  (camera-space)] X[-0.76080, 0.67657] Y[-0.43696, 0.40966] Z[0.95277, 1.10934]
[BBox liverMesh3D (moved)] X[-0.43421, 0.66186] Y[-0.22486, 0.72220] Z[0.68166, 1.52367]
[Board] UV from intrinsics (20736 verts, V corrected for flipY+upload-flip)
[Board] Loaded with texture: ../../../depth_output/texture.png
[BBox boardMesh3D] X[-0.79817, 0.82401] Y[-0.49632, 0.42665] Z[0.88876, 1.11275]
[Window] Resized to 1824x972 (calib 1920x1080 + sidebar 400)
[Window] Note: clamped to display work area; AR preview is shown at reduced size, but ARSave outputs at native 1920x1080.
[OBJ Setup] target cloud ready: 940471 points; mode=Registration
[MeshBackup] Initial pose snapshot saved (6 organs)
[TargetAABB] full : c_aabb=(-0.04212,-0.01365,1.03105) mean=(-0.03937,0.04595,1.03574) diag=1.67550  n=954133
[TargetAABB] +X   : c_aabb=(0.31723,-0.01365,1.04987) mean=(0.28374,0.02745,1.05534) diag=1.11687  n=479572  (= Position 'Left' in radiology)
[TargetAABB] -X   : c_aabb=(-0.40146,0.08303,1.00877) mean=(-0.36589,0.06462,1.01561) diag=0.94181  n=474561  (= Position 'Right' in radiology)
[SourceAABB] full : c_aabb=(0.11382,0.24867,1.10266) mean=(0.03369,0.19733,1.15184) diag=1.67550  n=9992
[SourceAABB] +X   : c_aabb=(0.38791,0.17545,1.17431) mean=(0.31349,0.10453,1.24031) diag=1.11555  n=3621  (= source half for Position 'Left')
[SourceAABB] -X   : c_aabb=(-0.16023,0.25186,1.10266) mean=(-0.12534,0.25007,1.10155) diag=1.37627  n=6371  (= source half for Position 'Right')
[InitRot] Quadrant mask changed: Q:AR+AL (0x3)  -- press 'Apply Init Pose' to apply, or 'Ctrl+G' to run V3-R
[InitRot] Apply Init Pose button pressed: preset=Base  mask=Q:AR+AL (0x3)
[Session] New session #2
[Session] New session: Base @ Q:AR+AL  (mask=0x3)
[InitRot] LiverCC (Shift+H) not yet computed, auto-running...
[CC] LiverRegion (Shift+R) not yet computed, auto-running...
[Region] calling labelVertices with target_rim_mm=8.00000  origDiag_mm=296.28168
[Region] V=9992  diag=1.67550  PCA eigvals=(0.03080, 0.04686, 0.09620)
         view_axis=(-0.59951, -0.45020, 0.66174)
[Region] visible from +d: 4609  from -d: 3870  (BVH build 6.43782 ms, raycast x2 20.20446 ms)
[Region] nei concavity-floor: 1637 vertices, centroid_proj_along_d = +0.08525
[Region] swapped: anterior = -d side  (nei centroid on +d side => +d is posterior)
[Region] scale: bboxDiag=1.67550  origCTDiag(mm)=296.28168  -> units/mm=0.00566
[Region] mean_edge=0.02026 (mesh-units)  target_in_mesh=0.04524  n_rings=2  (target 8.00000 mm)
[Region] rim CCs: 13  largest=765
[Region] rim cleanup: kept largest CC (765 v); merged 100 into anterior, 0 back to posterior
[Region] core=3970  rim=765  post=5257  (rim/core=19.26952%)
[Region] timing: BVH=6.43782 raycast=20.20446 edges=11.19073 bfs=0.27434 cc=0.25731 TOTAL=39.40265 ms
[Region] viz subsample: ant=1500 rim=765 post=1200
[CC] LiverLR (Y) not yet computed, auto-running...
[LR] calling labelVertices with right_pure_fraction=0.60000  right_full_fraction=0.70000  flip_manual=false
[LR] V=9992  F=20000
[LR] diag=1.67550
[LR] d_AP (smallest) = [-0.59951  -0.45020  0.66174]
[LR] mid-extent = 0.80535, large-extent = 1.26521  ->  d_LR = largest-eigvec
[LR] d_LR (before sign) = [0.76148  -0.57541  0.29840]
[LR] BVH build: 9.44541 ms
[LR] sign signals along d_LR (positive = +d_LR is right):
[LR]   ECLIPSE (raycast):
[LR]     +d_LR cam: 3574 visible verts, 0.89977 visible area
[LR]     -d_LR cam: 2449 visible verts, 0.71809 visible area
[LR]     diff vert = +1125 (+37.35680%)  diff area = +0.18169 (+22.45996%)
[LR]     PRIMARY (area): -d_LR is RIGHT  |  diag (verts): -d_LR is RIGHT  |  decisive=True
[LR]   area-centroid lean (informational): -0.02335  (-d_LR is right by lean)
[LR]   -> primary = eclipse asymmetry (raycast, area-weighted)
[LR]   -> +d_LR is LEFT
[LR] eclipse decision: 28.00596 ms
[LR] right_pure_fraction = 0.60000  right_full_fraction = 0.70000
[LR] thresholds along d_LR : p_thr_pure = -0.35028  p_thr_full = -0.42580  (boundary band = 0.07552)
[LR] pure-R : boundary : pure-L  by vertex mass = 60.00285 : 10.00592 : 29.99124
[LR] pure-R : boundary : pure-L  by extracted face area = 61.88920 : 7.98783 : 30.12297
[LR] vertex counts: PURE_R=5994  BOUNDARY=1103  PURE_L=2895  (total=9992)
[LR] boundary faces dropped: 0.13901 (4.90522% of total area)
[LR] labelVertices done in 39.63404 ms
[LR] viz subsample: R=1500 boundary=600 L=1200
[CC] calling labelVertices  flip_manual=false
[CC] V=9992  F=20000  flip_manual=false
[CC] diag=1.67550
[CC] LR axis idx=2  -> CC axis idx=1  d_cc_raw=[0.24643  0.68280  0.68779]
[CC] mean_plus=0.15010  mean_minus=0.16961  area_plus=0.09601  area_minus=0.09398  n_rim+=366  n_rim-=399
[CC] confidence=6.10086%  [OK]  flipped_manual=false  d_cc=[0.24643  0.68280  0.68779]  (+d_cc -> CRANIAL)
[CC] cranial=4788  caudal=5204  (cranial/total=47.91833%)
[CC] timing: face_rough=4.77190 vert_rough+areas=0.32005 decide_sign=0.02241 label=0.11280 TOTAL=5.52548 ms
[CC] viz subsample: cranial=1500 caudal=1500
[InitRot/PreApply] BEFORE transform:
    centroid (rotation pivot) = (0.03369,0.19733,1.15184)
    src_subset.center (orig AABB midpoint) = (0.11382,0.24867,1.10266)  diag=1.67550
    rotated AABB (real vertices, n=4735): min=(-0.67452,0.01461,0.71345)  max=(0.59070,0.61291,1.51879)  center=(-0.04191,0.31376,1.11612)
    target_full.center (固定参照点 = AABB midpoint) = (-0.04212,-0.01365,1.03105)  diag=1.67550
    computed: scale=1.00000  t_pos=(-0.00021,-0.32741,-0.08507)
[InitRot] applied: preset=Base  mask=Q:AR+AL (0x3)  scale=1.00000  t_pos=(-0.00021,-0.32741,-0.08507)  d_lr=[-0.76148,0.57541,-0.29840]  d_cc=[0.24643,0.68280,0.68779]
[InitRot/PostApply] VERIFICATION (AABB center):
    post-transform subset AABB center = (-0.04212,-0.01365,1.03105)  (n_used=4735)
    target_full.center                (-0.04212,-0.01365,1.03105)
    error vector = (0.00000,0.00000,0.00000)  |err|=0.00000  [OK: subset AABB center == target center]
[InitRot] Preset selected: Up
[Session] New session #3
[Session] New session: Up @ Q:AR+AL  (mask=0x3)
[InitRot/PreApply] BEFORE transform:
    centroid (rotation pivot) = (0.03369,0.19733,1.15184)
    src_subset.center (orig AABB midpoint) = (0.11382,0.24867,1.10266)  diag=1.67550
    rotated AABB (real vertices, n=4735): min=(-0.67452,-0.06935,0.74444)  max=(0.59070,0.61999,1.51041)  center=(-0.04191,0.27532,1.12742)
    target_full.center (固定参照点 = AABB midpoint) = (-0.04212,-0.01365,1.03105)  diag=1.67550
    computed: scale=1.00000  t_pos=(-0.00021,-0.28897,-0.09637)
[InitRot] applied: preset=Up  mask=Q:AR+AL (0x3)  scale=1.00000  t_pos=(-0.00021,-0.28897,-0.09637)  d_lr=[-0.76148,0.57541,-0.29840]  d_cc=[0.24643,0.68280,0.68779]
[InitRot/PostApply] VERIFICATION (AABB center):
    post-transform subset AABB center = (-0.04212,-0.01365,1.03105)  (n_used=4735)
    target_full.center                (-0.04212,-0.01365,1.03105)
    error vector = (0.00000,-0.00000,0.00000)  |err|=0.00000  [OK: subset AABB center == target center]
[PoseLibrary] Undo snapshot saved

=== QuadCyclic-RANSAC Registration (Shift+Ctrl+P) ===
[FGR] Deterministic seed set: 20260420
[Seed] FGR=20260420  (trial=20260420, callIdx=0)
[Shift+Ctrl+P] quadrant_mask = Q:AR+AL  (0x3)  subset_size=4735/9992
[Shift+Ctrl+P] liverMesh3D normals missing -- computing from faces...
[Normals] computed for 9992 vertices
  Camera position: (0.00000, 0.00000, 0.00000)
  View direction: (0.00000, 0.00000, 1.00000)
  Total vertices: 9992
  Visible: 2112
  Occluded: 7880
  Backface: 0
[Shift+Ctrl+P/Source] AR-visible=2112  silh∩quad=341  quad_only=1306  silh_only=236  final=341  cosThresh=0.40000
[Shift+Ctrl+P/Target] full=940471  boundary_kept=42184  (boundaryPx<12.00000, instPx>=80.00000)
[Sectors] N=24  src_occupied=24/24  tgt_occupied=18/24
[QCR] valid_subsets=728  (kSectors=24, K=3, MIN_SPREAD=4 sec)  trials_total=34944  trials_planned=34944  stride=1  (cap=100000)
[QCR] Stage 1: tried=34944  rot_filtered=6780  valid=2218  topK=20  lambda_s1=0.300  max_rot=90.0deg  (10.8 ms)
[QCR] Stage 2: 20 candidates eval  rot_rejected=0  lambda_disp=0.500  max_axis_rot=90.0deg  (11.1 ms)
[QCR] Best: K=3  subset={0,12,20}  shift=1  dir=forward(CW)  score_s1=0.2396  chamfer=0.1957  disp=0.1838  total=0.2876  rot=[-5.9,-3.9,-15.3]deg
[QCR] Refine: SKIPPED  (inliers=18, thr=0.3594, median_resid=0.1578) -> using initial 3-pt T  [reason: total worsened or insane refinement]
[QCR] top20 (rank by score_total):
    #1 sub={0,12,20} sh=1/F  s1=0.2396 cham=0.1957 disp=0.1838 tot=0.2876  rot=[-5.9,-3.9,-15.3]
    #2 sub={0,9,13} sh=1/F  s1=0.2214 cham=0.2064 disp=0.1670 tot=0.2899  rot=[7.2,-3.9,-10.9]
    #3 sub={1,9,13} sh=1/F  s1=0.2213 cham=0.2040 disp=0.1720 tot=0.2901  rot=[4.8,-5.3,-15.2]
    #4 sub={9,13,23} sh=1/F  s1=0.2402 cham=0.2231 disp=0.1444 tot=0.2953  rot=[13.5,2.8,-10.9]
    #5 sub={1,9,14} sh=1/F  s1=0.2225 cham=0.2060 disp=0.1880 tot=0.3000  rot=[6.1,-5.2,-15.9]
    #6 sub={0,10,20} sh=1/F  s1=0.2310 cham=0.2171 disp=0.1663 tot=0.3002  rot=[1.3,-8.5,-13.4]
    #7 sub={1,12,21} sh=1/F  s1=0.2408 cham=0.2058 disp=0.1890 tot=0.3003  rot=[3.1,-6.4,-19.5]
    #8 sub={0,9,14} sh=1/F  s1=0.2239 cham=0.2102 disp=0.1826 tot=0.3016  rot=[8.8,-3.6,-11.4]
    #9 sub={1,10,20} sh=1/F  s1=0.2367 cham=0.2167 disp=0.1700 tot=0.3017  rot=[-0.0,-9.1,-16.3]
    #10 sub={0,9,20} sh=1/F  s1=0.2437 cham=0.2087 disp=0.1866 tot=0.3020  rot=[-3.7,-5.7,-12.2]
[Shift+Ctrl+P] Applied RANSAC prealign T (scale=1.31663)  -> proceeding to ICP refinement
[TargetCache] injected target cloud (42184 points)

================================================
| Custom Registration (Single Mesh Source)    |
================================================
  Source vertices:      341
  Max iterations:       1
  Convergence threshold:0.00113824
  Min fitness required: 0.35
  Scale estimation:     ON

  Evaluating initial state...
  [Cached] returning injected target cloud (42184 points, boundaryDist: YES)
  [Custom] Original points: 341
  [Custom] VoxelDownSample: 341 -> 108 (voxel=0.0682947)
  [Custom] Downsampled to: 108 points
  [Custom] estimateNormals: 108 normals computed (radius=0.136589, k=30)
  [Custom] Normals estimated
  [Custom] Original points: 42184
  [Custom] VoxelDownSample: 42184 -> 67 (voxel=0.0682947)
  [Custom] Downsampled to: 67 points
  Initial fitness: 0.0277778
  [Cached] returning injected target cloud (42184 points, boundaryDist: YES)

[TargetCache] Pre-computing target voxel + FPFH (std + retry)...
  std voxel:   0.0682947
  retry voxel: 0.0819536
  [Custom] Original points: 42184
  [Custom] VoxelDownSample: 42184 -> 67 (voxel=0.0682947)
  [Custom] Downsampled to: 67 points
  [Custom] computeFPFH: 67 points, dim=33 (radius=0.341473, max_nn=100)
  [Custom] Original points: 42184
  [Custom] VoxelDownSample: 42184 -> 57 (voxel=0.0819536)
  [Custom] Downsampled to: 57 points
  [Custom] computeFPFH: 57 points, dim=33 (radius=0.409768, max_nn=100)
  [TargetCache] built in 1 ms (std=67 pts, retry=57 pts)

----------------------------------------
  ITERATION 1/1
----------------------------------------
Step 1: Building point clouds...
  Source points: 341
  Target points: 42184
  [Time] Step1: 0 ms

Step 2: Preprocessing (voxel size: 0.0682947)...
  [TargetCache] HIT (std voxel=0.0682947)
  [Custom] Original pointsm voxel cell centroids back to the nearest original vertex
 *      ("V-C: NN reverse map" approach -- voxel_downsample_v3 itself is
 *      NOT touched).
 *   2. tgt_to_eval[i] holds a voxel-space global index, identical
 *      semantics to V3, so evaluate_one_v3 (V3 function) is reused
 *      verbatim with no fork.
 *   3. compute_full_rmse_local is reused verbatim by default
 *      (full-vertex RMSE, plan E1: A/B comparison stays one-to-one).
 *      A future subset-RMSE path is gated by ParamsV3R::full_rmse_use_subset
 *      (currently always falls through to the full path).
 *
 * V3 byte-identical contract (CRITICAL):
 *   When ParamsV3R::quadrant_mask == QUAD_ALL (0x0F), subset_idx_voxel
 *   becomes (0, 1, ..., N_voxel-1), so:
 *     - voxel_downsample_v3 output identical (V3R does not touch it),
 *     - subset KDTree input == full KDTree input,
 *     - tgt_to_eval[i] == V3 value (subset_idx_voxel[nn_local] == nn_local),
 *     - evaluate_one_v3 reused unchanged,
 *     - compute_full_rmse_local reused unchanged (full-vertex path).
 *   Net result: Ctrl+G [QUAD_ALL] produces digit-for-digit identical
 *   per-Gen / per-Run / final logs as Shift+G when both are run from
 *   the same (g_trialSeed, g_callIdx) state. This is the S4 acceptance
 *   gate (HANDOVER §2.6).
 *
 * Fork map (vs V3):
 *   build_eval_context_v3      -> build_eval_context_v3r      (FORK)
 *   rebuild_correspondences_v3 -> rebuild_correspondences_v3r (FORK)
 *   run_one_bipop              -> run_one_bipop_v3r           (FORK)
 *   runBipopCmaesV3            -> runBipopCmaesV3R            (FORK)
 *   evaluate_one_v3            -> reused verbatim (subset-agnostic)
 *   compute_full_rmse_local    -> reused verbatim (default path)
 *   voxel_downsample_v3        -> reused verbatim
 *   srt_*, build_srt_matrix_v3, compute_centroid_v3, apply_srt_to_points
 *                              -> reused verbatim
 *
 * Determinism contract (must match V3 byte-for-byte at QUAD_ALL):
 *   outer_seed = g_trialSeed + 1000u + g_callIdx * 97u   (caller)
 *   cma_base   = g_trialSeed + 2000u + g_callIdx * 10u   (caller)
 *   d01(rng) consumption order per Run: identical 8-call sequence
 *   g_callIdx++ at session end (caller responsibility, same as V3)
 * ----------------------------------------------------------------------
 */

#include "CmaesRefineV3.h"            // V3 functions reused verbatim
#include "LiverLeftRightLabel.h"      // QuadrantMask, makeQuadrantSubsetIdx, etc.

#include <vector>
#include <cstdint>
#include <unordered_set>
#include <algorithm>   // std::count (rim diagnostics)

namespace CmaesRefineV3R {

// =====================================================================
// [NEW V3R/SearchMode] Reduced-DoF search modes (additive, opt-in).
// ---------------------------------------------------------------------
// SEVEN_DOF (default) is byte-identical to original V3R behaviour --
// the runtime path through cmaes_init, srt_from_population_v3r_mode,
// and mask_jitter_for_mode all reproduce the V3R-equivalent code path
// when search_mode == SEVEN_DOF.
//
// The reduced modes lock specific 7-DoF axes to identity BEFORE the
// CMA-ES inner loop sees them, by three coupled changes inside
// run_one_bipop_v3r:
//   1. CMA-ES decision-vector dimension DIM is set to 6 or 4
//      (cmaes_init(DIM=...) sees a smaller search space).
//   2. srt_from_population_v3r_mode() decodes only the active
//      components from each sample and fills the inactive components
//      with identity (scale=1, tz=0, rz=0) so downstream code
//      (build_srt_matrix_v3, compute_full_rmse_local, etc.) still
//      sees a full 7-field SRTParamsV3 and needs no changes.
//   3. mask_jitter_for_mode() zeroes out the inactive jitter
//      components AFTER the 8-draw d01() sequence in the session
//      driver, so the rng state evolves identically across modes
//      for a given outer_seed.
//
// Motivation (HANDOVER_UNIFIED_ALL.md §III/3.1, "scale blowup"):
//   Repeated Ctrl+G sessions exhibit a failure mode where each
//   session adopts a scale slightly >1.0, accumulating over
//   iterations until the source mesh covers more target points by
//   inflation rather than alignment. SIX_DOF_RIGID locks scale;
//   FOUR_DOF_XYRXRY additionally locks tz/rz for final-stage polish
//   under a fixed-AR-camera workflow.
//
// Recommended workflow:
//   1) 7-DoF Ctrl+G    (coarse, 1-2 sessions)
//   2) 6-DoF rigid     (after scale converged, 1-2 sessions)
//   3) 4-DoF XY+RX+RY  (final polish, no scale/Z/roll drift)
// =====================================================================
enum class SearchMode : uint8_t {
    SEVEN_DOF       = 0,   // tx,ty,tz, rx,ry,rz, scale  (V3R byte-identical)
    SIX_DOF_RIGID   = 1,   // tx,ty,tz, rx,ry,rz         (scale fixed = 1.0)
    FOUR_DOF_XYRXRY = 2,   // tx,ty,    rx,ry            (tz=rz=0, scale=1)
};

// Effective CMA-ES decision-vector dimension for each mode.
inline int dim_for_search_mode(SearchMode m) {
    switch (m) {
        case SearchMode::SEVEN_DOF:       return 7;
        case SearchMode::SIX_DOF_RIGID:   return 6;
        case SearchMode::FOUR_DOF_XYRXRY: return 4;
    }
    return 7;   // safe default if enum is corrupted
}

// Human-readable mode name for log lines.
inline const char* name_of_search_mode(SearchMode m) {
    switch (m) {
        case SearchMode::SEVEN_DOF:       return "7-DoF";
        case SearchMode::SIX_DOF_RIGID:   return "6-DoF-rigid";
        case SearchMode::FOUR_DOF_XYRXRY: return "4-DoF-xyRxRy";
    }
    return "?";
}

// Mask Run-startup jitter to the active DoFs of the current mode.
// Applied AFTER the 8-draw d01 sequence in runBipopCmaesV3R so that
// the rng state matches V3R(7-DoF) byte-for-byte for the same
// outer_seed; only the jitter VALUES differ across modes.
// SEVEN_DOF: pass-through. SIX_DOF_RIGID: clamp scale=1. FOUR_DOF:
// additionally clamp tz=rz_deg=0.
inline CmaesRefine::SRTParamsV3 mask_jitter_for_mode(
    CmaesRefine::SRTParamsV3 j, SearchMode m)
{
    switch (m) {
        case SearchMode::SEVEN_DOF:
            return j;
        case SearchMode::SIX_DOF_RIGID:
            j.scale = 1.0f;
            return j;
        case SearchMode::FOUR_DOF_XYRXRY:
            j.tz     = 0.0f;
            j.rz_deg = 0.0f;
            j.scale  = 1.0f;
            return j;
    }
    return j;
}

// =====================================================================
// ParamsV3R -- V3 params + 4-quadrant region info.
// Caller fills the *input* fields (quadrant_mask, region_labels,
// lr_labels, full_rmse_use_subset); runBipopCmaesV3R fills the
// *session-derived* fields (voxel_to_orig, subset_idx_voxel) before
// dispatching to run_one_bipop_v3r. This keeps run_one_bipop_v3r free
// of any extra computation per Run -- subset is computed exactly once
// per session.
// =====================================================================
struct ParamsV3R : public CmaesRefine::ParamsV3 {
    // ----- Caller-set inputs ----------------------------------------
    // 4-quadrant bitmask (LiverLeftRightLabel::QUAD_AR / QUAD_AL /
    // QUAD_PR / QUAD_PL, OR'd together; QUAD_ALL = 0x0F).
    uint8_t quadrant_mask = LiverLeftRightLabel::QUAD_ALL;

    // Per-vertex region (anterior_core / rim / posterior) and L-R
    // (pure_right / boundary / pure_left) labels in ORIGINAL vertex
    // index space (size == N_full, i.e. start_liver_verts.size()).
    // Caller copies these from g_liverRegion.labels and g_liverLR.labels.
    std::vector<uint8_t> region_labels;
    std::vector<uint8_t> lr_labels;

    // When true AND quadrant_mask != QUAD_ALL, the per-Run screening
    // RMSE (best_rmse_full) is computed over the subset only (future
    // work). Default false = full-vertex RMSE (plan E1, A/B comparison
    // friendly). At S4 the "true" branch is not yet implemented; the
    // flag is reserved here for forward-compatibility.
    bool full_rmse_use_subset = false;

    // ----- Session-derived (filled by runBipopCmaesV3R, not caller) -
    // NN reverse map: voxel_to_orig[i] = index into start_liver_verts
    // of the original vertex nearest to session_voxel_liver[i].
    // Size == session_voxel_liver.size() (e.g. 4091 for LIVER01).
    std::vector<int> voxel_to_orig;

    // Subset of voxel-after-downsample indices selected by quadrant_mask.
    // For QUAD_ALL, this is (0, 1, ..., N_voxel-1).
    // For QUAD_AR alone, this is the voxel indices whose voxel_to_orig
    // lookup hits an original vertex with both ant-membership and
    // R-membership (rim/boundary overlap policy = OR, see HANDOVER §2.2).
    std::vector<int> subset_idx_voxel;

    // =================================================================
    // Rim-weighted V3R extension (additive; defaults preserve V3R/V3
    // byte-identical behavior).
    // -----------------------------------------------------------------
    // When ALL of the following hold, the inner CMA-ES loop falls back
    // to the original evaluate_one_v3 verbatim (byte-identical):
    //   - use_arvis_filter   == false
    //   - beta_rim_weight    == 0.0f
    //   - is_rim_src_orig    is empty
    //   - tgt_boundary_dist_full is empty
    // i.e. when the caller provides no rim/visibility inputs.
    // =================================================================

    // Caller-set: enable AR-camera visibility filter on the source
    // subset. If true, runBipopCmaesV3R will AND subset_idx_voxel with
    // arvis_voxel (see below). Default false preserves S4 behavior.
    bool use_arvis_filter = false;

    // Caller-set (in original-vertex index space; size == N_full).
    // arvis_orig[i] = 1 if original vertex i is visible from the AR
    // camera (cam_pos = (0,0,0), look-at = (0,0,1)) under the BVH of
    // liverMesh3D. Filled by the wrapper using
    // LiverRegionLabel::raycastVisibilityBVH. Empty when filter is off.
    std::vector<uint8_t> arvis_orig;

    // Caller-set (in original-vertex index space; size == N_full).
    // is_rim_orig[i] = 1 if g_liverRegion.labels[i] == RIM, i.e. the
    // mesh-intrinsic anatomical rim band (鎌状間膜あたり). Used by the
    // weighted evaluator. Empty when rim weighting is off.
    std::vector<uint8_t> is_rim_orig;

    // Caller-set (parallel to tgt_points; size == tgt_points.size()).
    // boundaryDist values from the cached PointCloud. Used internally
    // to derive is_rim_tgt_voxel after voxel downsampling. Empty when
    // rim weighting is off.
    std::vector<float> tgt_boundary_dist_full;

    // Caller-set: pixel threshold for target rim membership.
    //   is_rim_tgt[i] = (tgt_boundary_dist_voxel[i] < rim_tgt_threshold_px)
    // 12.0 matches Shift+P (kBoundaryPxTh).
    float rim_tgt_threshold_px = 12.0f;

    // Caller-set: rim-rim pair weight multiplier.
    //   w_i = 1 + beta * is_rim_src[j] * is_rim_tgt[i]
    // beta=0 -> uniform (byte-identical to evaluate_one_v3).
    // beta=1 -> rim-rim pairs counted twice; beta=3 -> 4x; etc.
    float beta_rim_weight = 0.0f;

    // ----- Session-derived (filled by runBipopCmaesV3R) --------------
    // arvis_voxel[k] = arvis_orig[voxel_to_orig[k]] (or 1 if filter off).
    // Size == session_voxel_liver.size() when filter on; empty otherwise.
    std::vector<uint8_t> arvis_voxel;

    // is_rim_src_voxel[k] = is_rim_orig[voxel_to_orig[k]] (or 0 if
    // is_rim_orig empty). Size == session_voxel_liver.size() when rim
    // weighting on; empty otherwise.
    std::vector<uint8_t> is_rim_src_voxel;

    // is_rim_tgt_voxel[i] = (boundary_dist_at_voxel_tgt[i] <
    // rim_tgt_threshold_px). Size == session_voxel_tgt.size() when rim
    // weighting on; empty otherwise.
    std::vector<uint8_t> is_rim_tgt_voxel;

    // =================================================================
    // Caudal-only filter (R-feat-2; anatomical CC axis, mesh-intrinsic).
    // -----------------------------------------------------------------
    // Orthogonal to use_arvis_filter:
    //   - AR-vis filter = view-based (raycast from AR camera).
    //   - Caudal-only   = anatomy-based (LiverCranioCaudalLabel::CAUDAL).
    // Both filter the source subset; combine_mode controls how they
    // interact when both are simultaneously requested.
    //
    // Defaults preserve V3R / V3 byte-identical behaviour:
    //   use_caudal_only == false  AND  is_caudal_orig.empty()
    //                            -> caudal_voxel becomes empty
    //                            -> filter_by_quadrant_with_arvis_caudal
    //                               degenerates to filter_by_quadrant
    //                               (or filter_by_quadrant_with_arvis
    //                               when arvis is also requested).
    // =================================================================

    // Caller-set: enable the caudal-only filter on the source subset.
    bool use_caudal_only = false;

    // Caller-set: how to combine arvis and caudal filters when BOTH
    // are simultaneously enabled. Ignored otherwise.
    //   0 = AND : vertex must be AR-visible AND caudal.
    //   1 = OR  : vertex passes if AR-visible OR caudal.
    // Default 0 (AND) matches the rim-matching intent: most restrictive.
    uint8_t arvis_caudal_combine = 0;

    // Caller-set (in original-vertex index space; size == N_full).
    // is_caudal_orig[i] = 1 iff g_liverCC.labels[i] == CAUDAL. Empty
    // when use_caudal_only is off.
    std::vector<uint8_t> is_caudal_orig;

    // ----- Session-derived (filled by runBipopCmaesV3R) --------------
    // is_caudal_voxel[k] = is_caudal_orig[voxel_to_orig[k]] (or empty
    // if is_caudal_orig is empty). Size == session_voxel_liver.size()
    // when caudal filter is on; empty otherwise.
    std::vector<uint8_t> is_caudal_voxel;

    // =================================================================
    // [NEW V3R/SearchMode] Reduced-DoF search mode (Ctrl+G UI).
    // -----------------------------------------------------------------
    // SEVEN_DOF (default) is byte-identical to pre-feature V3R: cmaes_init
    // sees DIM=7 and the same lb/ub/xstart, and srt_from_population_v3r_mode
    // forwards to the SEVEN_DOF branch which exactly reproduces the math
    // of CmaesRefine::srt_from_population_v3.
    //
    // Caller-set (RegistrationActions.h::runBipopCmaesV3R wrapper) from
    // g_ctrlgSearchMode. The wrapper also pre-scales tx/ty/tz/rx/ry/rz
    // _range and jitter_local/global_t by 0.7x (SIX_DOF_RIGID) or 0.5x
    // (FOUR_DOF_XYRXRY) so the reduced-DoF search stays appropriately
    // local; min_match_ratio is similarly exposed for tuning.
    // =================================================================
    SearchMode search_mode = SearchMode::SEVEN_DOF;
};

// =====================================================================
// [NEW V3R/SearchMode] Mode-aware decode of the CMA-ES population vector.
// ---------------------------------------------------------------------
// SEVEN_DOF: byte-identical to CmaesRefine::srt_from_population_v3.
// SIX_DOF_RIGID: pop_k[0..2]=t, pop_k[3..5]=r; scale forced to 1.
// FOUR_DOF_XYRXRY: pop_k[0..1]=tx,ty; pop_k[2..3]=rx,ry; tz=rz=0, scale=1.
//
// In all cases the returned SRTParamsV3 has every field set, so
// build_srt_matrix_v3, apply_srt_to_points, and compute_full_rmse_local
// see a normal 7-field SRT and need no changes.
//
// The 6-DoF and 4-DoF branches READ ONLY the first 6 or 4 entries of
// pop_k; the trailing entries are out of bounds (cmaes_init was called
// with the matching DIM, so cmaes does not write past those entries).
// =====================================================================
inline CmaesRefine::SRTParamsV3 srt_from_population_v3r_mode(
    const double* pop_k, const ParamsV3R& p)
{
    using CmaesRefine::SRTParamsV3;
    SRTParamsV3 s;
    switch (p.search_mode) {
    case SearchMode::SEVEN_DOF: {
        // Byte-identical to srt_from_population_v3 in CmaesRefineV3.h.
        s.tx     = (float)(pop_k[0] * p.tx_range);
        s.ty     = (float)(pop_k[1] * p.ty_range);
        s.tz     = (float)(pop_k[2] * p.tz_range);
        s.rx_deg = (float)(pop_k[3] * p.rx_range);
        s.ry_deg = (float)(pop_k[4] * p.ry_range);
        s.rz_deg = (float)(pop_k[5] * p.rz_range);
        s.scale  = p.scale_lo
                 + (float)((pop_k[6] + 1.0) * 0.5)
                       * (p.scale_hi - p.scale_lo);
        return s;
    }
    case SearchMode::SIX_DOF_RIGID: {
        // Rigid 6-DoF: scale frozen at identity.
        s.tx     = (float)(pop_k[0] * p.tx_range);
        s.ty     = (float)(pop_k[1] * p.ty_range);
        s.tz     = (float)(pop_k[2] * p.tz_range);
        s.rx_deg = (float)(pop_k[3] * p.rx_range);
        s.ry_deg = (float)(pop_k[4] * p.ry_range);
        s.rz_deg = (float)(pop_k[5] * p.rz_range);
        s.scale  = 1.0f;
        return s;
    }
    case SearchMode::FOUR_DOF_XYRXRY: {
        // Minimal 4-DoF: tx, ty, rx, ry only. tz/rz/scale all identity.
        s.tx     = (float)(pop_k[0] * p.tx_range);
        s.ty     = (float)(pop_k[1] * p.ty_range);
        s.tz     = 0.0f;
        s.rx_deg = (float)(pop_k[2] * p.rx_range);
        s.ry_deg = (float)(pop_k[3] * p.ry_range);
        s.rz_deg = 0.0f;
        s.scale  = 1.0f;
        return s;
    }
    }
    return s;   // unreachable; conservative fallback
}

// Body intentionally identical to srt_from_population_v3r_mode; the
// separate name documents the call site (best-so-far parameters used
// at UPDATE_INTERVAL rebuild and final writeback). Mirrors V3's split
// of srt_from_population_v3 vs srt_from_xvec_v3.
inline CmaesRefine::SRTParamsV3 srt_from_xvec_v3r_mode(
    const double* x, const ParamsV3R& p)
{
    return srt_from_population_v3r_mode(x, p);
}

// =====================================================================
// build_voxel_to_orig -- V-C reverse-NN map from voxel cells to
//                        original vertices.
// ---------------------------------------------------------------------
// Called once per session (in runBipopCmaesV3R, after voxel downsample).
// Builds a KDTree on the FULL (non-downsampled) liver vertices, then
// for each voxel-cell centroid finds its nearest full-vertex index.
// The label of that full vertex (from region_labels / lr_labels) is
// adopted as the voxel cell's label.
//
// Cost: O(N_voxel * log N_full). For LIVER01 (4091 voxel, 9986 full)
//       this is ~1 ms; negligible compared to the per-Run KDTree.
// QUAD_ALL behavior: every voxel cell is included regardless of label,
//                    so the result is unused -- but we still compute
//                    it (cheap, keeps code path uniform).
// =====================================================================
inline std::vector<int> build_voxel_to_orig(
    const std::vector<glm::vec3>& full_liver,
    const std::vector<glm::vec3>& voxel_liver)
{
    std::vector<int> voxel_to_orig(voxel_liver.size(), -1);
    if (full_liver.empty() || voxel_liver.empty()) {
        return voxel_to_orig;
    }

    Reg3DCustom::NanoflannAdaptor adaptor(full_liver);
    auto tree = Reg3DCustom::buildKDTree(adaptor);

    for (size_t i = 0; i < voxel_liver.size(); i++) {
        size_t nnIdx;
        float  dist_sq;
        if (Reg3DCustom::searchKNN1(*tree, voxel_liver[i], nnIdx, dist_sq)) {
            voxel_to_orig[i] = (int)nnIdx;
        }
        // else: leave -1 (degenerate; should never happen on non-empty input)
    }
    return voxel_to_orig;
}

// =====================================================================
// filter_by_quadrant -- compute subset_idx_voxel from voxel_to_orig + masks.
// ---------------------------------------------------------------------
// For each voxel index i, look up its representative original vertex
// idx_orig = voxel_to_orig[i], then test whether the original vertex's
// (region_label, lr_label) pair satisfies the quadrant_mask.
//
// Uses LiverLeftRightLabel::makeQuadrantSubsetIdx in original-index
// space first (S1 helper), then maps the resulting set to voxel space
// via membership test on voxel_to_orig.
//
// QUAD_ALL fast path: every i is included, returns (0, 1, ..., N_voxel-1).
//                     This makes byte-identical reproduction trivial:
//                     subset_idx_voxel[nn_local] == nn_local.
// =====================================================================
inline std::vector<int> filter_by_quadrant(
    const std::vector<int>&       voxel_to_orig,
    const std::vector<uint8_t>&   region_labels,
    const std::vector<uint8_t>&   lr_labels,
    uint8_t                        quadrant_mask)
{
    std::vector<int> out;
    out.reserve(voxel_to_orig.size());

    // ----- QUAD_ALL fast path (byte-identical guarantee) --------------
    // No filtering; every voxel index is included in original order.
    // This avoids touching the label arrays at all when no quadrant
    // restriction is active, which is also the only configuration
    // where labels may legitimately be empty (e.g. label computation
    // hasn't run yet but caller still wants V3-equivalent behavior).
    if (quadrant_mask == LiverLeftRightLabel::QUAD_ALL) {
        for (int i = 0; i < (int)voxel_to_orig.size(); i++) {
            out.push_back(i);
        }
        return out;
    }

    // ----- Filtered path: original-index subset, then voxel mapping ---
    // makeQuadrantSubsetIdx gives indices in ORIGINAL vertex space.
    // Convert to a hash-set for O(1) membership test below.
    std::vector<int> subset_orig =
        LiverLeftRightLabel::makeQuadrantSubsetIdx(
            region_labels, lr_labels, quadrant_mask);
    std::unordered_set<int> orig_set(subset_orig.begin(), subset_orig.end());

    for (int i = 0; i < (int)voxel_to_orig.size(); i++) {
        const int idx_orig = voxel_to_orig[i];
        if (idx_orig >= 0 && orig_set.count(idx_orig) > 0) {
            out.push_back(i);
        }
    }
    return out;
}

// =====================================================================
// =====  Rim-weighted V3R extension (additive, opt-in) ================
// =====================================================================
// The functions below implement the AR-visibility filter on the source
// subset and the rim-rim multiplicative weight on the inner cost.
// They are invoked ONLY when the corresponding ParamsV3R flags are set
// by the caller; otherwise runBipopCmaesV3R falls through to the
// original V3R path and (under QUAD_ALL) remains byte-identical to V3.
// =====================================================================

// =====================================================================
// filter_by_quadrant_with_arvis -- variant of filter_by_quadrant that
// additionally AND-filters by per-voxel AR-visibility.
// ---------------------------------------------------------------------
// arvis_voxel: per-voxel-index 0/1 mask (size == voxel_to_orig.size()).
// If arvis_voxel is empty, this is identical to filter_by_quadrant
// (no visibility filtering applied).
//
// QUAD_ALL + non-empty arvis_voxel: subset shrinks to visible voxels
// only; the byte-identical contract w.r.t. Shift+G no longer holds and
// the caller is expected to acknowledge this (e.g. via UI checkbox).
// =====================================================================
inline std::vector<int> filter_by_quadrant_with_arvis(
    const std::vector<int>&       voxel_to_orig,
    const std::vector<uint8_t>&   region_labels,
    const std::vector<uint8_t>&   lr_labels,
    uint8_t                        quadrant_mask,
    const std::vector<uint8_t>&   arvis_voxel)   // empty == no filter
{
    // First pass: pure quadrant filter (delegates to existing helper).
    std::vector<int> base = filter_by_quadrant(
        voxel_to_orig, region_labels, lr_labels, quadrant_mask);

    // No visibility filter requested: return as-is (byte-identical).
    if (arvis_voxel.empty()) return base;

    // Second pass: AND with visibility.
    std::vector<int> out;
    out.reserve(base.size());
    for (int vi : base) {
        if (vi >= 0 && (size_t)vi < arvis_voxel.size() && arvis_voxel[vi]) {
            out.push_back(vi);
        }
    }
    return out;
}

// =====================================================================
// filter_by_quadrant_with_arvis_caudal -- generalisation that combines
// AR-visibility and caudal-only filters with selectable AND / OR mode.
// ---------------------------------------------------------------------
// Inputs (each "empty" means: this filter is NOT requested):
//   arvis_voxel  : per-voxel 0/1, view-based front-facing mask.
//   caudal_voxel : per-voxel 0/1, anatomical CAUDAL-side mask.
//   combine_mode : 0 = AND, 1 = OR.  Effective ONLY when BOTH
//                  arvis_voxel and caudal_voxel are non-empty; when only
//                  one is non-empty, that one alone applies (combine
//                  mode is irrelevant). When both are empty, returns
//                  the pure quadrant subset unchanged (V3R behaviour).
//
// Pass / reject rule per voxel vi (inside the quadrant subset):
//   a := (arvis_voxel.empty())  OR arvis_voxel[vi]
//   c := (caudal_voxel.empty()) OR caudal_voxel[vi]
//   pass := (both empty)        -> true        (no extra filtering)
//         : (only one non-empty)-> a && c       (degenerate: shortcut)
//         : (both non-empty)    -> combine_mode==0 ? (a && c) : (a || c)
//
// QUAD_ALL byte-identical preserved when both inputs are empty
// (delegates to filter_by_quadrant).
// =====================================================================
inline std::vector<int> filter_by_quadrant_with_arvis_caudal(
    const std::vector<int>&       voxel_to_orig,
    const std::vector<uint8_t>&   region_labels,
    const std::vector<uint8_t>&   lr_labels,
    uint8_t                        quadrant_mask,
    const std::vector<uint8_t>&   arvis_voxel,    // empty == not requested
    const std::vector<uint8_t>&   caudal_voxel,   // empty == not requested
    uint8_t                        combine_mode)  // 0=AND, 1=OR
{
    std::vector<int> base = filter_by_quadrant(
        voxel_to_orig, region_labels, lr_labels, quadrant_mask);

    const bool a_on = !arvis_voxel.empty();
    const bool c_on = !caudal_voxel.empty();

    // Neither filter requested: byte-identical to filter_by_quadrant.
    if (!a_on && !c_on) return base;

    std::vector<int> out;
    out.reserve(base.size());
    for (int vi : base) {
        if (vi < 0) continue;
        // Empty array is treated as "all pass" for that filter (via
        // short-circuit on a_on / c_on).
        const bool a = (!a_on) ||
                       ((size_t)vi < arvis_voxel.size()  && arvis_voxel[vi]);
        const bool c = (!c_on) ||
                       ((size_t)vi < caudal_voxel.size() && caudal_voxel[vi]);
        bool pass;
        if (a_on && c_on) {
            // Both requested: combine_mode picks AND or OR.
            pass = (combine_mode == 0) ? (a && c) : (a || c);
        } else {
            // Exactly one requested: pass = the active flag. The
            // "inactive" flag is true via the shortcut above, so
            // a && c collapses to the active one.
            pass = a && c;
        }
        if (pass) out.push_back(vi);
    }
    return out;
}

// =====================================================================
// derive_arvis_voxel -- map per-original-vertex visibility to per-voxel.
// ---------------------------------------------------------------------
// arvis_orig: per-original-vertex 0/1 (size == start_liver_verts.size())
// voxel_to_orig: voxel index -> original vertex index (V3R session map).
//
// Returns a vector of size voxel_to_orig.size() with arvis_voxel[k] =
// arvis_orig[voxel_to_orig[k]] (or 0 if the lookup is degenerate).
// If arvis_orig is empty (caller didn't request the filter), returns
// an empty vector — downstream code treats empty as "no filter".
// =====================================================================
inline std::vector<uint8_t> derive_arvis_voxel(
    const std::vector<uint8_t>& arvis_orig,
    const std::vector<int>&     voxel_to_orig)
{
    if (arvis_orig.empty() || voxel_to_orig.empty()) return {};
    std::vector<uint8_t> out(voxel_to_orig.size(), 0);
    for (size_t k = 0; k < voxel_to_orig.size(); k++) {
        const int io = voxel_to_orig[k];
        if (io >= 0 && (size_t)io < arvis_orig.size()) {
            out[k] = arvis_orig[(size_t)io];
        }
    }
    return out;
}

// =====================================================================
// derive_is_caudal_voxel -- map per-original-vertex CAUDAL flag to voxel.
// ---------------------------------------------------------------------
// is_caudal_orig: 1 iff original vertex is on the caudal (foot-side)
// half of the mesh (LiverCranioCaudalLabel::CAUDAL).
// Returns empty vector if is_caudal_orig is empty (caller-driven opt-in).
// Same shape as derive_arvis_voxel; provided as a distinct symbol so
// call sites read intentionally rather than passing the wrong array.
// =====================================================================
inline std::vector<uint8_t> derive_is_caudal_voxel(
    const std::vector<uint8_t>& is_caudal_orig,
    const std::vector<int>&     voxel_to_orig)
{
    if (is_caudal_orig.empty() || voxel_to_orig.empty()) return {};
    std::vector<uint8_t> out(voxel_to_orig.size(), 0);
    for (size_t k = 0; k < voxel_to_orig.size(); k++) {
        const int io = voxel_to_orig[k];
        if (io >= 0 && (size_t)io < is_caudal_orig.size()) {
            out[k] = is_caudal_orig[(size_t)io];
        }
    }
    return out;
}

// =====================================================================
// derive_is_rim_src_voxel -- map per-original-vertex RIM flag to voxel.
// ---------------------------------------------------------------------
// is_rim_orig: 1 iff original vertex is on the LiverRegionLabel::RIM band.
// Returns empty vector if is_rim_orig is empty (caller-driven opt-in).
// =====================================================================
inline std::vector<uint8_t> derive_is_rim_src_voxel(
    const std::vector<uint8_t>& is_rim_orig,
    const std::vector<int>&     voxel_to_orig)
{
    if (is_rim_orig.empty() || voxel_to_orig.empty()) return {};
    std::vector<uint8_t> out(voxel_to_orig.size(), 0);
    for (size_t k = 0; k < voxel_to_orig.size(); k++) {
        const int io = voxel_to_orig[k];
        if (io >= 0 && (size_t)io < is_rim_orig.size()) {
            out[k] = is_rim_orig[(size_t)io];
        }
    }
    return out;
}

// =====================================================================
// derive_is_rim_tgt_voxel -- map full-target boundaryDist to voxel-space.
// ---------------------------------------------------------------------
// After tgt voxelization, the per-point boundaryDist field is lost.
// We rebuild it via a 1-NN query: for each session_voxel_tgt[i], find
// the nearest tgt_full point and copy its boundaryDist; threshold to
// produce the rim flag. KDTree build on tgt_full is the only added
// cost (~20 ms for 100k points; one-shot per session).
//
// Returns empty if tgt_boundary_dist_full is empty (opt-in).
// =====================================================================
inline std::vector<uint8_t> derive_is_rim_tgt_voxel(
    const std::vector<glm::vec3>& tgt_full,
    const std::vector<float>&     tgt_boundary_dist_full,
    const std::vector<glm::vec3>& tgt_voxel,
    float                          rim_threshold_px)
{
    if (tgt_boundary_dist_full.empty() ||
        tgt_full.empty()                ||
        tgt_voxel.empty()               ||
        tgt_boundary_dist_full.size() != tgt_full.size())
    {
        return {};
    }

    Reg3DCustom::NanoflannAdaptor adaptor(tgt_full);
    auto tree = Reg3DCustom::buildKDTree(adaptor);

    std::vector<uint8_t> out(tgt_voxel.size(), 0);
    for (size_t i = 0; i < tgt_voxel.size(); i++) {
        size_t nnIdx;
        float  dist_sq;
        if (Reg3DCustom::searchKNN1(*tree, tgt_voxel[i], nnIdx, dist_sq)) {
            const float bd = tgt_boundary_dist_full[nnIdx];
            // bd == 9999 (out-of-mask sentinel) is auto-rejected since
            // it always exceeds any finite threshold.
            out[i] = (bd < rim_threshold_px) ? 1 : 0;
        }
    }
    return out;
}

// =====================================================================
// evaluate_one_v3r_weighted -- rim-rim weighted variant of evaluate_one_v3.
// ---------------------------------------------------------------------
// Same SRT and apply steps as evaluate_one_v3; only the RMSE accumulator
// differs:
//
//   w_i      = 1 + beta * is_rim_src_voxel[j] * is_rim_tgt_voxel[i]
//   sumSq_w += w_i * sq
//   w_sum   += w_i
//   final    = sqrt(sumSq_w / w_sum)
//
// Notes:
//   - Iteration order, max_dist_sq gate, and matched count semantics
//     are IDENTICAL to evaluate_one_v3. Only the final reduction
//     denominator and accumulator change.
//   - When all weights are 1 (no rim hits), sqrt(sumSq/w_sum) reduces
//     to sqrt(sumSq/count), i.e. exactly the V3 formula. So even with
//     beta>0, scenes with no rim-rim coincidences produce the V3 RMSE.
//   - count is used only for the matched_min_required gate (same as V3).
//
// Caller contract (run_one_bipop_v3r enforces these):
//   - is_rim_src_voxel.size() == S.base_positions.size()
//   - is_rim_tgt_voxel.size() == S.tgt_points.size()
//   - beta >= 0.0f (negative betas not supported; would invert the
//     weighting and could yield negative w_sum)
// =====================================================================
inline float evaluate_one_v3r_weighted(
    const CmaesRefine::EvalContextStaticV3& S,
    CmaesRefine::EvalContextScratchV3&      W,
    const CmaesRefine::SRTParamsV3&         srt,
    const std::vector<uint8_t>&             is_rim_src_voxel,
    const std::vector<uint8_t>&             is_rim_tgt_voxel,
    float                                    beta,
    int&                                     matched_out)
{
    using namespace CmaesRefine;

    // ----- 1. SRT matrix (identical to V3) ---------------------------
    const glm::mat4 M = build_srt_matrix_v3(srt, S.centroid);

    // ----- 2. Apply M (identical to V3) ------------------------------
    apply_srt_to_points(M, S.base_positions, W.work_positions);

    // ----- 3. Weighted RMSE accumulator -----------------------------
    const size_t T = S.tgt_points.size();
    float sumSq_w = 0.0f;
    float w_sum   = 0.0f;
    int   count   = 0;
    for (size_t i = 0; i < T; i++) {
        const int j = S.tgt_to_eval[i];
        if (j < 0) continue;
        const glm::vec3 d  = W.work_positions[(size_t)j] - S.tgt_points[i];
        const float     sq = glm::dot(d, d);
        if (sq < S.max_dist_sq) {
            // 乗算 AND: source も target も rim のときだけ重み加算。
            // 配列サイズ不一致があれば 0 倍 (= w=1) にフォールバック。
            float boost = 0.0f;
            if ((size_t)j < is_rim_src_voxel.size() &&
                i           < is_rim_tgt_voxel.size())
            {
                boost = (float)(is_rim_src_voxel[(size_t)j] &
                                 is_rim_tgt_voxel[i]);
            }
            const float w = 1.0f + beta * boost;
            sumSq_w += w * sq;
            w_sum   += w;
            count++;
        }
    }

    matched_out     = count;
    W.matched_count = count;

    if (count == 0 || w_sum <= 0.0f) {
        W.last_rmse = 9.9f;
        return 9.9f;
    }
    W.last_rmse = std::sqrt(sumSq_w / w_sum);
    return W.last_rmse;
}

// =====================================================================
// build_eval_context_v3r -- FORK of build_eval_context_v3.
// ---------------------------------------------------------------------
// Differences from V3 (annotated in-line as "[V3R]"):
//   - Step E (KDTree on base_positions) is replaced by a KDTree on the
//     SUBSET of base_positions selected by subset_idx_voxel.
//   - tgt_to_eval[i] is written as subset_idx_voxel[nn_local], i.e. a
//     voxel-space GLOBAL index (so evaluate_one_v3 sees the same index
//     semantics as in V3 and can be reused without modification).
//
// All other steps (A: tgt copy, B: base copy, C: centroid, D:
// max_dist_sq, F-2: initial_rmse, F: matched_min_required, G: log)
// are byte-identical to V3, including accumulation order in step E
// (V1 fastComputeRMSE: d = sqrt(d2); sumSq += d*d).
//
// QUAD_ALL byte-identical claim:
//   subset_idx_voxel == (0..N-1)
//   -> liver_subset is element-wise equal to S.base_positions
//   -> KDTree built on liver_subset is structurally identical to one
//      built on S.base_positions (same input, same nanoflann adaptor)
//   -> nn_local returned by searchKNN1 equals what V3 would return
//   -> tgt_to_eval[i] = subset_idx_voxel[nn_local] = nn_local
//   -> initial_rmse / post_jitter_matched / matched_min identical
// =====================================================================
inline CmaesRefine::EvalContextStaticV3 build_eval_context_v3r(
    const std::vector<glm::vec3>& src_positions,
    const std::vector<glm::vec3>& src_normals,
    const std::vector<glm::vec3>& tgt_points,
    const ParamsV3R&              params,
    int                           init_matched,
    const std::vector<int>&       subset_idx_voxel,
    int*                          out_post_jitter_matched = nullptr)   // V3R: Issue 1 (jitter retry) のため後段で参照
{
    using namespace CmaesRefine;
    EvalContextStaticV3 S;

    // ----- 0. Empty-input guard (V3R extension: subset empty too) -----
    if (src_positions.empty() || tgt_points.empty() ||
        subset_idx_voxel.empty()) {
        if (params.verbose) {
            std::cerr << "[V3R] build_eval_context: empty src ("
                      << src_positions.size() << "), tgt ("
                      << tgt_points.size() << "), or subset ("
                      << subset_idx_voxel.size() << "); aborting build."
                      << std::endl;
        }
        if (out_post_jitter_matched) *out_post_jitter_matched = 0;
        return S;
    }

    // ----- A. tgt_points (verbatim from V3) ---------------------------
    S.tgt_points = tgt_points;

    // ----- B. base_positions / base_normals (verbatim from V3) --------
    // Note: base_positions still holds the FULL voxel-after-jitter cloud.
    // Only the KDTree (step E) sees the subset; evaluate_one_v3 still
    // applies SRT to all base_positions (subset-external entries
    // contribute no work because tgt_to_eval[i] never points to them).
    S.base_positions = src_positions;
    S.base_normals   = src_normals;

    // ----- C. Centroid (verbatim from V3; V1 byte-identical order) ----
    S.centroid = compute_centroid_v3(S.base_positions);

    // ----- D. max_dist_sq (verbatim from V3; V1 constant 7.36) --------
    constexpr float kRefSceneDiag = 7.36f;
    const float max_dist = params.scene_diag * (1.0f / kRefSceneDiag);
    S.max_dist_sq = max_dist * max_dist;

    // ----- E. [V3R] KDTree on SUBSET of base_positions ----------------
    // Build a small auxiliary array holding only the subset of vertices
    // selected by quadrant_mask. KDTree input is this subset; nn_local
    // is then mapped back to global voxel index via subset_idx_voxel.
    //
    // QUAD_ALL: liver_subset == S.base_positions, KDTree identical to V3.
    std::vector<glm::vec3> liver_subset;
    liver_subset.reserve(subset_idx_voxel.size());
    for (int vi : subset_idx_voxel) {
        // Defensive: vi must be in [0, base_positions.size()).
        // filter_by_quadrant guarantees this when voxel_to_orig is
        // sized to the voxel cloud, but we don't trust silently.
        if (vi >= 0 && (size_t)vi < S.base_positions.size()) {
            liver_subset.push_back(S.base_positions[(size_t)vi]);
        }
    }
    if (liver_subset.empty()) {
        if (params.verbose) {
            std::cerr << "[V3R] build_eval_context: subset mapped to "
                         "empty cloud; aborting build." << std::endl;
        }
        // Returning S with empty base_positions triggers the same
        // EmptyContext abort in run_one_bipop_v3r as a normal empty.
        S.base_positions.clear();
        return S;
    }

    Reg3DCustom::NanoflannAdaptor adaptor(liver_subset);
    auto tree = Reg3DCustom::buildKDTree(adaptor);

    S.tgt_to_eval.assign(S.tgt_points.size(), -1);
    int   post_jitter_matched = 0;
    float post_jitter_sumSq   = 0.0f;
    for (size_t i = 0; i < S.tgt_points.size(); i++) {
        size_t nn_local;
        float  dist_sq;
        if (Reg3DCustom::searchKNN1(*tree, S.tgt_points[i], nn_local, dist_sq)
            && dist_sq < S.max_dist_sq) {
            // [V3R] Map subset-local index back to global voxel index.
            // For QUAD_ALL this is an identity map (subset_idx_voxel[k] == k).
            S.tgt_to_eval[i] = subset_idx_voxel[nn_local];
            post_jitter_matched++;
            // V1 fastComputeRMSE accumulation: d = sqrt(d2); sumSq += d*d.
            // Preserved verbatim from V3 step E.
            const float d = std::sqrt(dist_sq);
            post_jitter_sumSq += d * d;
        }
    }

    // ----- F-2. Post-jitter starting RMSE (verbatim from V3) ----------
    if (post_jitter_matched == 0) {
        S.initial_rmse = 0.0f;
    } else {
        S.initial_rmse = std::sqrt(post_jitter_sumSq
                                   / (float)post_jitter_matched);
    }

    // ----- F. matched_min_required (verbatim from V3) ------------------
    (void)init_matched;
    int min_ok = (int)(post_jitter_matched * params.min_match_ratio);
    if (min_ok < 10) min_ok = 10;
    S.matched_min_required = min_ok;

    // ----- G. Verbose log (V3R-prefixed for log-diff clarity) ---------
    if (params.verbose) {
        std::cout << "[V3R] build_eval_context"
                  << "  src=" << S.base_positions.size()
                  << "  subset=" << liver_subset.size()
                  << "  tgt=" << S.tgt_points.size()
                  << "  session_init_matched=" << init_matched
                  << "  post_jitter_matched=" << post_jitter_matched
                  << "  matched_min=" << S.matched_min_required
                  << "  post_jitter_rmse=" << S.initial_rmse
                  << "  max_dist=" << std::sqrt(S.max_dist_sq) << "m"
                  << "  scene_diag=" << params.scene_diag
                  << std::endl;
    }

    // V3R: Issue 1 案A 用に matched 数を呼び出し側へ返す。
    if (out_post_jitter_matched) *out_post_jitter_matched = post_jitter_matched;

    return S;
}

// =====================================================================
// rebuild_correspondences_v3r -- FORK of rebuild_correspondences_v3.
// ---------------------------------------------------------------------
// Difference: KDTree is built on the SUBSET of `transformed`, and
// tgt_to_eval[i] receives subset_idx_voxel[nn_local] (global voxel idx).
//
// QUAD_ALL: same KDTree input as V3; same nn_local; tgt_to_eval[i]
// matches V3 byte-for-byte at every UPDATE_INTERVAL boundary.
// =====================================================================
inline void rebuild_correspondences_v3r(
    CmaesRefine::EvalContextStaticV3& S,
    const CmaesRefine::SRTParamsV3&   cur_best,
    const ParamsV3R&                  params,
    const std::vector<int>&           subset_idx_voxel)
{
    using namespace CmaesRefine;

    if (S.base_positions.empty() || S.tgt_points.empty() ||
        subset_idx_voxel.empty()) {
        if (params.verbose) {
            std::cerr << "[V3R] rebuild_correspondences: empty context"
                      << " or subset; skipping refresh." << std::endl;
        }
        return;
    }

    // ----- 1. SRT matrix using S.centroid (frozen at build time) ------
    const glm::mat4 M = build_srt_matrix_v3(cur_best, S.centroid);

    // ----- 2. Stack-local SRT-applied buffer (full cloud, V3 same) ----
    // Apply M to ALL base_positions, not just the subset. This keeps
    // step semantics byte-identical to V3 in the QUAD_ALL case.
    // The unused subset-external transforms cost a few hundred us;
    // optimize later if needed (Phase 4 priority is correctness).
    std::vector<glm::vec3> transformed;
    apply_srt_to_points(M, S.base_positions, transformed);

    // ----- 3. [V3R] Subset-only KDTree --------------------------------
    std::vector<glm::vec3> subset_transformed;
    subset_transformed.reserve(subset_idx_voxel.size());
    for (int vi : subset_idx_voxel) {
        if (vi >= 0 && (size_t)vi < transformed.size()) {
            subset_transformed.push_back(transformed[(size_t)vi]);
        }
    }
    if (subset_transformed.empty()) {
        // No vertices available for refresh; leave tgt_to_eval untouched.
        return;
    }

    Reg3DCustom::NanoflannAdaptor adaptor(subset_transformed);
    auto tree = Reg3DCustom::buildKDTree(adaptor);

    // Overwrite every entry. matched -> subset_idx_voxel[nn_local];
    // unmatched -> -1.
    for (size_t i = 0; i < S.tgt_points.size(); i++) {
        size_t nn_local;
        float  dist_sq;
        if (Reg3DCustom::searchKNN1(*tree, S.tgt_points[i], nn_local, dist_sq)
            && dist_sq < S.max_dist_sq) {
            S.tgt_to_eval[i] = subset_idx_voxel[nn_local];
        } else {
            S.tgt_to_eval[i] = -1;
        }
    }
}

// =====================================================================
// run_one_bipop_v3r -- FORK of run_one_bipop.
// ---------------------------------------------------------------------
// Differences from V3 (in-line as "[V3R]"):
//   - Step 2 calls build_eval_context_v3r (subset-aware).
//   - Step 5's UPDATE_INTERVAL refresh calls rebuild_correspondences_v3r.
//   - Step 6 dispatches to compute_full_rmse_local (V3, full-vertex)
//     OR a future subset-RMSE function based on
//     params.full_rmse_use_subset. At S4 only the V3 path is wired up.
//   - Log prefix "[V3]" -> "[V3R]" throughout for log-diff clarity.
// All other code (jitter SRT, CMA-ES init, main loop, time logging)
// is verbatim from V3.
// =====================================================================
inline void run_one_bipop_v3r(
    CmaesRefine::RunContext& rc,
    const ParamsV3R&         params)
{
    using namespace CmaesRefine;

    // ----- 0. Validate inputs (verbatim from V3) ----------------------
    if (!rc.liver_voxel_positions || !rc.liver_full_positions
        || !rc.tgt_voxel_points    || !rc.tgt_full_points) {
        std::cerr << "[V3R] run_one_bipop: null input pointer(s); "
                     "Run aborted." << std::endl;
        rc.best_rmse_inner = 9.9f;
        rc.best_rmse_full  = rc.rmse_before;
        rc.improved        = false;
        rc.stop_reason     = "NullInput";
        return;
    }
    if (rc.liver_voxel_positions->empty() || rc.tgt_voxel_points->empty()) {
        std::cerr << "[V3R] run_one_bipop: empty liver_voxel or tgt_voxel; "
                     "Run aborted." << std::endl;
        rc.best_rmse_inner = 9.9f;
        rc.best_rmse_full  = rc.rmse_before;
        rc.improved        = false;
        rc.stop_reason     = "EmptyInput";
        return;
    }

    // ----- 1. Apply jitter to the VOXEL snapshot ---------------------
    // [V3R Issue 1 案A]: subset 縮小モード (非 QUAD_ALL) では post_jitter
    // 後に subset 全点が tgt KDTree の max_dist を超えて飛び (matched=0)、
    // CMA-ES が Gen 0 で best=0 を記録 → TolFun で即終了する現象が発生
    // (HANDOVER V3 §3.1)。これを防ぐため、matched 不足を検出したら
    // jitter を 0.5x で縮小して最大 3 回リトライし、それでも不十分なら
    // identity (jitter なし) で起点とする。QUAD_ALL では byte-identical
    // 契約を絶対に壊さないため retry を一切発動させない (二重ガード)。
    const glm::vec3 c_jitter = compute_centroid_v3(*rc.liver_voxel_positions);

    // ----- 1.5. Per-Step timing (verbatim from V3) --------------------
    auto step_now    = []{ return std::chrono::high_resolution_clock::now(); };
    using step_ms    = std::chrono::duration<double, std::milli>;
    const auto t_step_t0 = step_now();

    // ----- 2. [V3R] Apply jitter + build eval context (with retry) ----
    SRTParamsV3            jitter_used = rc.jitter;   // rng 由来の値から開始
    std::vector<glm::vec3> liver_after_jitter;
    std::vector<glm::vec3> liver_normals_after_jitter;
    int                    post_jitter_matched = 0;

    const bool is_quad_all =
        (params.quadrant_mask == LiverLeftRightLabel::QUAD_ALL);
    // 二重ガードの 1 つ目: 閾値。
    // QUAD_ALL の subset_size = full voxel (~4076) → min_required ~204、
    // 実機で post_jitter_matched は数千レベルなので絶対に閾値割れしない。
    const int min_required_for_retry = std::max(
        10,
        (int)(params.subset_idx_voxel.size() * 0.05));

    int   jitter_retry = 0;
    float sigma_factor = 1.0f;
    while (true) {
        const glm::mat4 M_jitter_local =
            build_srt_matrix_v3(jitter_used, c_jitter);
        liver_after_jitter.clear();
        apply_srt_to_points(M_jitter_local,
                            *rc.liver_voxel_positions, liver_after_jitter);
        liver_normals_after_jitter.clear();   // V3 と同じく空のまま渡す

        rc.ctx = build_eval_context_v3r(
            liver_after_jitter,
            liver_normals_after_jitter,
            *rc.tgt_voxel_points,
            params,
            rc.init_matched,
            params.subset_idx_voxel,
            &post_jitter_matched);   // V3R: matched 数を取得

        // 二重ガードの 2 つ目: QUAD_ALL では retry 厳禁。
        if (is_quad_all) break;

        // matched 十分なら break。
        if (post_jitter_matched >= min_required_for_retry) break;

        // 既に 3 回リトライ済みなら break (fallback で抜ける)。
        if (jitter_retry >= 3) break;

        // jitter を 0.5x に縮小して再試行。tx/ty/tz/rx/ry/rz_deg は
        // 単純に半減、scale は identity (1.0) との中点に寄せる。
        sigma_factor *= 0.5f;
        jitter_used.tx     *= 0.5f;
        jitter_used.ty     *= 0.5f;
        jitter_used.tz     *= 0.5f;
        jitter_used.rx_deg *= 0.5f;
        jitter_used.ry_deg *= 0.5f;
        jitter_used.rz_deg *= 0.5f;
        jitter_used.scale   = 1.0f + (jitter_used.scale - 1.0f) * 0.5f;
        jitter_retry++;
        std::cerr << "[V3R] jitter retry " << jitter_retry
                  << "/3: sigma_factor=" << sigma_factor
                  << "  post_jitter_matched=" << post_jitter_matched
                  << " < " << min_required_for_retry
                  << "  (Q=" << LiverLeftRightLabel::quadrantMaskString(
                         params.quadrant_mask) << ")"
                  << std::endl;
    }

    // 3 回リトライしても依然 matched 不足なら identity (jitter なし) で起点
    // とする。これは "no jitter" run と同等で、最低限 HemiAuto 後の pose
    // から BIPOP を始められることを保証する。
    if (!is_quad_all
        && jitter_retry >= 3
        && post_jitter_matched < min_required_for_retry) {
        std::cerr << "[V3R] WARNING: jitter retry exhausted, "
                  << "starting from un-jittered pose"
                  << "  (Q=" << LiverLeftRightLabel::quadrantMaskString(
                         params.quadrant_mask) << ")"
                  << std::endl;
        jitter_used = SRTParamsV3{};                       // identity
        liver_after_jitter = *rc.liver_voxel_positions;    // jitter なし
        liver_normals_after_jitter.clear();
        rc.ctx = build_eval_context_v3r(
            liver_after_jitter,
            liver_normals_after_jitter,
            *rc.tgt_voxel_points,
            params,
            rc.init_matched,
            params.subset_idx_voxel,
            &post_jitter_matched);
    }

    // 重要: driver の best_jitter capture が rc.jitter を参照するため、
    // retry/fallback で実際に使った値を rc.jitter に書き戻す。
    // QUAD_ALL では retry が一切発動しないので jitter_used == rc.jitter
    // となり、書き戻しても V3 と byte-identical のまま。
    rc.jitter = jitter_used;

    const auto t_step_t1 = step_now();
    const double t_step_build_eval =
        step_ms(t_step_t1 - t_step_t0).count();

    if (rc.ctx.base_positions.empty() || rc.ctx.tgt_points.empty()) {
        std::cerr << "[V3R] run_one_bipop: build_eval_context failed; "
                     "Run aborted." << std::endl;
        rc.best_rmse_inner = 9.9f;
        rc.best_rmse_full  = rc.rmse_before;
        rc.improved        = false;
        rc.stop_reason     = "EmptyContext";
        return;
    }

    if (params.verbose) {
        std::cout << "[V3R] Run " << (rc.run_index + 1)
        << "  " << (rc.is_local_regime ? "Local" : "Global")
        << "  sigma0=" << rc.sigma0
        << "  cma_seed=" << rc.cma_seed
        << "  mode=" << name_of_search_mode(params.search_mode)
        << "(DIM=" << dim_for_search_mode(params.search_mode) << ")"
        << std::endl;
    }

    // ----- 3. cmaes_init + deterministic srand (V3 verbatim @ SEVEN_DOF) -
    // [NEW V3R/SearchMode] DIM is mode-derived: 7 (SEVEN_DOF, V3
    // byte-identical), 6 (SIX_DOF_RIGID, scale frozen), or 4
    // (FOUR_DOF_XYRXRY, tz/rz/scale frozen). lb/ub/xstart are
    // declared with the static maximum (7); cmaes_init reads only
    // the first DIM entries, and only the first DIM components of
    // any returned sample are written by the C-CMA-ES library.
    const int DIM = dim_for_search_mode(params.search_mode);
    double lb[7], ub[7], xstart[7];
    for (int d = 0; d < DIM; d++) {
        lb[d] = -1.0; ub[d] = 1.0; xstart[d] = 0.0;
    }
    cmaes_t* evo = cmaes_init(DIM, xstart, rc.sigma0,
                              params.lambda, lb, ub);
    if (rc.cma_seed != 0) {
        srand(rc.cma_seed);
        if (params.verbose) {
            std::cout << "[V3R] Deterministic seed: " << rc.cma_seed
                      << std::endl;
        }
    }

    // ----- 4. CMA-ES state (V3 verbatim @ SEVEN_DOF) ------------------
    // [NEW V3R/SearchMode] best_x is sized to the static max (7) but
    // only the first DIM entries are written by the inner copy loop
    // below (`for (int d = 0; d < DIM; d++) best_x[d] = pop[k][d];`).
    // The trailing zeros are never read by srt_from_xvec_v3r_mode in
    // SIX_DOF_RIGID / FOUR_DOF_XYRXRY since those branches index only
    // x[0..5] / x[0..3].
    double best_x[7] = {0,0,0,0,0,0,0};
    float  best_rmse   = rc.ctx.initial_rmse;

    std::vector<EvalContextScratchV3> scratch_pool(1);

    // ----- 5. CMA-ES main loop (verbatim from V3, except rebuild call)-
    const char* stop = nullptr;
    auto now = []{ return std::chrono::high_resolution_clock::now(); };
    using ms_dur = std::chrono::duration<double, std::milli>;

    const auto t_loop_start = now();
    double t_eval = 0.0, t_rebuild = 0.0;

    int gen = 0;
    for (gen = 0; gen < params.maxgen && !stop; gen++) {
        auto tg0 = now();

        double**            pop = cmaes_SamplePopulation(evo);
        std::vector<double> fval(evo->lambda);

        for (int k = 0; k < evo->lambda; k++) {
            SRTParamsV3 srt = srt_from_population_v3r_mode(pop[k], params);
            int   matched = 0;
            // [V3R] Dispatch:
            //   - Default (beta=0 or rim arrays empty): evaluate_one_v3
            //     REUSED VERBATIM. tgt_to_eval[i] holds a global voxel
            //     index (subset_idx_voxel[nn_local]); S.base_positions
            //     still holds the full voxel cloud, so the lookup
            //     `W.work_positions[j]` is valid for every matched i.
            //     Subset-external entries are simply never referenced
            //     by tgt_to_eval (filtered out at KDTree build time).
            //     Byte-identical contract with V3 holds at QUAD_ALL.
            //   - Rim weighting active: evaluate_one_v3r_weighted is
            //     called instead. Same iteration order, gate, and
            //     matched count as V3; only the RMSE reduction uses
            //     w_i = 1 + beta * is_rim_src[j] * is_rim_tgt[i].
            const bool weighted_path =
                (params.beta_rim_weight > 0.0f) &&
                !params.is_rim_src_voxel.empty() &&
                !params.is_rim_tgt_voxel.empty();
            float rmse;
            if (weighted_path) {
                rmse = evaluate_one_v3r_weighted(
                    rc.ctx, scratch_pool[0], srt,
                    params.is_rim_src_voxel,
                    params.is_rim_tgt_voxel,
                    params.beta_rim_weight,
                    matched);
            } else {
                rmse = evaluate_one_v3(rc.ctx, scratch_pool[0],
                                       srt, matched);
            }

            const bool bad = (matched < rc.ctx.matched_min_required)
                             || (rmse == 0.0f);
            fval[k] = bad ? (double)params.penalty_value : (double)rmse;

            if (fval[k] < best_rmse) {
                best_rmse = (float)fval[k];
                for (int d = 0; d < DIM; d++) best_x[d] = pop[k][d];
            }
        }

        auto tg1 = now();
        t_eval += ms_dur(tg1 - tg0).count();

        cmaes_UpdateDistribution(evo, fval.data());

        // UPDATE_INTERVAL refresh ([V3R] subset-aware rebuild) ---------
        if (gen > 0 && gen % params.update_interval == 0) {
            auto tr0 = now();
            const SRTParamsV3 cur_best = srt_from_xvec_v3r_mode(best_x, params);
            rebuild_correspondences_v3r(rc.ctx, cur_best, params,
                                        params.subset_idx_voxel);
            auto tr1 = now();
            t_rebuild += ms_dur(tr1 - tr0).count();
        }

        if (params.verbose && (gen % params.log_every == 0)) {
            std::cout << "[V3R] Gen " << std::setw(4) << gen
                      << "  best=" << std::fixed << std::setprecision(5)
                      << best_rmse
                      << "  sigma=" << std::setprecision(4) << evo->sigma
                      << std::endl;
        }

        stop = cmaes_TestForTermination(evo, params.maxgen,
                                        params.tolfun, params.tolx);
    }

    const double t_loop_total = ms_dur(now() - t_loop_start).count();
    const int    evo_lambda   = evo->lambda;
    rc.generations            = evo->gen;
    rc.stop_reason            = stop ? stop : "MaxGen";
    cmaes_exit(evo);

    // ----- 6. Decode best -- DEFER full-RMSE (V3R 案D') ---------------
    rc.best_srt        = srt_from_xvec_v3r_mode(best_x, params);
    rc.best_rmse_inner = best_rmse;

    // [V3R 案D'] compute_full_rmse_local is deferred to
    // runBipopCmaesV3R, which calls it only for the top-kTopK runs
    // by best_rmse_inner (default kTopK=3). Sentinel -1.0f signals
    // "not yet computed" to the caller. The full_rmse_use_subset
    // dispatch (future hook) warning is also emitted once in the
    // driver instead of per-Run.
    rc.best_rmse_full = -1.0f;   // deferred sentinel
    rc.improved       = false;   // driver sets this after top-K compute

    // ----- 7. Time-breakdown log (V3R prefix) -------------------------
    if (params.verbose) {
        const int    total_evals    = rc.generations * evo_lambda;
        const double t_total        = t_loop_total;
        const double t_other        = t_total - t_eval - t_rebuild;
        const int    rebuild_calls  = (rc.generations - 1)
                                  / params.update_interval;

        std::cout << std::fixed << std::setprecision(1)
                  << "[V3R] === Run " << (rc.run_index + 1)
                  << " Time Breakdown (total " << (int)t_total
                  << " ms, " << total_evals << " evals) ===" << std::endl;
        if (t_total > 0.0) {
            std::cout << "[V3R]   evaluate_one (sum) : " << (int)t_eval
                      << " ms (" << (int)(100*t_eval/t_total) << "%)"
                      << std::endl
                      << "[V3R]   rebuild_corr (sum) : " << (int)t_rebuild
                      << " ms (" << (int)(100*t_rebuild/t_total) << "%)"
                      << "  [" << rebuild_calls << " calls]"
                      << std::endl
                      << "[V3R]   cmaes/log/other    : " << (int)t_other
                      << " ms (" << (int)(100*t_other/t_total) << "%)"
                      << std::endl;
        }
        std::cout << "[V3R]   build_eval_ctx     : "
                  << (int)t_step_build_eval << " ms"
                  << "   compute_full_rmse  : (deferred to top-K)"
                  << std::endl;
        std::cout << std::defaultfloat << std::setprecision(6)
                  << "[V3R] Run " << (rc.run_index + 1)
                  << "  best_inner=" << rc.best_rmse_inner
                  << "  best_full=(deferred)"
                  << "  stop="       << rc.stop_reason
                  << std::endl;
    }
}

// =====================================================================
// runBipopCmaesV3R -- FORK of runBipopCmaesV3.
// ---------------------------------------------------------------------
// Session driver. Delta from V3:
//   - Phase C (after voxel downsample): build voxel_to_orig via NN
//     reverse map (build_voxel_to_orig).
//   - Phase D: derive subset_idx_voxel from voxel_to_orig + labels +
//     quadrant_mask (filter_by_quadrant). Stored on the (mutable)
//     params object so run_one_bipop_v3r can reach them without an
//     extra parameter slot.
//   - Run loop dispatches to run_one_bipop_v3r instead of run_one_bipop.
//   - 案D': post-loop top-K full-RMSE screening. compute_full_rmse_local
//     is called only for the kTopK=3 runs with lowest best_rmse_inner,
//     reducing expensive KDTree calls from N_STARTS(10) to 3.
//     full_rmse_use_subset warning emitted once here instead of per-Run.
//
// Determinism: outer_seed and cma_base are passed in by the caller and
// used IDENTICALLY to V3 (mt19937, d01 sequence, jitter formulas, run==0
// gate). At QUAD_ALL the entire sequence of double draws is unchanged
// from V3.
//
// Note on the params&: we accept ParamsV3R by non-const reference so we
// can write the session-derived voxel_to_orig / subset_idx_voxel back
// onto it. The caller's instance is mutated in this single way; all
// other fields are read-only here.
// =====================================================================
inline CmaesRefine::ResultV3 runBipopCmaesV3R(
    const std::vector<glm::vec3>& start_liver_verts,
    const std::vector<glm::vec3>& start_liver_normals,
    const std::vector<glm::vec3>& tgt_points,
    ParamsV3R&                    params,
    float                         rmse_before,
    int                           init_matched,
    uint32_t                      outer_seed,
    uint32_t                      cma_base)
{
    using namespace CmaesRefine;

    ResultV3 r;
    r.rmse_before = rmse_before;
    r.rmse_after  = rmse_before;
    r.improved    = false;

    // ----- Empty-input early return (verbatim from V3) ----------------
    if (start_liver_verts.empty() || tgt_points.empty()) {
        std::cerr << "[V3R] runBipopCmaesV3R: empty start_liver_verts ("
                  << start_liver_verts.size() << ") or tgt_points ("
                  << tgt_points.size() << "); aborting." << std::endl;
        return r;
    }

    // ----- [NEW V3R/SearchMode] One-shot mode banner ------------------
    if (params.verbose) {
        std::cout << "[V3R session] search_mode="
                  << name_of_search_mode(params.search_mode)
                  << " (DIM=" << dim_for_search_mode(params.search_mode) << ")"
                  << "  tx_range=" << std::fixed << std::setprecision(4)
                  << params.tx_range
                  << "  jitter_local_t=" << params.jitter_local_t
                  << "  min_match_ratio=" << std::setprecision(2)
                  << params.min_match_ratio
                  << std::defaultfloat << std::setprecision(6)
                  << std::endl;
    }

    // ----- Pre-compute session-wide max_dist_sq (verbatim from V3) ----
    constexpr float kRefSceneDiag_session = 7.36f;
    const float max_dist_session    = params.scene_diag
                                   * (1.0f / kRefSceneDiag_session);
    const float max_dist_sq_session = max_dist_session * max_dist_session;

    // ----- V3-2 case C: voxel-downsample (verbatim from V3) -----------
    auto sess_now    = []{ return std::chrono::high_resolution_clock::now(); };
    using sess_ms    = std::chrono::duration<double, std::milli>;
    auto t_sess_t0   = sess_now();

    std::vector<glm::vec3> session_voxel_liver;
    std::vector<glm::vec3> session_voxel_tgt;
    const float src_voxel_size = (params.src_voxel_ratio > 0.0f)
                                     ? (params.src_voxel_ratio * params.scene_diag) : 0.0f;
    const float tgt_voxel_size = (params.tgt_voxel_ratio > 0.0f)
                                     ? (params.tgt_voxel_ratio * params.scene_diag) : 0.0f;

    auto t_sess_voxel0 = sess_now();
    voxel_downsample_v3(start_liver_verts, src_voxel_size,
                        session_voxel_liver);
    auto t_sess_voxel1 = sess_now();
    voxel_downsample_v3(tgt_points,        tgt_voxel_size,
                        session_voxel_tgt);
    auto t_sess_voxel2 = sess_now();

    if (params.verbose) {
        std::cout << "[V3R session/Time] voxel src ("
                  << start_liver_verts.size() << "->"
                  << session_voxel_liver.size() << ") : "
                  << std::fixed << std::setprecision(1)
                  << sess_ms(t_sess_voxel1 - t_sess_voxel0).count()
                  << " ms" << std::defaultfloat << std::endl
                  << "[V3R session/Time] voxel tgt ("
                  << tgt_points.size() << "->"
                  << session_voxel_tgt.size() << ") : "
                  << std::fixed << std::setprecision(1)
                  << sess_ms(t_sess_voxel2 - t_sess_voxel1).count()
                  << " ms" << std::defaultfloat << std::endl;
    }

    // ----- Phase C [V3R]: build voxel_to_orig via NN reverse map ------
    auto t_sess_v2o0 = sess_now();
    params.voxel_to_orig =
        build_voxel_to_orig(start_liver_verts, session_voxel_liver);
    auto t_sess_v2o1 = sess_now();

    // ----- Phase C2 [V3R-W]: derive arvis_voxel + rim arrays (opt-in) -
    // All three are session-derived from caller-provided original-space
    // arrays. Empty input -> empty output -> downstream falls back to
    // the standard V3R path (no visibility filter, no rim weighting).
    if (params.use_arvis_filter) {
        params.arvis_voxel = derive_arvis_voxel(
            params.arvis_orig, params.voxel_to_orig);
        if (params.verbose) {
            int n_vis = 0;
            for (uint8_t v : params.arvis_voxel) if (v) n_vis++;
            std::cout << "[V3R-W] arvis: visible voxels = "
                      << n_vis << "/" << params.arvis_voxel.size()
                      << "  (orig src visible="
                      << std::count(params.arvis_orig.begin(),
                                    params.arvis_orig.end(), (uint8_t)1)
                      << "/" << params.arvis_orig.size() << ")"
                      << std::endl;
        }
    } else {
        params.arvis_voxel.clear();
    }

    if (params.beta_rim_weight > 0.0f) {
        params.is_rim_src_voxel = derive_is_rim_src_voxel(
            params.is_rim_orig, params.voxel_to_orig);
        params.is_rim_tgt_voxel = derive_is_rim_tgt_voxel(
            tgt_points, params.tgt_boundary_dist_full,
            session_voxel_tgt, params.rim_tgt_threshold_px);
        if (params.verbose) {
            int n_rim_src = 0;
            for (uint8_t v : params.is_rim_src_voxel) if (v) n_rim_src++;
            int n_rim_tgt = 0;
            for (uint8_t v : params.is_rim_tgt_voxel) if (v) n_rim_tgt++;
            std::cout << "[V3R-W] rim weighting: beta="
                      << params.beta_rim_weight
                      << "  src_rim=" << n_rim_src
                      << "/" << params.is_rim_src_voxel.size()
                      << "  tgt_rim=" << n_rim_tgt
                      << "/" << params.is_rim_tgt_voxel.size()
                      << "  thresh=" << params.rim_tgt_threshold_px << "px"
                      << std::endl;
        }
    } else {
        params.is_rim_src_voxel.clear();
        params.is_rim_tgt_voxel.clear();
    }

    // Caudal voxel mask (R-feat-2). Independent of arvis. Empty input
    // -> empty output -> filter_by_quadrant_with_arvis_caudal treats
    // it as "not requested" (no caudal filtering).
    if (params.use_caudal_only) {
        params.is_caudal_voxel = derive_is_caudal_voxel(
            params.is_caudal_orig, params.voxel_to_orig);
        if (params.verbose) {
            int n_caudal = 0;
            for (uint8_t v : params.is_caudal_voxel) if (v) n_caudal++;
            std::cout << "[V3R-W] caudal: caudal voxels = "
                      << n_caudal << "/" << params.is_caudal_voxel.size()
                      << "  (orig src caudal="
                      << std::count(params.is_caudal_orig.begin(),
                                    params.is_caudal_orig.end(), (uint8_t)1)
                      << "/" << params.is_caudal_orig.size() << ")"
                      << "  combine="
                      << (params.arvis_caudal_combine == 0 ? "AND" : "OR")
                      << std::endl;
        }
    } else {
        params.is_caudal_voxel.clear();
    }

    // ----- Phase D [V3R]: derive subset_idx_voxel ---------------------
    // Combined filter handles arvis + caudal in one pass with selectable
    // AND/OR mode. When both arvis_voxel and is_caudal_voxel are empty
    // (neither feature requested), it degenerates to filter_by_quadrant
    // verbatim, so QUAD_ALL byte-identical with V3 is preserved.
    params.subset_idx_voxel = filter_by_quadrant_with_arvis_caudal(
        params.voxel_to_orig,
        params.region_labels,
        params.lr_labels,
        params.quadrant_mask,
        params.arvis_voxel,
        params.is_caudal_voxel,
        params.arvis_caudal_combine);
    auto t_sess_v2o2 = sess_now();

    if (params.verbose) {
        std::cout << "[V3R session/Time] voxel_to_orig ("
                  << session_voxel_liver.size() << " NN lookups) : "
                  << std::fixed << std::setprecision(1)
                  << sess_ms(t_sess_v2o1 - t_sess_v2o0).count()
                  << " ms" << std::defaultfloat << std::endl
                  << "[V3R session/Time] subset_filter : "
                  << std::fixed << std::setprecision(1)
                  << sess_ms(t_sess_v2o2 - t_sess_v2o1).count()
                  << " ms" << std::defaultfloat << std::endl
                  << "[V3R] quadrant_mask="
                  << LiverLeftRightLabel::quadrantMaskString(params.quadrant_mask)
                  << "  arvis=" << (params.use_arvis_filter ? "ON" : "OFF")
                  << "  caudal=" << (params.use_caudal_only ? "ON" : "OFF")
                  << "  combine="
                  << (params.arvis_caudal_combine == 0 ? "AND" : "OR")
                  << "  subset_size=" << params.subset_idx_voxel.size()
                  << "/" << session_voxel_liver.size()
                  << " (voxel-space)" << std::endl;
    }

    // Defensive: subset empty (e.g. QUAD_NONE) -> abort cleanly.
    if (params.subset_idx_voxel.empty()) {
        std::cerr << "[V3R] subset_idx_voxel is empty (mask=0x"
                  << std::hex << (int)params.quadrant_mask << std::dec
                  << "); no vertices selected. Aborting session."
                  << std::endl;
        return r;
    }

    // ----- BIPOP outer rng init (verbatim from V3) --------------------
    std::mt19937 rng(outer_seed);
    std::uniform_real_distribution<float> d01(0.0f, 1.0f);

    if (params.verbose) {
        std::cout << "[V3R] === Starting BIPOP-CMA-ES V3R ===" << std::endl
                  << "[V3R] outer_seed=" << outer_seed
                  << "  cma_base=" << cma_base
                  << "  rmse_before=" << rmse_before
                  << "  init_matched=" << init_matched
                  << "  scene_diag=" << params.scene_diag
                  << std::endl
                  << "[V3R] src: " << start_liver_verts.size()
                  << " -> " << session_voxel_liver.size()
                  << " (voxel=" << src_voxel_size << ", ratio="
                  << params.src_voxel_ratio << ")" << std::endl
                  << "[V3R] tgt: " << tgt_points.size()
                  << " -> " << session_voxel_tgt.size()
                  << " (voxel=" << tgt_voxel_size << ", ratio="
                  << params.tgt_voxel_ratio << ")" << std::endl;
    }

    // ----- 10 BIPOP runs (verbatim from V3, except dispatch target) ---
    const int N_STARTS = 10;

    float       best_rmse_full     = rmse_before;
    int         best_run_idx       = -1;
    SRTParamsV3 best_jitter;
    SRTParamsV3 best_srt;
    std::string best_stop_reason   = "NoImprovement";
    int         total_generations  = 0;

    auto t_sess_runs0 = sess_now();
    double t_run_outer_wall_sum = 0.0;

    // [V3R 案D'] Minimal per-Run summary for post-loop top-K
    // full-RMSE screening. Only stores the fields needed to call
    // compute_full_rmse_local after the Run loop. The full RunContext
    // (EvalContext with KDTree) is NOT saved -- only the small SRT
    // parameters and inner-RMSE are needed. Session-level pointers
    // (liver_full, liver_voxel, tgt_full) are identical for all Runs
    // and referenced directly from session variables below.
    struct RunSummaryD {
        SRTParamsV3 jitter;
        SRTParamsV3 best_srt;
        float       best_rmse_inner = 9.9f;
        float       best_rmse_full  = -1.0f;  // -1.0f = not yet computed
        bool        improved        = false;
        std::string stop_reason     = "NotRun";
        int         generations     = 0;
    };
    std::vector<RunSummaryD> all_summaries(N_STARTS);

    for (int run = 0; run < N_STARTS; run++) {
        const auto t_one_run_t0 = sess_now();
        RunContext rc;
        rc.run_index             = run;
        rc.is_local_regime       = (run % 2 == 0);
        rc.cma_seed              = cma_base + (uint32_t)run;
        rc.rmse_before           = rmse_before;
        rc.init_matched          = init_matched;
        rc.max_dist_sq           = max_dist_sq_session;
        rc.liver_full_positions  = &start_liver_verts;
        rc.liver_full_normals    = &start_liver_normals;
        rc.liver_voxel_positions = &session_voxel_liver;
        rc.tgt_full_points       = &tgt_points;
        rc.tgt_voxel_points      = &session_voxel_tgt;

        // ----- Generate sigma0 + jitter (V1 d01 consumption order) ---
        // [V3R] IDENTICAL to V3: 8 d01 draws per Run, exact same order
        // (sigma0, tx, ty, tz, rx, ry, rz, sc), so the rng state
        // sequence matches V3 byte-for-byte.
        SRTParamsV3 jitter;
        if (rc.is_local_regime) {
            rc.sigma0     = 0.3 + d01(rng) * 0.4;
            const float lt = params.jitter_local_t;
            jitter.tx     = (d01(rng) * 2.0f - 1.0f) * lt;
            jitter.ty     = (d01(rng) * 2.0f - 1.0f) * lt;
            jitter.tz     = (d01(rng) * 2.0f - 1.0f) * lt;
            jitter.rx_deg = (d01(rng) * 2.0f - 1.0f) * 10.0f;
            jitter.ry_deg = (d01(rng) * 2.0f - 1.0f) * 10.0f;
            jitter.rz_deg = (d01(rng) * 2.0f - 1.0f) * 10.0f;
            jitter.scale  = 0.95f + d01(rng) * 0.10f;
        } else {
            rc.sigma0     = 0.5 + d01(rng) * 0.5;
            const float gt = params.jitter_global_t;
            jitter.tx     = (d01(rng) * 2.0f - 1.0f) * gt;
            jitter.ty     = (d01(rng) * 2.0f - 1.0f) * gt;
            jitter.tz     = (d01(rng) * 2.0f - 1.0f) * gt;
            jitter.rx_deg = (d01(rng) * 2.0f - 1.0f) * 30.0f;
            jitter.ry_deg = (d01(rng) * 2.0f - 1.0f) * 30.0f;
            jitter.rz_deg = (d01(rng) * 2.0f - 1.0f) * 30.0f;
            jitter.scale  = 0.90f + d01(rng) * 0.20f;
        }

        if (run == 0) {
            rc.jitter = SRTParamsV3{};
        } else {
            // [NEW V3R/SearchMode] Mask inactive jitter components for
            // reduced-DoF modes. The 8-draw d01() sequence above is
            // intentionally executed in full regardless of search_mode
            // so the rng state matches V3R(7-DoF) byte-for-byte for the
            // same outer_seed; only the VALUES stored in rc.jitter
            // differ. SEVEN_DOF: pass-through.
            rc.jitter = mask_jitter_for_mode(jitter, params.search_mode);
        }

        if (params.verbose) {
            std::cout << "[V3R] Run " << (run+1) << "/" << N_STARTS
                      << "  " << (rc.is_local_regime ? "Local " : "Global")
                      << "  sigma0=" << std::fixed << std::setprecision(4)
                      << rc.sigma0
                      << "  cma_seed=" << rc.cma_seed
                      << std::defaultfloat << std::setprecision(6)
                      << std::endl;
        }

        // ----- Execute one Run ([V3R] dispatch target) ----------------
        run_one_bipop_v3r(rc, params);

        // ----- Save Run summary for post-loop top-K screening (案D') --
        // Store only the minimal fields needed: jitter (SRTParamsV3 for
        // compute_full_rmse_local centroid alignment), best_srt (the
        // CMA-ES result), best_rmse_inner (for top-K sort), and
        // bookkeeping. The heavy EvalContext (KDTree) is NOT saved.
        all_summaries[run].jitter          = rc.jitter;
        all_summaries[run].best_srt        = rc.best_srt;
        all_summaries[run].best_rmse_inner = rc.best_rmse_inner;
        all_summaries[run].best_rmse_full  = rc.best_rmse_full;  // -1.0f
        all_summaries[run].improved        = false;              // deferred
        all_summaries[run].stop_reason     = rc.stop_reason;
        all_summaries[run].generations     = rc.generations;
        total_generations += rc.generations;

        const auto t_one_run_t1 = sess_now();
        t_run_outer_wall_sum   += sess_ms(t_one_run_t1 - t_one_run_t0).count();
    }

    auto t_sess_runs1 = sess_now();
    if (params.verbose) {
        std::cout << "[V3R session/Time] runs loop wall-clock : "
                  << std::fixed << std::setprecision(1)
                  << sess_ms(t_sess_runs1 - t_sess_runs0).count()
                  << " ms"
                  << "  (sum of per-run outer = "
                  << t_run_outer_wall_sum << " ms)"
                  << std::defaultfloat << std::endl;
    }

    // ----------------------------------------------------------------
    // [V3R 案D'] Post-loop top-K full-RMSE computation.
    // ----------------------------------------------------------------
    // Select the kTopK runs with lowest best_rmse_inner and compute
    // compute_full_rmse_local only for those. This reduces expensive
    // KDTree-build+query calls from N_STARTS (10) to kTopK (3),
    // saving ~70% of Phase D time. The rationale: a Run whose CMA-ES
    // inner RMSE is not in the top-K is extremely unlikely to rank #1
    // by full RMSE (empirically verified; see HANDOVER V3-3 §7.1).
    //
    // full_rmse_use_subset dispatch is handled here once instead of
    // per-Run; the future hook warning is emitted on first call only.
    // ----------------------------------------------------------------
    {
        constexpr int   kTopK               = 3;
        constexpr float kRefSceneDiag_full  = 7.36f;
        const float max_dist_full    = params.scene_diag
                                    * (1.0f / kRefSceneDiag_full);
        const float max_dist_sq_full = max_dist_full * max_dist_full;

        // One-time warning for the future full_rmse_use_subset hook.
        if (params.full_rmse_use_subset &&
            params.quadrant_mask != LiverLeftRightLabel::QUAD_ALL) {
            static bool warned_once_d = false;
            if (!warned_once_d) {
                std::cerr << "[V3R/D'] warning: full_rmse_use_subset=true "
                             "is reserved for future work; using full-vertex "
                             "RMSE for top-K screening." << std::endl;
                warned_once_d = true;
            }
        }

        // Sort run indices by best_rmse_inner (ascending). Use
        // partial_sort so only the first kTopK are guaranteed sorted.
        std::vector<std::pair<float, int>> inner_rank(N_STARTS);
        for (int i = 0; i < N_STARTS; i++) {
            inner_rank[i] = { all_summaries[i].best_rmse_inner, i };
        }
        const int actual_topk = std::min(kTopK, N_STARTS);
        std::partial_sort(inner_rank.begin(),
                          inner_rank.begin() + actual_topk,
                          inner_rank.end());

        if (params.verbose) {
            std::cout << "[V3R/D'] top-" << actual_topk
                      << " by inner (will compute full RMSE):" << std::endl;
            for (int ki = 0; ki < actual_topk; ki++) {
                std::cout << "[V3R/D']   rank " << (ki + 1)
                << " -> Run " << (inner_rank[ki].second + 1)
                << "  inner=" << std::fixed << std::setprecision(6)
                << inner_rank[ki].first
                << std::defaultfloat << std::endl;
            }
        }

        const auto t_topk_t0 = sess_now();

        for (int ki = 0; ki < actual_topk; ki++) {
            const int run = inner_rank[ki].second;
            RunSummaryD& s = all_summaries[run];

            s.best_rmse_full = compute_full_rmse_local(
                s.jitter, s.best_srt,
                session_voxel_liver,    // shared session data
                start_liver_verts,      // shared session data
                tgt_points,             // shared session data
                max_dist_sq_full);
            s.improved = (s.best_rmse_full < rmse_before);

            if (params.verbose) {
                std::cout << "[V3R/D'] Run " << (run + 1)
                << "  inner=" << std::fixed << std::setprecision(6)
                << s.best_rmse_inner
                << "  full="  << s.best_rmse_full
                << (s.improved ? "  [+]" : "  [-]")
                << std::defaultfloat << std::endl;
            }

            if (s.best_rmse_full < best_rmse_full) {
                best_rmse_full   = s.best_rmse_full;
                best_run_idx     = run;
                best_jitter      = s.jitter;
                best_srt         = s.best_srt;
                best_stop_reason = s.stop_reason;
            }
        }

        const auto t_topk_t1 = sess_now();
        if (params.verbose) {
            std::cout << "[V3R/D'] top-" << actual_topk << " full-RMSE cost : "
                      << std::fixed << std::setprecision(1)
                      << sess_ms(t_topk_t1 - t_topk_t0).count() << " ms"
                      << "  best_run=" << (best_run_idx < 0 ? std::string("none")
                                                            : std::to_string(best_run_idx + 1))
                      << "  best_full=" << std::setprecision(6) << best_rmse_full
                      << std::defaultfloat << std::endl;
        }
    } // end 案D' top-K block

    // ----- Assemble best_world_matrix (verbatim from V3) --------------
    if (best_run_idx >= 0) {
        const glm::vec3 c_pre = compute_centroid_v3(session_voxel_liver);
        const glm::mat4 M_jit = build_srt_matrix_v3(best_jitter, c_pre);
        std::vector<glm::vec3> voxel_after_jitter;
        apply_srt_to_points(M_jit, session_voxel_liver, voxel_after_jitter);
        const glm::vec3 c_post = compute_centroid_v3(voxel_after_jitter);
        const glm::mat4 M_best = build_srt_matrix_v3(best_srt, c_post);

        r.best_world_matrix = M_best * M_jit;
        r.best_jitter       = best_jitter;
        r.best_srt          = best_srt;
        r.rmse_after        = best_rmse_full;
        r.improved          = (best_rmse_full < rmse_before);
        r.best_run_idx      = best_run_idx;
        r.last_stop_reason  = best_stop_reason;
        r.total_generations = total_generations;
    } else {
        r.best_world_matrix = glm::mat4(1.0f);
        r.rmse_after        = rmse_before;
        r.improved          = false;
        r.best_run_idx      = -1;
        r.last_stop_reason  = "NoImprovement";
        r.total_generations = total_generations;
    }

    if (params.verbose) {
        const double t_session_total =
            sess_ms(sess_now() - t_sess_t0).count();
        std::cout << std::defaultfloat << std::setprecision(6);
        std::cout << "[V3R] === BIPOP-CMA-ES V3R DONE ==="
                  << "  best_run="
                  << (best_run_idx < 0 ? std::string("none")
                                       : std::to_string(best_run_idx + 1))
                  << "  RMSE: " << rmse_before << " -> " << r.rmse_after
                  << "  delta=" << (rmse_before - r.rmse_after)
                  << (r.improved ? "  [IMPROVED]" : "  [NO CHANGE]")
                  << "  total_gens=" << total_generations
                  << "  quadrant="
                  << LiverLeftRightLabel::quadrantMaskString(params.quadrant_mask)
                  << std::endl
                  << "[V3R session/Time] DRIVER TOTAL : "
                  << std::fixed << std::setprecision(1)
                  << t_session_total << " ms"
                  << std::defaultfloat << std::endl;
    }

    // (g_callIdx++ is the caller's responsibility, performed in the
    //  RegistrationActions::runBipopCmaesV3R wrapper after this returns.)
    return r;
}

} // namespace CmaesRefineV3R

#endif // CMAES_REFINE_V3R_H
