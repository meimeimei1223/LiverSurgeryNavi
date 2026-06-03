/* hansen_renames.h  — AUTO-GENERATED (do not hand-edit)
 *
 * Prefixes every external cmaes_* identifier of the upstream Hansen
 * c-cmaes (src/) with HANSEN_, so the upstream library can be compiled
 * and linked alongside the thin engine-facing API in cmaes.h without
 * symbol clashes. cmaes_boundary_* is intentionally NOT renamed.
 *
 * Included only by hansen_cmaes_renamed.c (the C shim) and by the
 * upstream-include section of cmaes_adapter.cpp. Never by the engines.
 * Upstream src/ files are byte-identical to upstream; renaming is done
 * purely via these macros (HANDOFF v2 FIX3).
 *
 * Regenerate: see scripts note in README. Token count: 62
 */
#ifndef HANSEN_RENAMES_H
#define HANSEN_RENAMES_H
#define cmaes_FATAL                        HANSEN_cmaes_FATAL
#define cmaes_Get                          HANSEN_cmaes_Get
#define cmaes_GetInto                      HANSEN_cmaes_GetInto
#define cmaes_GetNew                       HANSEN_cmaes_GetNew
#define cmaes_GetPtr                       HANSEN_cmaes_GetPtr
#define cmaes_NewDouble                    HANSEN_cmaes_NewDouble
#define cmaes_Optimize                     HANSEN_cmaes_Optimize
#define cmaes_PerturbSolutionInto          HANSEN_cmaes_PerturbSolutionInto
#define cmaes_ReSampleSingle               HANSEN_cmaes_ReSampleSingle
#define cmaes_ReSampleSingle_old           HANSEN_cmaes_ReSampleSingle_old
#define cmaes_ReadFromFile                 HANSEN_cmaes_ReadFromFile
#define cmaes_ReadFromFilePtr              HANSEN_cmaes_ReadFromFilePtr
#define cmaes_ReadSignals                  HANSEN_cmaes_ReadSignals
#define cmaes_ReestimateDistribution       HANSEN_cmaes_ReestimateDistribution
#define cmaes_SamplePopulation             HANSEN_cmaes_SamplePopulation
#define cmaes_SampleSingleInto             HANSEN_cmaes_SampleSingleInto
#define cmaes_SayHello                     HANSEN_cmaes_SayHello
#define cmaes_SetMean                      HANSEN_cmaes_SetMean
#define cmaes_Test                         HANSEN_cmaes_Test
#define cmaes_TestForTermination           HANSEN_cmaes_TestForTermination
#define cmaes_TestMinStdDevs               HANSEN_cmaes_TestMinStdDevs
#define cmaes_UpdateDistribution           HANSEN_cmaes_UpdateDistribution
#define cmaes_UpdateEigensystem            HANSEN_cmaes_UpdateEigensystem
#define cmaes_WriteToFile                  HANSEN_cmaes_WriteToFile
#define cmaes_WriteToFileAW                HANSEN_cmaes_WriteToFileAW
#define cmaes_WriteToFilePtr               HANSEN_cmaes_WriteToFilePtr
#define cmaes_exit                         HANSEN_cmaes_exit
#define cmaes_h                            HANSEN_cmaes_h
#define cmaes_init                         HANSEN_cmaes_init
#define cmaes_init_final                   HANSEN_cmaes_init_final
#define cmaes_init_para                    HANSEN_cmaes_init_para
#define cmaes_initials                     HANSEN_cmaes_initials
#define cmaes_interface                    HANSEN_cmaes_interface
#define cmaes_prefix                       HANSEN_cmaes_prefix
#define cmaes_random                       HANSEN_cmaes_random
#define cmaes_random_Gauss                 HANSEN_cmaes_random_Gauss
#define cmaes_random_Start                 HANSEN_cmaes_random_Start
#define cmaes_random_Uniform               HANSEN_cmaes_random_Uniform
#define cmaes_random_exit                  HANSEN_cmaes_random_exit
#define cmaes_random_init                  HANSEN_cmaes_random_init
#define cmaes_random_t                     HANSEN_cmaes_random_t
#define cmaes_readpara_Read                HANSEN_cmaes_readpara_Read
#define cmaes_readpara_ReadFromFile        HANSEN_cmaes_readpara_ReadFromFile
#define cmaes_readpara_SetWeights          HANSEN_cmaes_readpara_SetWeights
#define cmaes_readpara_SupplementDefaults  HANSEN_cmaes_readpara_SupplementDefaults
#define cmaes_readpara_WriteToFile         HANSEN_cmaes_readpara_WriteToFile
#define cmaes_readpara_exit                HANSEN_cmaes_readpara_exit
#define cmaes_readpara_init                HANSEN_cmaes_readpara_init
#define cmaes_readpara_t                   HANSEN_cmaes_readpara_t
#define cmaes_resume_distribution          HANSEN_cmaes_resume_distribution
#define cmaes_signals                      HANSEN_cmaes_signals
#define cmaes_t                            HANSEN_cmaes_t
#define cmaes_timings_                     HANSEN_cmaes_timings_
#define cmaes_timings_init                 HANSEN_cmaes_timings_init
#define cmaes_timings_start                HANSEN_cmaes_timings_start
#define cmaes_timings_started              HANSEN_cmaes_timings_started
#define cmaes_timings_t                    HANSEN_cmaes_timings_t
#define cmaes_timings_tic                  HANSEN_cmaes_timings_tic
#define cmaes_timings_toc                  HANSEN_cmaes_timings_toc
#define cmaes_timings_update               HANSEN_cmaes_timings_update
#define cmaes_version                      HANSEN_cmaes_version
#define cmaes_write                        HANSEN_cmaes_write
#endif /* HANSEN_RENAMES_H */
