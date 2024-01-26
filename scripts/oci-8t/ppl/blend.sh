#!/bin/bash
# BASE_DIR="/lustre/fsw/portfolios/adlr/projects/adlr_nlp_arch/adlr_nlp_sharing/nvllm-8t/data/tokens-shuffle"
# BASE_DIR="/lustre/fs3/portfolios/adlr/projects/adlr_nlp_arch/adlr_nlp_sharing/nvllm-8t/data/tokens-shuffle"
BASE_DIR="/lustre/fs6/portfolios/adlr/users/lmcafee/retro/megatrons/core-converter/scripts/oci-8t/ppl/data"

#english datasets home
NONCC_HOME="${BASE_DIR}/english/non-cc"
MTNLG_HOME="${BASE_DIR}/english/mtnlg"
CC_HOME="${BASE_DIR}/english/cc"

#multilingual datasets home
NON_ENGLISH_HOME="${BASE_DIR}/non-english"

#code datasets home
STARCODER_V1_HOME="${BASE_DIR}/code/starcoder_v1"
STARCODER_V2_HOME="${BASE_DIR}/code/starcoder_v2"

### NON CC
CLR="${NONCC_HOME}/CourtListener_fixed_shuf_text_document"
P2O="${NONCC_HOME}/Pes2o_hq_shuf_text_document"
GLR="${NONCC_HOME}/Global_Reg_Fixed_shuf_text_document"
BRXV="${NONCC_HOME}/bioRxiv_fixed_shuf_text_document"
PMC="${NONCC_HOME}/PubmedCentral_hq_shuf_text_document"
DOHTML="${NONCC_HOME}/DOAB-HTML_shuf_text_document"
USPTO="${NONCC_HOME}/USPTO_hq_shuf_text_document"
OWM="${NONCC_HOME}/Open-Web-Math_hq_shuf_text_document"
NDLTD="${NONCC_HOME}/ndltd_fixed_shuf_text_document"
DOPDF="${NONCC_HOME}/doab_pdf_shuf_text_document"
LIBGENHF="${NONCC_HOME}/libgen-100k-clean-00_shuf_text_document"
LIBGEN="${NONCC_HOME}/libgen-rs-epub_shuf_text_document"

### MTNLG
B3="${MTNLG_HOME}/Books3_fuzzy_deduped_english_all_text_document"
OWT2="${MTNLG_HOME}/OpenWebText2_hq_text_document"
SE="${MTNLG_HOME}/StackExchange_fuzzy_deduped_english_all_shuf_text_document"
PMA="${MTNLG_HOME}/PubMedAbs_fuzzy_deduped_english_all_text_document"
WIK2023="${MTNLG_HOME}/Wikipedia-2023-05-01_fuzzy_deduped_english_all_text_document"
GUT="${MTNLG_HOME}/Gutenberg_fuzzy_deduped_english_all_text_document"
BC2="${MTNLG_HOME}/BookCorpus_fuzzy_deduped_english_all_text_document"
NIH="${MTNLG_HOME}/NIHExporter_fuzzy_deduped_english_all_text_document"
ARX2023="${MTNLG_HOME}/ArXiv-2023-05-01_fuzzy_deduped_english_all_text_document"
ST="${MTNLG_HOME}/Stories_fuzzy_deduped_english_all_text_document"
BIGSC="${MTNLG_HOME}/BigScience_fuzzy_deduped_english_all_text_document"
REDDIT="${MTNLG_HOME}/Reddit+_fuzzy_deduped_english_all_text_document"
SEC="${MTNLG_HOME}/SEC_fuzzy_deduped_english_all_text_document"
CCNEWS="${MTNLG_HOME}/CC-NEWS_fuzzy_deduped_english_all_text_document"

PCC="${CC_HOME}/Pile-CC_hq_text_document"
MC4="${CC_HOME}/mC4_hq_text_document"
MTNLGCCMAIN202104="${CC_HOME}/MTNLG-CC-MAIN-2021-04_hq_text_document"

CC201730="${CC_HOME}/CC-MAIN-2017-30_hq_text_document"
CC201830_0="${CC_HOME}/CC-MAIN-2018-30-00_hq_text_document"
CC201830_1="${CC_HOME}/CC-MAIN-2018-30-01_hq_text_document"
CC201935="${CC_HOME}/CC-MAIN-2019-35_hq_text_document"
CC202029="${CC_HOME}/CC-MAIN-2020-29_hq_text_document"
CC202050="${CC_HOME}/CC-MAIN-2020-50_hq_text_document"
CC202131="${CC_HOME}/CC-MAIN-2021-31_hq_text_document"
CC202233="${CC_HOME}/CC-MAIN-2022-33_hq_text_document"
CC202240_0="${CC_HOME}/CC-MAIN-2022-40-00_hq_text_document"
CC202240_1="${CC_HOME}/CC-MAIN-2022-40-01_hq_text_document"
CC202314="${CC_HOME}/CC-MAIN-2023-14_hq_text_document"


## NEW CC
CC202323="${CC_HOME}/CC-MAIN-2023-23_hq_shuf_text_document"
CC202306="${CC_HOME}/CC-MAIN-2023-06_hq_shuf_text_document"
CC202249="${CC_HOME}/CC-MAIN-2022-49_hq_shuf_text_document"
CC202227="${CC_HOME}/CC-MAIN-2022-27_hq_shuf_text_document"
CC202221="${CC_HOME}/CC-MAIN-2022-21_hq_shuf_text_document"
CC202205="${CC_HOME}/CC-MAIN-2022-05_hq_shuf_text_document"
CC202149="${CC_HOME}/CC-MAIN-2021-49_hq_shuf_text_document"
CC202143="${CC_HOME}/CC-MAIN-2021-43_hq_shuf_text_document"
CC202139="${CC_HOME}/CC-MAIN-2021-39_hq_shuf_text_document"
CC202125="${CC_HOME}/CC-MAIN-2021-25_hq_shuf_text_document"
CC202121="${CC_HOME}/CC-MAIN-2021-21_hq_shuf_text_document"
CC202117="${CC_HOME}/CC-MAIN-2021-17_hq_shuf_text_document"
CC202110="${CC_HOME}/CC-MAIN-2021-10_hq_shuf_text_document"
CC202104="${CC_HOME}/CC-MAIN-2021-04_hq_shuf_text_document"
CC202045="${CC_HOME}/CC-MAIN-2020-45_hq_shuf_text_document"
CC202040="${CC_HOME}/CC-MAIN-2020-40_hq_shuf_text_document"
CC202034="${CC_HOME}/CC-MAIN-2020-34_hq_shuf_text_document"
CC202024="${CC_HOME}/CC-MAIN-2020-24_hq_shuf_text_document"
CC202016="${CC_HOME}/CC-MAIN-2020-16_hq_shuf_text_document"
CC202010="${CC_HOME}/CC-MAIN-2020-10_hq_shuf_text_document"
CC202005="${CC_HOME}/CC-MAIN-2020-05_hq_shuf_text_document"
CC201951="${CC_HOME}/CC-MAIN-2019-51_hq_shuf_text_document"
CC201947="${CC_HOME}/CC-MAIN-2019-47_hq_shuf_text_document"
CC201943="${CC_HOME}/CC-MAIN-2019-43_hq_shuf_text_document"
CC201939="${CC_HOME}/CC-MAIN-2019-39_hq_shuf_text_document"
CC201930="${CC_HOME}/CC-MAIN-2019-30_hq_shuf_text_document"
CC201926="${CC_HOME}/CC-MAIN-2019-26_hq_shuf_text_document"
CC201922="${CC_HOME}/CC-MAIN-2019-22_hq_shuf_text_document"
CC201918="${CC_HOME}/CC-MAIN-2019-18_hq_shuf_text_document"
CC201913="${CC_HOME}/CC-MAIN-2019-13_hq_shuf_text_document"
CC201909="${CC_HOME}/CC-MAIN-2019-09_hq_shuf_text_document"
CC201904="${CC_HOME}/CC-MAIN-2019-04_hq_shuf_text_document"
CC201851="${CC_HOME}/CC-MAIN-2018-51_hq_shuf_text_document"
CC201847="${CC_HOME}/CC-MAIN-2018-47_hq_shuf_text_document"
CC201843="${CC_HOME}/CC-MAIN-2018-43_hq_shuf_text_document"
CC201839="${CC_HOME}/CC-MAIN-2018-39_hq_shuf_text_document"
CC201834="${CC_HOME}/CC-MAIN-2018-34_hq_shuf_text_document"
CC201826="${CC_HOME}/CC-MAIN-2018-26_hq_shuf_text_document"
CC201822="${CC_HOME}/CC-MAIN-2018-22_hq_shuf_text_document"
CC201817="${CC_HOME}/CC-MAIN-2018-17_hq_shuf_text_document"
CC201813="${CC_HOME}/CC-MAIN-2018-13_hq_shuf_text_document"
CC201809="${CC_HOME}/CC-MAIN-2018-09_hq_shuf_text_document"
CC201805="${CC_HOME}/CC-MAIN-2018-05_hq_shuf_text_document"
CC201751="${CC_HOME}/CC-MAIN-2017-51_hq_shuf_text_document"
CC201747="${CC_HOME}/CC-MAIN-2017-47_hq_shuf_text_document"
CC201743="${CC_HOME}/CC-MAIN-2017-43_hq_shuf_text_document"
CC201739="${CC_HOME}/CC-MAIN-2017-39_hq_shuf_text_document"
CC201734="${CC_HOME}/CC-MAIN-2017-34_hq_shuf_text_document"
CC201726="${CC_HOME}/CC-MAIN-2017-26_hq_shuf_text_document"
CC201722="${CC_HOME}/CC-MAIN-2017-22_hq_shuf_text_document"
CC201717="${CC_HOME}/CC-MAIN-2017-17_hq_shuf_text_document"
CC201713="${CC_HOME}/CC-MAIN-2017-13_hq_shuf_text_document"
CC201709="${CC_HOME}/CC-MAIN-2017-09_hq_shuf_text_document"
CC201704="${CC_HOME}/CC-MAIN-2017-04_hq_shuf_text_document"
CC201650="${CC_HOME}/CC-MAIN-2016-50_hq_shuf_text_document"
CC201644="${CC_HOME}/CC-MAIN-2016-44_hq_shuf_text_document"
CC201640="${CC_HOME}/CC-MAIN-2016-40_hq_shuf_text_document"
CC201636="${CC_HOME}/CC-MAIN-2016-36_hq_shuf_text_document"
CC201630="${CC_HOME}/CC-MAIN-2016-30_hq_shuf_text_document"
CC201626="${CC_HOME}/CC-MAIN-2016-26_hq_shuf_text_document"
CC201622="${CC_HOME}/CC-MAIN-2016-22_hq_shuf_text_document"
CC201618="${CC_HOME}/CC-MAIN-2016-18_hq_shuf_text_document"
CC201607="${CC_HOME}/CC-MAIN-2016-07_hq_shuf_text_document"
CC201548="${CC_HOME}/CC-MAIN-2015-48_hq_shuf_text_document"
CC201540="${CC_HOME}/CC-MAIN-2015-40_hq_shuf_text_document"
CC201535="${CC_HOME}/CC-MAIN-2015-35_hq_shuf_text_document"
CC201532="${CC_HOME}/CC-MAIN-2015-32_hq_shuf_text_document"
CC201527="${CC_HOME}/CC-MAIN-2015-27_hq_shuf_text_document"
CC201522="${CC_HOME}/CC-MAIN-2015-22_hq_shuf_text_document"
CC201518="${CC_HOME}/CC-MAIN-2015-18_hq_shuf_text_document"
CC201514="${CC_HOME}/CC-MAIN-2015-14_hq_shuf_text_document"
CC201511="${CC_HOME}/CC-MAIN-2015-11_hq_shuf_text_document"
CC201506="${CC_HOME}/CC-MAIN-2015-06_hq_shuf_text_document"
CC201452="${CC_HOME}/CC-MAIN-2014-52_hq_shuf_text_document"
CC201449="${CC_HOME}/CC-MAIN-2014-49_hq_shuf_text_document"
CC201442="${CC_HOME}/CC-MAIN-2014-42_hq_shuf_text_document"
CC201441="${CC_HOME}/CC-MAIN-2014-41_hq_shuf_text_document"
CC201435="${CC_HOME}/CC-MAIN-2014-35_hq_shuf_text_document"
CC201423="${CC_HOME}/CC-MAIN-2014-23_hq_shuf_text_document"
CC201415="${CC_HOME}/CC-MAIN-2014-15_hq_shuf_text_document"
CC201410="${CC_HOME}/CC-MAIN-2014-10_hq_shuf_text_document"
CC201348="${CC_HOME}/CC-MAIN-2013-48_hq_shuf_text_document"
CC201320="${CC_HOME}/CC-MAIN-2013-20_hq_shuf_text_document"




# multilingual datasets
AR2240="${NON_ENGLISH_HOME}/AR_shuf_text_document"
AZ2240="${NON_ENGLISH_HOME}/AZ_shuf_text_document"
BG2240="${NON_ENGLISH_HOME}/BG_shuf_text_document"
BN2240="${NON_ENGLISH_HOME}/BN_shuf_text_document"
CA2240="${NON_ENGLISH_HOME}/CA_shuf_text_document"
CS2240="${NON_ENGLISH_HOME}/CS_shuf_text_document"
DA2240="${NON_ENGLISH_HOME}/DA_shuf_text_document"
DE2240="${NON_ENGLISH_HOME}/DE_shuf_text_document"
EL2240="${NON_ENGLISH_HOME}/EL_shuf_text_document"
ES2240="${NON_ENGLISH_HOME}/ES_shuf_text_document"
ET2240="${NON_ENGLISH_HOME}/ET_shuf_text_document"
FA2240="${NON_ENGLISH_HOME}/FA_shuf_text_document"
FI2240="${NON_ENGLISH_HOME}/FI_shuf_text_document"
FR2240="${NON_ENGLISH_HOME}/FR_shuf_text_document"
GL2240="${NON_ENGLISH_HOME}/GL_shuf_text_document"
HE2240="${NON_ENGLISH_HOME}/HE_shuf_text_document"
HI2240="${NON_ENGLISH_HOME}/HI_shuf_text_document"
HR2240="${NON_ENGLISH_HOME}/HR_shuf_text_document"
HU2240="${NON_ENGLISH_HOME}/HU_shuf_text_document"
HY2240="${NON_ENGLISH_HOME}/HY_shuf_text_document"
ID2240="${NON_ENGLISH_HOME}/ID_shuf_text_document"
IS2240="${NON_ENGLISH_HOME}/IS_shuf_text_document"
IT2240="${NON_ENGLISH_HOME}/IT_shuf_text_document"
JAMC4="${NON_ENGLISH_HOME}/JA_shuf_text_document"
KA2240="${NON_ENGLISH_HOME}/KA_shuf_text_document"
KK2240="${NON_ENGLISH_HOME}/KK_shuf_text_document"
KN2240="${NON_ENGLISH_HOME}/KN_shuf_text_document"
KO2240="${NON_ENGLISH_HOME}/KO_shuf_text_document"
LT2240="${NON_ENGLISH_HOME}/LT_shuf_text_document"
LV2240="${NON_ENGLISH_HOME}/LV_shuf_text_document"
MK2240="${NON_ENGLISH_HOME}/MK_shuf_text_document"
ML2240="${NON_ENGLISH_HOME}/ML_shuf_text_document"
MR2240="${NON_ENGLISH_HOME}/MR_shuf_text_document"
NE2240="${NON_ENGLISH_HOME}/NE_shuf_text_document"
NL2240="${NON_ENGLISH_HOME}/NL_shuf_text_document"
NMT="${NON_ENGLISH_HOME}/nmt_shuf_text_document"
NO2240="${NON_ENGLISH_HOME}/NO_shuf_text_document"
PL2240="${NON_ENGLISH_HOME}/PL_shuf_text_document"
PT2240="${NON_ENGLISH_HOME}/PT_shuf_text_document"
RO2240="${NON_ENGLISH_HOME}/RO_shuf_text_document"
RU2240="${NON_ENGLISH_HOME}/RU_shuf_text_document"
SK2240="${NON_ENGLISH_HOME}/SK_shuf_text_document"
SL2240="${NON_ENGLISH_HOME}/SL_shuf_text_document"
SQ2240="${NON_ENGLISH_HOME}/SQ_shuf_text_document"
SR2240="${NON_ENGLISH_HOME}/SR_shuf_text_document"
SV2240="${NON_ENGLISH_HOME}/SV_shuf_text_document"
TA2240="${NON_ENGLISH_HOME}/TA_shuf_text_document"
TE2240="${NON_ENGLISH_HOME}/TE_shuf_text_document"
TH2240="${NON_ENGLISH_HOME}/TH_shuf_text_document"
TR2240="${NON_ENGLISH_HOME}/TR_shuf_text_document"
UK2240="${NON_ENGLISH_HOME}/UK_shuf_text_document"
UR2240="${NON_ENGLISH_HOME}/UR_shuf_text_document"
VI2240="${NON_ENGLISH_HOME}/VI_shuf_text_document"
ZHMC4="${NON_ENGLISH_HOME}/ZH_shuf_text_document"

#code datasets
ASMB="${STARCODER_V2_HOME}/Assembly_content_document"
CPLA="${STARCODER_V2_HOME}/C_content_document"
CSHA="${STARCODER_V2_HOME}/C#_content_document"
CLIS="${STARCODER_V2_HOME}/Common_Lisp_content_document"
CPPP="${STARCODER_V2_HOME}/C++_content_document"
CSSL="${STARCODER_V2_HOME}/CSS_content_document"
CUDA="${STARCODER_V2_HOME}/Cuda_content_document"
DART="${STARCODER_V2_HOME}/Dart_content_document"
DOCK="${STARCODER_V2_HOME}/Dockerfile_content_document"
FORT="${STARCODER_V2_HOME}/Fortran_content_document"
GOPL="${STARCODER_V2_HOME}/Go_content_document"
HASK="${STARCODER_V2_HOME}/Haskell_content_document"
HTML="${STARCODER_V2_HOME}/HTML_content_document"
JAVA="${STARCODER_V2_HOME}/Java_content_document"
JASC="${STARCODER_V2_HOME}/JavaScript_content_document"
JSON="${STARCODER_V2_HOME}/JSON_content_document"
JULI="${STARCODER_V2_HOME}/Julia_content_document"
JUPY="${STARCODER_V1_HOME}/jupyter-scripts-dedup-filtered_repo_shuf_text_document"
LUAL="${STARCODER_V2_HOME}/Lua_content_document"
MAKE="${STARCODER_V2_HOME}/Makefile_content_document"
MARD="${STARCODER_V2_HOME}/Markdown_content_document"
MATH="${STARCODER_V2_HOME}/Mathematica_content_document"
OMNI="${STARCODER_V1_HOME}/python_merged_piiremoval_text_document"
PASC="${STARCODER_V2_HOME}/Pascal_content_document"
PERL="${STARCODER_V2_HOME}/Perl_content_document"
PHPL="${STARCODER_V2_HOME}/PHP_content_document"
PYTH="${STARCODER_V2_HOME}/Python_content_document"
R="${STARCODER_V2_HOME}/R_content_document"
RSTL="${STARCODER_V2_HOME}/reStructuredText_content_document"
RUBY="${STARCODER_V2_HOME}/Ruby_content_document"
RUST="${STARCODER_V2_HOME}/Rust_content_document"
SCAL="${STARCODER_V2_HOME}/Scala_content_document"
SHEL="${STARCODER_V2_HOME}/Shell_content_document"
SQLP="${STARCODER_V2_HOME}/SQL_content_document"
SWIF="${STARCODER_V2_HOME}/Swift_content_document"
SYSV="${STARCODER_V2_HOME}/SystemVerilog_content_document"
TEXP="${STARCODER_V2_HOME}/TeX_content_document"
TYPE="${STARCODER_V2_HOME}/TypeScript_content_document"
VERI="${STARCODER_V2_HOME}/Verilog_content_document"
VHDL="${STARCODER_V2_HOME}/VHDL_content_document"
VISU="${STARCODER_V2_HOME}/Visual_Basic_.NET_content_document"
XMLL="${STARCODER_V2_HOME}/XML_content_document"
YAML="${STARCODER_V2_HOME}/YAML_content_document"

DATA_BLEND="0.0047673653 ${AR2240} \
0.0003258308 ${AZ2240} \
0.0024053767 ${BG2240} \
0.0007052695 ${BN2240} \
0.0013478762 ${CA2240} \
0.0048540743 ${CS2240} \
0.0024715232 ${DA2240} \
0.0058274094 ${DE2240} \
0.0049157318 ${EL2240} \
0.0058274094 ${ES2240} \
0.0008259057 ${ET2240} \
0.0043978654 ${FA2240} \
0.0026218056 ${FI2240} \
0.0058274094 ${FR2240} \
0.0001762378 ${GL2240} \
0.0014760127 ${HE2240} \
0.0024991501 ${HI2240} \
0.0016713234 ${HR2240} \
0.0041439852 ${HU2240} \
0.0002718320 ${HY2240} \
0.0058274094 ${ID2240} \
0.0002489622 ${IS2240} \
0.0058274094 ${IT2240} \
0.0003192912 ${KA2240} \
0.0003251229 ${KK2240} \
0.0001980963 ${KN2240} \
0.0035205911 ${KO2240} \
0.0011568286 ${LT2240} \
0.0006396362 ${LV2240} \
0.0002145816 ${MK2240} \
0.0002689304 ${ML2240} \
0.0002910419 ${MR2240} \
0.0002343074 ${NE2240} \
0.0058274094 ${NL2240} \
0.0029582154 ${NO2240} \
0.0058274094 ${PL2240} \
0.0058274094 ${PT2240} \
0.0048925958 ${RO2240} \
0.0058274094 ${RU2240} \
0.0016278923 ${SK2240} \
0.0009077300 ${SL2240} \
0.0004030820 ${SQ2240} \
0.0008790375 ${SR2240} \
0.0046512559 ${SV2240} \
0.0008220983 ${TA2240} \
0.0002879028 ${TE2240} \
0.0015614227 ${TH2240} \
0.0056631786 ${TR2240} \
0.0034904115 ${UK2240} \
0.0003833945 ${UR2240} \
0.0058274094 ${VI2240} \
0.0058274094 ${JAMC4} \
0.0042483146 ${ZHMC4} \
0.0058274094 ${NMT} \
0.0004459041 ${ASMB} \
0.0105436682 ${CPLA} \
0.0087673702 ${CSHA} \
0.0000725140 ${CLIS} \
0.0101162720 ${CPPP} \
0.0021104680 ${CSSL} \
0.0000730255 ${CUDA} \
0.0003377996 ${DART} \
0.0005980918 ${DOCK} \
0.0001167468 ${FORT} \
0.0009136190 ${GOPL} \
0.0000667175 ${HASK} \
0.0091017427 ${HTML} \
0.0100319700 ${JAVA} \
0.0210871758 ${JASC} \
0.0009073651 ${JSON} \
0.0000276047 ${JULI} \
0.0003129019 ${JUPY} \
0.0006069022 ${LUAL} \
0.0044889844 ${MAKE} \
0.0229903001 ${MARD} \
0.0000617829 ${MATH} \
0.0000039916 ${OMNI} \
0.0001840261 ${PASC} \
0.0003425708 ${PERL} \
0.0081818860 ${PHPL} \
0.0139173366 ${PYTH} \
0.0014817065 ${R} \
0.0004538749 ${RSTL} \
0.0013542952 ${RUBY} \
0.0002113528 ${RUST} \
0.0002276800 ${SCAL} \
0.0078786823 ${SHEL} \
0.0033882050 ${SQLP} \
0.0005292320 ${SWIF} \
0.0000277598 ${SYSV} \
0.0012669446 ${TEXP} \
0.0015035209 ${TYPE} \
0.0003448138 ${VERI} \
0.0001721916 ${VHDL} \
0.0001614165 ${VISU} \
0.0010585878 ${XMLL} \
0.0035309987 ${YAML} \
0.0012363668 ${CLR} \
0.0020754325 ${GLR} \
0.0000033377 ${DOHTML} \
0.0001035107 ${NDLTD} \
0.0000690056 ${DOPDF} \
0.0026360637 ${LIBGENHF} \
0.0057635019 ${USPTO} \
0.0074756218 ${P2O} \
0.0147050765 ${LIBGEN} \
0.0022090945 ${OWM} \
0.0011869415 ${BRXV} \
0.0062800884 ${PMC} \
0.0125012575 ${B3} \
0.0056549488 ${OWT2} \
0.0108326227 ${SE} \
0.0021063731 ${PMA} \
0.0024156565 ${WIK2023} \
0.0011886097 ${GUT} \
0.0007630676 ${BC2} \
0.0001488756 ${NIH} \
0.0159003598 ${ARX2023} \
0.0026520433 ${ST} \
0.0517923240 ${BIGSC} \
0.0294216952 ${REDDIT} \
0.0466949199 ${CCNEWS} \
0.0102428173 ${SEC} \
0.0027094841 ${PCC} \
0.0276608242 ${CC201730} \
0.0134680023 ${CC201830_0} \
0.0134676233 ${CC201830_1} \
0.0213629179 ${CC201935} \
0.0137996676 ${CC202029} \
0.0169794413 ${CC202050} \
0.0031024883 ${CC202104} \
0.0207855627 ${CC202131} \
0.0204753132 ${CC202233} \
0.0171208895 ${CC202240_0} \
0.0171212227 ${CC202240_1} \
0.0201788023 ${CC202314} \
0.0235726487 ${MC4} \
0.0056953013 ${MTNLGCCMAIN202104} \
0.0151456161 ${CC202323} \
0.0078128979 ${CC202306} \
0.0081344803 ${CC202249} \
0.0072611614 ${CC202227} \
0.0071321184 ${CC202221} \
0.0055336178 ${CC202205} \
0.0051875064 ${CC202149} \
0.0055235805 ${CC202143} \
0.0052638149 ${CC202139} \
0.0042341841 ${CC202125} \
0.0042099179 ${CC202121} \
0.0048831621 ${CC202117} \
0.0034849876 ${CC202110} \
0.0036374166 ${CC202045} \
0.0039604850 ${CC202040} \
0.0028070991 ${CC202034} \
0.0036904659 ${CC202024} \
0.0032427812 ${CC202016} \
0.0030235581 ${CC202010} \
0.0028064238 ${CC202005} \
0.0017702678 ${CC201951} \
0.0023450382 ${CC201947} \
0.0025475749 ${CC201943} \
0.0022658751 ${CC201939} \
0.0026323152 ${CC201930} \
0.0022467158 ${CC201926} \
0.0024491039 ${CC201922} \
0.0023287280 ${CC201918} \
0.0020986675 ${CC201913} \
0.0026494172 ${CC201909} \
0.0031132685 ${CC201904} \
0.0038802931 ${CC201851} \
0.0032856621 ${CC201847} \
0.0031745528 ${CC201843} \
0.0020437128 ${CC201839} \
0.0026656467 ${CC201834} \
0.0040433182 ${CC201826} \
0.0034291541 ${CC201822} \
0.0030180346 ${CC201817} \
0.0042486887 ${CC201813} \
0.0055794995 ${CC201809} \
0.0055977726 ${CC201805} \
0.0018164272 ${CC201751} \
0.0016077672 ${CC201747} \
0.0032926332 ${CC201743} \
0.0018939656 ${CC201739} \
0.0019311542 ${CC201734} \
0.0018719664 ${CC201726} \
0.0022063933 ${CC201722} \
0.0018422081 ${CC201717} \
0.0016189259 ${CC201713} \
0.0013018193 ${CC201709} \
0.0013198558 ${CC201704} \
0.0010553561 ${CC201650} \
0.0014861927 ${CC201644} \
0.0007330596 ${CC201640} \
0.0005642192 ${CC201636} \
0.0006629334 ${CC201630} \
0.0003073195 ${CC201626} \
0.0005027372 ${CC201622} \
0.0004485573 ${CC201618} \
0.0013280815 ${CC201607} \
0.0007837150 ${CC201548} \
0.0003749385 ${CC201540} \
0.0005643464 ${CC201535} \
0.0005152844 ${CC201532} \
0.0004338532 ${CC201527} \
0.0006628744 ${CC201522} \
0.0007367593 ${CC201518} \
0.0004467464 ${CC201514} \
0.0005505624 ${CC201511} \
0.0005191697 ${CC201506} \
0.0006563720 ${CC201452} \
0.0004823729 ${CC201449} \
0.0011769118 ${CC201442} \
0.0010736869 ${CC201441} \
0.0010654126 ${CC201435} \
0.0018642670 ${CC201423} \
0.0017903601 ${CC201415} \
0.0017143821 ${CC201410} \
0.0021849669 ${CC201348} \
0.0066630607 ${CC201320}"

# >>>
DATA_BLEND=" \
0.0125012575 ${B3} \
0.0056549488 ${OWT2} \
0.0108326227 ${SE} \
0.0021063731 ${PMA} \
0.0024156565 ${WIK2023} \
0.0011886097 ${GUT} \
0.0007630676 ${BC2} \
0.0001488756 ${NIH} \
0.0159003598 ${ARX2023} \
0.0026520433 ${ST} \
0.0517923240 ${BIGSC} \
0.0294216952 ${REDDIT} \
0.0466949199 ${CCNEWS} \
"
# <<<
