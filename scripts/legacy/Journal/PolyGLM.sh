srun --mem=5G --time=2:00:00 python ../../../neat-compute_fitting.py /imatge/acasamitjana/Repositories/NeAT/config/Journal/adContinuum_surface_lh_ROIs.yaml --hemi left --parameters user_defined_parameters_glm.txt
srun --mem=5G --time=2:00:00 python ../../../neat-compute_fitting.py /imatge/acasamitjana/Repositories/NeAT/config/Journal/adContinuum_surface_rh_ROIs.yaml --hemi right --parameters user_defined_parameters_glm.txt


srun --mem=5G --time=1:00:00 python ../../../neat-compute_statistical_maps.py /imatge/acasamitjana/Repositories/NeAT/config/Journal/adContinuum_surface_lh_ROIs.yaml --hemi left --dirs correction_PolyGLM-prediction_PolyGLM
srun --mem=5G --time=1:00:00 python ../../../neat-compute_statistical_maps.py /imatge/acasamitjana/Repositories/NeAT/config/Journal/adContinuum_surface_rh_ROIs.yaml --hemi right --dirs correction_PolyGLM-prediction_PolyGLM

srun --mem=5G --time=1:00:00 python ../../../neat-compute_statistical_maps.py /imatge/acasamitjana/Repositories/NeAT/config/Journal/adContinuum_surface_lh_ROIs.yaml --hemi left --dirs correction_PolyGLM-prediction_PolyGLM --method aic
srun --mem=5G --time=1:00:00 python ../../../neat-compute_statistical_maps.py /imatge/acasamitjana/Repositories/NeAT/config/Journal/adContinuum_surface_rh_ROIs.yaml --hemi right --dirs correction_PolyGLM-prediction_PolyGLM --method aic

srun --mem=5G --time=1:00:00 python ../../../neat-compute_statistical_maps.py /imatge/acasamitjana/Repositories/NeAT/config/Journal/adContinuum_surface_lh_ROIs.yaml --hemi left --dirs correction_PolyGLM-prediction_PolyGLM --method vnprss
srun --mem=5G --time=1:00:00 python ../../../neat-compute_statistical_maps.py /imatge/acasamitjana/Repositories/NeAT/config/Journal/adContinuum_surface_rh_ROIs.yaml --hemi right --dirs correction_PolyGLM-prediction_PolyGLM --method vnprss


srun --mem=5G --time=2:00:00 python ../../../neat-compute_fitting.py /imatge/acasamitjana/Repositories/NeAT/config/Journal/adContinuum_surface_lh_ROIs_PTAU.yaml --hemi left --parameters user_defined_parameters_glm.txt
srun --mem=5G --time=2:00:00 python ../../../neat-compute_fitting.py /imatge/acasamitjana/Repositories/NeAT/config/Journal/adContinuum_surface_rh_ROIs_PTAU.yaml --hemi right --parameters user_defined_parameters_glm.txt

srun --mem=5G --time=1:00:00 python ../../../neat-compute_statistical_maps.py /imatge/acasamitjana/Repositories/NeAT/config/Journal/adContinuum_surface_lh_ROIs_PTAU.yaml --hemi left --dirs correction_PolyGLM-prediction_PolyGLM
srun --mem=5G --time=1:00:00 python ../../../neat-compute_statistical_maps.py /imatge/acasamitjana/Repositories/NeAT/config/Journal/adContinuum_surface_rh_ROIs_PTAU.yaml --hemi right --dirs correction_PolyGLM-prediction_PolyGLM

srun --mem=5G --time=1:00:00 python ../../../neat-compute_statistical_maps.py /imatge/acasamitjana/Repositories/NeAT/config/Journal/adContinuum_surface_lh_ROIs_PTAU.yaml --hemi left --dirs correction_PolyGLM-prediction_PolyGLM --method aic
srun --mem=5G --time=1:00:00 python ../../../neat-compute_statistical_maps.py /imatge/acasamitjana/Repositories/NeAT/config/Journal/adContinuum_surface_rh_ROIs_PTAU.yaml --hemi right --dirs correction_PolyGLM-prediction_PolyGLM --method aic

srun --mem=5G --time=1:00:00 python ../../../neat-compute_statistical_maps.py /imatge/acasamitjana/Repositories/NeAT/config/Journal/adContinuum_surface_lh_ROIs_PTAU.yaml --hemi left --dirs correction_PolyGLM-prediction_PolyGLM --method vnprss
srun --mem=5G --time=1:00:00 python ../../../neat-compute_statistical_maps.py /imatge/acasamitjana/Repositories/NeAT/config/Journal/adContinuum_surface_rh_ROIs_PTAU.yaml --hemi right --dirs correction_PolyGLM-prediction_PolyGLM --method vnprss

srun --mem=5G --time=1:00:00 python ../../../neat-compare_statistical_maps.py /imatge/acasamitjana/Repositories/NeAT/config/Journal/adContinuum_surface_rh_ROIs.yaml correction_PolyGLM-prediction_PolyGLM/rh-ftest_fitscores.maps correction_PolyGLM-prediction_GAM/rh-ftest_fitscores.maps correction_PolyGLM-prediction_PolySVR/rh-ftest_fitscores.maps correction_PolyGLM-prediction_GaussianSVR/rh-ftest_fitscores.maps
srun --mem=5G --time=1:00:00 python ../../../neat-compare_statistical_maps.py /imatge/acasamitjana/Repositories/NeAT/config/Journal/adContinuum_surface_lh_ROIs.yaml correction_PolyGLM-prediction_PolyGLM/lh-ftest_fitscores.maps correction_PolyGLM-prediction_GAM/lh-ftest_fitscores.maps correction_PolyGLM-prediction_PolySVR/lh-ftest_fitscores.maps correction_PolyGLM-prediction_GaussianSVR/lh-ftest_fitscores.maps --name lh_

srun --mem=5G --time=1:00:00 python ../../../neat-compare_statistical_maps.py /imatge/acasamitjana/Repositories/NeAT/config/Journal/adContinuum_surface_rh_ROIs_PTAU.yaml correction_PolyGLM-prediction_PolyGLM/rh-ftest_fitscores.maps correction_PolyGLM-prediction_GAM/rh-ftest_fitscores.maps correction_PolyGLM-prediction_PolySVR/rh-ftest_fitscores.maps correction_PolyGLM-prediction_GaussianSVR/rh-ftest_fitscores.maps --name rh_
srun --mem=5G --time=1:00:00 python ../../../neat-compare_statistical_maps.py /imatge/acasamitjana/Repositories/NeAT/config/Journal/adContinuum_surface_lh_ROIs_PTAU.yaml correction_PolyGLM-prediction_PolyGLM/lh-ftest_fitscores.maps correction_PolyGLM-prediction_GAM/lh-ftest_fitscores.maps correction_PolyGLM-prediction_PolySVR/lh-ftest_fitscores.maps correction_PolyGLM-prediction_GaussianSVR/lh-ftest_fitscores.maps --name lh_
