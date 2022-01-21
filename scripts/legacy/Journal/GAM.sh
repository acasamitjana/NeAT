srun --mem=5G --time=1:00:00 python ../../../neat-compute_statistical_maps.py /imatge/acasamitjana/Repositories/NeAT/config/Journal/adContinuum_surface_lh_ROIs.yaml --hemi left --dirs correction_PolyGLM-prediction_GAM --method aic
srun --mem=5G --time=1:00:00 python ../../../neat-compute_statistical_maps.py /imatge/acasamitjana/Repositories/NeAT/config/Journal/adContinuum_surface_rh_ROIs.yaml --hemi right --dirs correction_PolyGLM-prediction_GAM --method aic

srun --mem=5G --time=1:00:00 python ../../../neat-compute_statistical_maps.py /imatge/acasamitjana/Repositories/NeAT/config/Journal/adContinuum_surface_lh_ROIs.yaml --hemi left --dirs correction_PolyGLM-prediction_GAM --method vnprss
srun --mem=5G --time=1:00:00 python ../../../neat-compute_statistical_maps.py /imatge/acasamitjana/Repositories/NeAT/config/Journal/adContinuum_surface_rh_ROIs.yaml --hemi right --dirs correction_PolyGLM-prediction_GAM --method vnprss


srun --mem=5G --time=2:00:00 python ../../../neat-compute_fitting.py /imatge/acasamitjana/Repositories/NeAT/config/Journal/adContinuum_surface_lh_ROIs_PTAU.yaml --hemi left --parameters user_defined_parameters_gam.txt
srun --mem=5G --time=2:00:00 python ../../../neat-compute_fitting.py /imatge/acasamitjana/Repositories/NeAT/config/Journal/adContinuum_surface_rh_ROIs_PTAU.yaml --hemi right --parameters user_defined_parameters_gam.txt

srun --mem=5G --time=1:00:00 python ../../../neat-compute_statistical_maps.py /imatge/acasamitjana/Repositories/NeAT/config/Journal/adContinuum_surface_lh_ROIs_PTAU.yaml --hemi left --dirs correction_PolyGLM-prediction_GAM
srun --mem=5G --time=1:00:00 python ../../../neat-compute_statistical_maps.py /imatge/acasamitjana/Repositories/NeAT/config/Journal/adContinuum_surface_rh_ROIs_PTAU.yaml --hemi right --dirs correction_PolyGLM-prediction_GAM

srun --mem=5G --time=1:00:00 python ../../../neat-compute_statistical_maps.py /imatge/acasamitjana/Repositories/NeAT/config/Journal/adContinuum_surface_lh_ROIs_PTAU.yaml --hemi left --dirs correction_PolyGLM-prediction_GAM --method aic
srun --mem=5G --time=1:00:00 python ../../../neat-compute_statistical_maps.py /imatge/acasamitjana/Repositories/NeAT/config/Journal/adContinuum_surface_rh_ROIs_PTAU.yaml --hemi right --dirs correction_PolyGLM-prediction_GAM --method aic

srun --mem=5G --time=1:00:00 python ../../../neat-compute_statistical_maps.py /imatge/acasamitjana/Repositories/NeAT/config/Journal/adContinuum_surface_lh_ROIs_PTAU.yaml --hemi left --dirs correction_PolyGLM-prediction_GAM --method vnprss
srun --mem=5G --time=1:00:00 python ../../../neat-compute_statistical_maps.py /imatge/acasamitjana/Repositories/NeAT/config/Journal/adContinuum_surface_rh_ROIs_PTAU.yaml --hemi right --dirs correction_PolyGLM-prediction_GAM --method vnprss