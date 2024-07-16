import numpy as np
from encoder_dataclass import DataBaseConfig
from nilearn.maskers import NiftiLabelsMasker, NiftiMasker
data_config = DataBaseConfig()
import nibabel as nib
from pathlib import Path
from utils import get_within_ridge_average
# random_array = np.random.rand(104000)



# # mask_path = Path(
# #     f"{data_config.voxelwise_bold_dir}/{data_config.subject_id}/func/"
# #     f"{data_config.subject_id}_task-friends_space-T1w_atlas-Freesurfer_label-GM_res-func_mask.nii.gz") 

# filename = "sub-03_ses-001_task-s01e02a_space-MNI152NLin2009cAsym_desc-brain_mask.nii.gz"
# # atlas_img = load_img(mask_path)
# mask_path = f"/scratch/ibilgin/datasets/friends/cneuromod.friends.fmriprep/sub-03/ses-001/func/{filename}"

# masker = NiftiMasker(mask_img=mask_path, standardize=False)

# masker.fit()

# # map Pearson correlations onto brain parcels

# nii_file = masker.inverse_transform(
# np.array(random_array),
# )

# nib.save(nii_file,
#             f"{data_config.output_dir}/{data_config.encoding_level}/{data_config.subject_id}/{data_config.experiment}//random.nii.gz",
#         )

season = ["s01", "s02", "s04", "s05", "s06"]

for train in season:
    get_within_ridge_average(data_config, [train])