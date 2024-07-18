import numpy as np
from encoder_dataclass import DataBaseConfig
from nilearn.maskers import NiftiLabelsMasker, NiftiMasker
data_config = DataBaseConfig()
import nibabel as nib
from pathlib import Path
from utils import get_within_ridge_average, get_across_ridge_average
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
    # get_within_ridge_average(data_config, [train])
    get_across_ridge_average(data_config, train)




# import h5py

# def print_hdf5_structure(group, indent=0):
#     """Recursively prints the structure of an HDF5 file."""
#     spaces = ' ' * indent
#     if isinstance(group, h5py.File):
#         print(f"{spaces}File: {group.filename}")
#     else:
#         print(f"{spaces}Group: {group.name}")

#     for key in group.keys():
#         item = group[key]
#         if isinstance(item, h5py.Group):
#             print(f"{spaces}  Group: {key}")
#             print_hdf5_structure(item, indent + 4)
#         elif isinstance(item, h5py.Dataset):
#             print(f"{spaces}  Dataset: {key} - shape: {item.shape}, dtype: {item.dtype}")

# # Open the HDF5 file and print its structure
# file_path = '/scratch/ibilgin/datasets/movie10/parcelwise/movie10.timeseries/sub-01/func/sub-01_task-movie10_space-MNI152NLin2009cAsym_atlas-MIST_desc-444Cropped_timeseries.h5'
# # file_path = '/scratch/ibilgin/./datasets/friends/parcelwise/friends.timeseries/sub-01/func/sub-01_task-friends_space-MNI152NLin2009cAsym_atlas-MIST_desc-444_timeseries.h5'

# # file_path = "/scratch/ibilgin/Dropbox/cneuromax/data/friends_language_encoder/stimuli//gpt2/base/friends/friends_s02_embeddings_gpt2_base.h5"
# with h5py.File(file_path, 'r') as f:
#     print_hdf5_structure(f)

# # item = "ses-011_task-figures01_run-2_timeseries"
# # parts = item.split('_')

# # print(''.join([char for char in parts[1].split('-')[1] if not char.isdigit()]))

# # # Extract the 'run-2' part from the third component
# # print(parts[2])


# item = "ses-064_task-s06e21b_timeseries"
# print(item.split("-")[-1])