
import cortex
import nibabel as nib

# ginkgo '/scratch/ibilgin/encoder_venv/share/pycortex/db'
# '/home/ibilgin/.config/pycortex/options.cfg'
from save_3d_views import save_3d_views

# Load NIfTI file
nii_file = '/scratch/ibilgin/Dropbox/language_encoder/data/ridge_regression/sub-03/experiment_within/s01/val_mean_season_s02_layer1.nii.gz'
nii_data = nib.load(nii_file)
data = nii_data.get_fdata()


anat_vol = nib.load('path_to_anatomical_volume.nii.gz').get_fdata()
func_vol = data
cortex.mni.transform.write_linear_xfm('my_subject', 'my_transform', anat_vol, func_vol)

# Load the transform
subject = 's03'
xfm = cortex.xfm.Transform.from_fsl(subject, transform_file)



# Assuming you have a subject and transform for your data
xfm = 'your_transform'

# Create pycortex volume
volume = cortex.Volume(data, subject, xfm)

# Use the save_3d_views function with the created volume
root = 'output_directory'
base_name = 'base_image_name'
save_3d_views(volume, root, base_name)
