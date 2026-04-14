"""
Augmentation pipeline for ACDC cardiac MRI segmentation.
 
Applied ONLY to training set, on-the-fly per batch.
Never applied to validation or test sets.
 
Justified by:
- Rotation + flip: universal across Yang 2020, Singh 2023, Tesfaye 2023, Wijesinghe 2025
- Elastic deformation: Ronneberger 2015 (original U-Net), Tesfaye 2023
- Gaussian noise + gamma: ARW-Net Table 1 (Isensee [28]) — simulates MRI scanner variability
- Scale: Yang 2020, Singh 2023, Wijesinghe 2025
"""
 
import albumentations as A
from albumentations.pytorch import ToTensorV2
 
 
def get_train_augmentation() -> A.Compose:
    """
    Returns the training augmentation pipeline.
 
    All spatial transforms are applied consistently to both image and mask.
    Mask uses nearest-neighbour interpolation (interpolation=0) to preserve
    integer label values {0=BG, 1=RV, 2=MYO, 3=LV}.
 
    Returns
    -------
    A.Compose
        Albumentations composition with additional_targets for mask.
    """
    return A.Compose([
        # --- Spatial transforms (applied to image + mask together) ---
        A.Rotate(
            limit=15,
            interpolation=1,        # bilinear for image
            border_mode=0,          # constant padding (value=0)
            p=1.0
        ),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.0,
            scale_limit=0.10,       # ±10% scale
            rotate_limit=0,         # rotation already handled above
            interpolation=1,
            border_mode=0,
            p=0.5
        ),
        A.ElasticTransform(
            alpha=50,
            sigma=6,
            alpha_affine=0,         # no affine component — pure elastic only
            interpolation=1,
            border_mode=0,
            p=0.3
        ),
 
        # --- Intensity transforms (applied to image only, mask unchanged) ---
        A.GaussNoise(
            var_limit=(0.001, 0.005),
            mean=0.0,
            per_channel=False,
            p=0.5
        ),
        
    ],
    additional_targets={"mask": "mask"}   # ensures mask receives same spatial ops
    )
 
 
def get_val_augmentation() -> A.Compose:
    """
    Validation/test augmentation — no transforms, returns as-is.
 
    Kept as a Compose object for API consistency with the Dataset class.
    """
    return A.Compose([],
                     additional_targets={"mask": "mask"})
 