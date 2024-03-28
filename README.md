# U-net architecture implementation for MRI image segmentation
Research Paper: https://arxiv.org/pdf/1505.04597.pdf
![U-Net Architecture](/images/unet.png)

## Dataset
https://huggingface.co/datasets/rainerberger/Mri_segmentation


## How to run
- Run ```python data.py``` to download the MRI dataset from hugging face into dataset directory
- Run ```python train.py``` to train the model and predict the image segmentations 
