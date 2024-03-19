import datasets

# Loads the dataset from hugging face
#ds = datasets.load_dataset("rainerberger/Mri_segmentation")
#ds.save_to_disk("../datasets/huggingface/mri_segmentation_dataset")

mydata = datasets.load_from_disk("../datasets/huggingface/mri_segmentation_dataset")

print(mydata)