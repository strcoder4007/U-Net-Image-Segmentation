import os
import numpy as np

from datasets import load_dataset


image_size = 512

def load_and_save_dataset():
    # Get data from Hugging face
    master_dataset = load_dataset("rainerberger/Mri_segmentation")

    train_dataset = master_dataset['train']
    test_dataset = master_dataset['test']

    train_total = len(train_dataset)
    test_total = len(test_dataset)

    # Split the data into Training data and Test data and save

    # Processing training images 
    train_imgs = np.ndarray((train_total, image_size, image_size), dtype=np.uint8)
    train_imgs_mask = np.ndarray((train_total, image_size, image_size), dtype=np.uint8)

    for i in range(len(train_dataset)):
        img = train_dataset[i]['image']
        img_mask = train_dataset[i]['annotation']
        
        train_imgs[i] = img
        train_imgs_mask[i] = img_mask

        if i%100 == 0:
            print('Done: {0}/{1} images'.format(i, train_total))

    print('Training images load complete.')

    np.save('train_imgs.npy', train_imgs)
    np.save('train_imgs_mask.npy', train_imgs_mask)

    print('Saving to .npy files complete.')

    # Processing testing images
    test_imgs = np.ndarray((test_total, image_size, image_size), dtype=np.uint8)
    test_imgs_mask = np.ndarray((test_total, image_size, image_size), dtype=np.uint8)

    for i in range(len(test_dataset)):
        img = test_dataset[i]['image']
        img_mask = test_dataset[i]['annotation']
        
        test_imgs[i] = img
        test_imgs_mask[i] = img_mask

        if i%100 == 0:
            print('Done: {0}/{1} images'.format(i, test_total))

    print('Testing images load complete.')

    np.save('test_imgs.npy', test_imgs)
    np.save('test_imgs_mask.npy', test_imgs_mask)

    print('Saving to .npy files complete.')


    
def load_train_data():
    train_imgs = np.load('train_imgs.npy')
    train_imgs_mask = np.load('train_imgs_mask.npy')
    return train_imgs, train_imgs_mask


def load_test_data():
    test_imgs = np.load('test_imgs.npy')
    test_imgs_mask = np.load('test_imgs_mask.npy')
    return test_imgs, test_imgs_mask


if __name__ == '__main__':
    load_and_save_dataset()