import cv2
import uuid
import json
import argparse
import multiprocessing
import os
import sys

import numpy as np
import albumentations as A

from tqdm import tqdm
from pathlib import Path
from sklearn.model_selection import train_test_split


def get_augmented_name(filename: Path, modifier: str) -> str:
    """
    Returns a simple randomized file name
    
    Args:
    filename: Path
        file path to modify
    modifier: str
        custom text to inject between the file name and the random uuid
    """
    identifier = str(uuid.uuid4()).split('-')[0]
    return f'{filename.stem}_{modifier}_{identifier}{filename.suffix}'


def resize_and_crop(resize=256, crop_size=(224, 224), random=False):
    """
    Returns a function(x) handle to resize and crop an image
    
    Args:
    resize: int
        size, in pixels, of the shorter side of an image to be resized
    crop_size: tuple(int, int)
        size of the central crop to be taken in pixels
    random: bool
        whether to take a random crop (useful for training) 
        or the central crop (useful for test/validation)
    """
    composition = [A.SmallestMaxSize(max_size=resize)]
    if crop_size:
        if random:
            composition.append(A.RandomCrop(height=crop_size[0], width=crop_size[1]))
        else:
            composition.append(A.CenterCrop(height=crop_size[0], width=crop_size[1]))
    return A.Compose(composition, p=1)


def strong_aug(p=1):
    """
    Returns a strong augmentation composition
    
    Args:
    p: float
        probability of executing the augmentation
    shape: int
        max_shape length for the image resize
    """
    return A.Compose([
            A.ImageCompression(80, 100),
            A.MotionBlur(),
            A.RandomBrightnessContrast(),
            A.HorizontalFlip(),
            A.Affine(scale=(0.9,1.1), rotate=(-10,10), shear=(-10,10)),
            A.HueSaturationValue()
        ], 
        p=p)




class ImageAugmentationProcessor:
    """
    Augmentation class. It takes SageMaker ground truth lines and generates a copy of the orinnal image, 
    plus a set of augmented versions.
    
    Args:
    input_dir: Path
        local path where images are stored
    output_dir: Path
        local path to store the augmented images
    num_augmentations: int
        number of augmentations to apply to each image
    resize: int
        size, in pixels, of the shorter side of an image to be resized
    crop_size: tuple(int, int)
        size of the central crop to be taken in pixels
    """
    def __init__(self, 
                 output_dir,  
                 num_augmentations=1, 
                 resize=256, 
                 crop_shape=(224, 224), 
                 ):
        self.output_dir = output_dir
        self.num_augmentations = num_augmentations
        self.resize = resize
        self.crop_shape = crop_shape
        output_dir.mkdir(exist_ok=True, parents=True)

    def __call__(self, file):
        
        # This forces TQDM to print in new lines, 
        # which is nice for CloudWatch. Disable if annoying.
        # print(f'Processing {filename}.') 
        
        label = file.parent.name
        img = cv2.imread(str(file))
        
        output_image_dir = self.output_dir/label
        output_image_dir.mkdir(exist_ok=True, parents=True)
        output_image_path = output_image_dir/file.name
        
        out_img = resize_and_crop(self.resize, self.crop_shape, random=False)(image=img)['image']
        cv2.imwrite(str(output_image_path), out_img)
        
        # robust augmentation
        for i in range(self.num_augmentations):
            out_img = strong_aug()(image=img)['image']
            out_img = resize_and_crop(self.resize, self.crop_shape, random=True)(image=out_img)['image']
            
            output_name = get_augmented_name(file, 'augmented')
            output_image_path = output_image_dir/output_name
            cv2.imwrite(str(output_image_path), out_img)
    
    
def main():
    parser = argparse.ArgumentParser(description='create new dataset')

    parser.add_argument('--input', type=str,default="/opt/ml/processing/input/", help="this is where all images are expected to be")
    parser.add_argument('--output', type=str, default="/opt/ml/processing/output/", help="this is the output location for /train, /val, and /test datasets")
    
    parser.add_argument('--resize', type=int, default=256, help="the maximal length of a resized image before being cropped")
    parser.add_argument('--crop-shape', type=int, default=224, help="the maximal length of a cropped image fitting the input layer. A value of 0 means no crop is done.")
    parser.add_argument('--num-augmentations', type=int, default=10, help="the number of random augmentations to create within the dataset")
    parser.add_argument('--val-split', type=float, default=0.2, help="fraction of dataset to hold out as validation")
    parser.add_argument('--min-sample-count', type=int, default=10, help="minimum number of samples per class to include.")
    
    args = parser.parse_args()
    
    # Folder structure creation
    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(exist_ok=True)
    
    lines = list(input_dir.glob('*/*.jpg'))
    class_labels = [f.parent.name for f in lines]
    
    print(f'Filtering classes with less than {args.min_sample_count} samples')
    sample_count = {c:class_labels.count(c) for c in class_labels}
    lines = list(filter(lambda x: sample_count[x.parent.name]>=args.min_sample_count, lines))
    class_labels = [f.parent.name for f in lines]
    
    # train/val/test split
    val_size = args.val_split
    x_train, x_val = train_test_split(lines, test_size=val_size, stratify=class_labels)
    
    val_class_labels = [f.parent.name for f in x_val]
    x_val, x_test = train_test_split(x_val, test_size=0.5, stratify=val_class_labels)
    
    image_sets = {'train': x_train, 'val': x_val, 'test': x_test}
    
    mp = multiprocessing.cpu_count() > 1
    if mp:
        print('Using multiprocessing.')
    else:
        print('Use single thread processing')
    
    # train and val set generation
    for image_set in ['test', 'val', 'train']:
        
        print(f'Processing {len(image_sets[image_set])} {image_set} images.')
        
        # Folder structure creation
        working_dir = output_dir/image_set
        working_dir.mkdir(exist_ok=True, parents=True)
        
        proc = ImageAugmentationProcessor(working_dir, 
                                          num_augmentations=args.num_augmentations if image_set=='train' else 0, 
                                          resize=args.resize,
                                          crop_shape=(args.crop_shape, args.crop_shape) if args.crop_shape else None,
                                         )
        
        if mp:
            pool = multiprocessing.Pool()
            pool.imap(proc,tqdm(image_sets[image_set],
                                position=-1, # position trick for CloudWatch logging
                                miniters=int(len(image_sets[image_set])/100)
                               ) # update every 1%)
                     )
            pool.close()
            pool.join()
        else:
            for i in tqdm(image_sets[image_set],
                          position=-1, # position trick for CloudWatch logging
                          miniters=int(len(image_sets[image_set])/100)
                         ): # update every 1%))):
                proc(i)
            
        print(f'Generating {image_set} done.')
        
    print(f'Finished.')

    
if __name__ == '__main__':
    main()