from torch.utils.data import Dataset
import os, cv2
import numpy as np
import torch

class PolypDatasetLoader(Dataset):
    def __init__(self, image_dir, mask_dir, input_image_size=512):
        self.input_image_size = input_image_size
        self.dataset = self.__make_dataset(image_dir, mask_dir)
    

    ## Write code for augmenting data to increase the dataset size.
    def __augmentation(self, image, mask, rotate = False):
        ## Wtite code here and the return tpe will be a augmented image and mask tensor
        pass
        
    
    ## Write code to make a list of image paths and their respective mask paths
    def __make_dataset(self, image_dir, mask_dir):
        ## Write code here and there will be no return type
        pass
    


    def __len__(self):
        return len(self.dataset)
    

    
    def __normalize_image(self, image, mean=[0,0,0], std=[1,1,1]):
        for channel in range(3):
            image[:,:,channel] = (image[:,:,channel] - mean[channel]) / std[channel]
        return image
    

    ## write code here to read the image and mask, apply augmentation operation on the image, convert them to pytorch tensor and return the augmented image and the mask
    def __getitem__(self, index):
        ## Write code here and return type will be a image and mask tensor
        pass