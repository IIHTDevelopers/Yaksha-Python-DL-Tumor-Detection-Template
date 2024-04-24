from torch.utils.data import Dataset
import os, cv2
import numpy as np
import torch

class PolypDatasetLoader(Dataset):
    def __init__(self, image_dir, mask_dir, input_image_size=512):
        self.input_image_size = input_image_size
        self.dataset = self.__make_dataset(image_dir, mask_dir)
    

    def __augmentation(self, image, mask, rotate = False):
        rows_x,cols_x, _ = image.shape
        rows_y,cols_y = mask.shape

        if rotate:
            rand_num = np.random.randint(-40,40)
            M1 = cv2.getRotationMatrix2D((cols_x/2,rows_x/2), rand_num,1)
            M2 = cv2.getRotationMatrix2D((cols_y/2,rows_y/2), rand_num,1)
            image1 = cv2.warpAffine(image,M1,(cols_x,rows_x))
            mask1 = cv2.warpAffine(mask.astype('float32'),M2,(cols_y,rows_y))
            return (np.array(image1), np.array(mask1.astype('int')))
        
        else:
            image2 = cv2.flip(image, 1)
            mask2 = cv2.flip(mask.astype('float32'), 1)
            return (np.array(image2), np.array(mask2.astype('int')))
        

    def __make_dataset(self, image_dir, mask_dir):
        img_list = os.listdir(image_dir)
        img_list.sort()
        img_list_new = []
        for image in img_list:
            for choice in ["_normal", "_rotate", "_flip"]:
                img_name = image.split(".")
                img_name[0] = img_name[0] + choice
                temp = ".".join(img_name)
                img_list_new.append(image_dir+"/"+temp)

        # img_list = [img_dir+"\\"+image for image in img_list]
        mask_list = os.listdir(mask_dir)
        mask_list.sort()
        mask_list_new = []
        for mask in mask_list:
            for choice in ["_normal", "_rotate", "_flip"]:
                mask_name = mask.split(".")
                mask_name[0] = mask_name[0] + choice
                temp = ".".join(mask_name)
                mask_list_new.append(mask_dir+"/"+temp)
        return [(image,mask) for image,mask in zip(img_list_new, mask_list_new)]
    
    def __len__(self):
        return len(self.dataset)
    

    
    def __normalize_image(self, image, mean=[0,0,0], std=[1,1,1]):
        for channel in range(3):
            image[:,:,channel] = (image[:,:,channel] - mean[channel]) / std[channel]
        return image
    

    def __getitem__(self, index):
        image_path, mask_path = self.dataset[index]

        # image name
        img = image_path.split(".")
        name = img[0].split("_")
        operation = name[-1]
        img[0] = "_".join(name[:-1]).strip("_")
        image_path = ".".join(img)
        # mask name
        mask = mask_path.split(".")
        name = mask[0].split("_")
        mask[0] = "_".join(name[:-1]).strip("_")
        mask_path = ".".join(mask)

        # print(image_path)
        try:
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if image.shape[0] != mask.shape[0] or image.shape[1] != mask.shape[1]:
                raise (RuntimeError("Image & Mask shape mismatch: " + image_path + " " + mask_path + "\n"))
        except:
            print(image_path)
        
        # resizing image ans mask
        image = cv2.resize(image, (self.input_image_size,
        self.input_image_size), interpolation = cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.input_image_size,
        self.input_image_size), interpolation = cv2.INTER_NEAREST)
        # binary mask
        mask = np.float32(mask>128)

        #normalizing data
        image = self.__normalize_image(np.array(image) / 255.0, 
                            mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
        
        # getting augmentation on choice
        
        if operation == "rotate":
            image, mask = self.__augmentation(image, mask)
        elif operation == "flip":
            image, mask = self.__augmentation(image, mask, rotate=False)

        mask=np.reshape(mask,(1, self.input_image_size, self.input_image_size))

        # converting to pytorch tensor
        image = torch.from_numpy(image).movedim(2,0)
        image = image.type(torch.FloatTensor)
        mask = torch.from_numpy(mask)
        mask = mask.type(torch.FloatTensor)

        return image, mask

