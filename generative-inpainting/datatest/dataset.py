import sys
#import torch.utils.data as data
from os import listdir
from utils.tools import default_loader, is_image_file, normalize
import os
import numpy as np
import paddle

#import torchvision.transforms as transforms

import paddle.vision.transforms as transforms

from paddle.io import Dataset


class Dataset(Dataset):
    def __init__(self, data_path, image_shape, with_subfolder=False, random_crop=True, return_name=False):
        super(Dataset, self).__init__()
        if with_subfolder:
            self.samples = self._find_samples_in_subfolders(data_path)
        else:
            self.samples = [x for x in listdir(data_path) if is_image_file(x)]
        self.data_path = data_path
        self.image_shape = image_shape[:-1]
        self.random_crop = random_crop
        self.return_name = return_name

    def __getitem__(self, index):

        path = os.path.join(self.data_path, self.samples[index])
        img = default_loader(path)

        if self.random_crop:
            imgw, imgh = img.size
            if imgh < self.image_shape[0] or imgw < self.image_shape[1]:
                img = transforms.Resize(min(self.image_shape))(img)
            img = transforms.RandomCrop(self.image_shape)(img)
        else:
            img = transforms.Resize(self.image_shape)(img)
            img = transforms.RandomCrop(self.image_shape)(img)

        img = transforms.ToTensor()(img)  # turn the image to a tensor

        #img = img.numpy()
        #img=np.around(img,4)
        #img=paddle.to_tensor(img)
        
        img = normalize(img)
        img=img.numpy()

        if self.return_name:
            return self.samples[index], img
        else:
            return img

    def _find_samples_in_subfolders(self, dir):
        """
        Finds the class folders in a dataset.
        Args:
            dir (string): Root directory path.
        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.
        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            #classes = [d.name for d in os.scandir(dir) if d.is_dir()]
            classes = [d for d in os.listdir(dir)]
        else:
            classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        #class_to_idx = {classes[i]: i for i in range(len(classes))}
        samples = []
        #for target in sorted(class_to_idx.keys()):
        for target in sorted(classes):
            d = os.path.join(dir, target)
            samples.append(d)
        return samples

    def __len__(self):
        return len(self.samples)
