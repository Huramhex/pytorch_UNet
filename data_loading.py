import os.path
from os import listdir
from os.path import splitext

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

transform = transforms.ToTensor()


class MyDataset(Dataset):
    def __init__(self, images_path, masks_path, scale = 1):
        self.images_path = images_path
        self.masks_path = masks_path
        self.scale = scale
        self.name = [splitext(file)[0] for file in listdir(masks_path) if not file.startswith('.')]
        self.images_suffix = str(splitext(listdir(images_path)[0])[-1])
        self.masks_suffix = str(splitext(listdir(masks_path)[0])[-1])
        if not self.name:
            raise RuntimeError(f'No input file found in {images_path}, make sure you put your images there')

    def __len__(self):
        return len(self.name)

    def keep_size_same(self, file, is_mask):
        img = Image.open(file)
        size = self.size
        max_size = max(img.size)
        new_img = Image.new('RGB', (max_size, max_size), (0, 0, 0))
        new_img.paste(img, (0, 0))
        new_img = new_img.resize(size, resample=Image.NEAREST if is_mask else Image.BICUBIC)
        return new_img

    @staticmethod
    def preprocess(pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img_ndarray = np.asarray(pil_img)

        if not is_mask:
            if img_ndarray.ndim == 2:
                img_ndarray = img_ndarray[np.newaxis, ...]
            else:
                img_ndarray = img_ndarray.transpose((2, 0, 1))

            img_ndarray = img_ndarray / 255  # 归一化

        return img_ndarray


    def __getitem__(self, index):
        name = self.name[index]
        images_suffix = self.images_suffix
        masks_suffix = self.masks_suffix

        images_file = os.path.join(self.images_path, name + images_suffix)
        masks_file = os.path.join(self.masks_path, name + masks_suffix)

        # images = self.keep_size_same(images_file, is_mask=False)
        # masks_image = self.keep_size_same(masks_file, is_mask=True)
        #
        # return {'image': transform(images), 'mask': transform(masks_image)}
        mask = Image.open(masks_file)
        img = Image.open(images_file)

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(img, self.scale, is_mask=False)
        mask = self.preprocess(mask, self.scale, is_mask=True)


        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


if __name__ == '__main__':
    images_path = r'E:\PycharmProjects\pythonProject\unet-nested-multiple-classification-master\data\images'
    masks_path = r'E:\PycharmProjects\pythonProject\unet-nested-multiple-classification-master\data\masks'
    dataset = MyDataset(images_path, masks_path)
    print(len(dataset))
