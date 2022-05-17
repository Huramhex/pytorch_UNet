import matplotlib.pyplot as plt
import matplotlib.image as mp
from PIL import Image
import cv2


def show_img(path):
    img = mp.imread(path)
    print('图片的shape:', img.shape)

    plt.imshow(img)
    plt.show()


if __name__ == '__main__':
    show_img(r'E:\PycharmProjects\pythonProject\UNet\data\VOCdevkit\VOC2012\JPEGImages\2007_000032.jpg')
