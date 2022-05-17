import torch.nn.functional as F
from matplotlib import pyplot as plt

from HuramUNet.Model.Unet_model import UNet
from HuramUNet.Model.NestedUNet_model import *
from data_loading import *
import Set_Mode


def predict_img(net,
                full_img,
                device,
                scale_factor=1):
    net.eval()
    img = torch.from_numpy(MyDataset.preprocess(full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)

        if stm.n_classes > 1:
            probs = F.softmax(output, dim=1)[0]
        else:
            probs = torch.sigmoid(output)[0]

        tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((full_img.size[1], full_img.size[0])),
            transforms.ToTensor()
        ])

    if stm.n_classes == 1:
        probs = tf(probs.cpu())
        mask = probs.squeeze().cpu().numpy()
        return mask > stm.rate_threshold
    else:
        masks = []
        for prob in probs:
            prob = tf(prob.cpu())
            mask = prob.squeeze().cpu().numpy()
            mask = mask > stm.rate_threshold
            masks.append(mask)
        return masks


def mask_to_image(mask: np.ndarray):
    if stm.n_classes == 1:
        image_idx = Image.fromarray((mask * 255).astype(np.uint8))


    else:
        for idx in range(0, len(mask)):
            image_idx = Image.fromarray((mask[idx] * 255).astype(np.uint8))
    return image_idx


if __name__ == '__main__':

    stm = Set_Mode.set_param()
    net = UNet(stm)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net.to(device=device)
    Model_file = r'E:\PycharmProjects\pythonProject\HuramUNet\data\checkpoints\Best_val_Checkpoint.pth'
    if os.path.exists(Model_file):
        state_dict = torch.load(Model_file, map_location=device)
        # print(type(state_dict))
        net.load_state_dict(state_dict['state_dict'])

        print('Successful load module')

    else:
        print('No loading')

    # _input = input("please input image path:")
    input_image_path = r'E:\PycharmProjects\pythonProject\UNet\data\VOCdevkit\VOC2012\JPEGImages\2007_000032.jpg'
    mask_image_path = r'E:\PycharmProjects\pythonProject\UNet\data\VOCdevkit\VOC2012\SegmentationClass\2007_000032.png'

    input_img = Image.open(input_image_path)

    pred_img = predict_img(net=net,
                           full_img=input_img,
                           scale_factor=1,
                           device=device)

    pred_img = mask_to_image(pred_img)
    mask_img = Image.open(mask_image_path)

    plt.subplot(131)
    plt.imshow(input_img)
    plt.subplot(132)
    plt.imshow(pred_img)
    plt.subplot(133)
    plt.imshow(mask_img)
    plt.show()
