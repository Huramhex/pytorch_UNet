import copy
import time
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from tqdm import tqdm

# from HuramUNet.Lossfn.losses import LovaszLossSoftmax, LovaszLossHinge
from HuramUNet.losses import LovaszLossSoftmax, LovaszLossHinge
from HuramUNet.Model.Unet_model import *
from data_loading import *
import Set_Mode

stm = Set_Mode.set_param()


def train_net(model, stm, best_val_score):
    dataset = MyDataset(stm.images_path, stm.masks_path)

    # Split into train / validation partitions

    n_val = int(len(dataset) * stm.val_precent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_set,
                              batch_size=stm.batch_size,
                              shuffle=True,
                              num_workers=0,
                              pin_memory=True)
    val_loader = DataLoader(val_set,
                            batch_size=stm.batch_size,
                            shuffle=False,
                            num_workers=0,
                            pin_memory=True)

    print(f'''Starting training:
        Epochs:          {stm.num_epoch}
        Batch size:      {stm.batch_size}
        Learning rate:   {stm.learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        n_classes:       {stm.n_classes}
        Device:          {device.type}
    ''')

    optimizer = optim.Adam(model.parameters(), lr=stm.learning_rate, weight_decay=stm.weight_decay)
    # scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
    #                                            milestones=stm.lr_decay_milestones,
    #                                            gamma=stm.lr_decay_gamma)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=3, threshold=0.001)

    if stm.n_classes > 1:
        criterion = LovaszLossSoftmax()
    else:
        criterion = LovaszLossHinge()

    since = time.time()
    global_step = 0
    val_loss_history = []
    train_loss_history = []

    LRs = [optimizer.param_groups[0]['lr']]
    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(stm.num_epoch):

        model.train()
        model.to(device)
        running_loss = 0
        running_corrects = 0
        with tqdm(total=n_train, desc=f'Epoch{epoch + 1}/{stm.num_epoch}', ncols=80) as pbar:
            for batch in train_loader:
                train_image = batch['image']
                mask_image = batch['mask']

                assert train_image.shape[1] == model.input_channel, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {train_image.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                train_image = train_image.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if model.n_classes == 1 else torch.long
                mask_image = mask_image.to(device=device, dtype=mask_type)

                pred_image = model(train_image)

                if model.n_classes == 1:
                    pred_img = pred_image.squeeze(1)
                    true_masks = mask_image.squeeze(1)
                else:
                    pred_img = pred_image
                    true_masks = mask_image

                if stm.deepsupervision:
                    loss = 0
                    for inference_mask in pred_img:
                        loss += criterion(inference_mask, true_masks)
                    loss /= len(pred_img)
                else:
                    loss = criterion(pred_img, true_masks)

                loss = (torch.tensor(0.0, requires_grad=True) if loss == 0 else loss)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update(train_image.shape[0])
                global_step += 1
                running_loss += loss.item()

                pbar.set_postfix({'loss (batch)': loss.item()})
        pbar.close()
        model.eval()

        epoch_loss = running_loss
        scheduler.step(epoch_loss)

        epoch_val_score = eval_net(model, val_loader, device, n_val, stm)
        train_loss_history.append(epoch_loss)
        val_loss_history.append(epoch_val_score)
        LRs.append(optimizer.param_groups[0]['lr'])
        time_elapsed = time.time() - since
        print(f'Epoch{epoch + 1} finish')
        print('------Time elapsed {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('------Epoch train Loss: {:.6f}'.format(epoch_loss))
        if stm.n_classes > 1:
            print('------Epoch validation LovaszLossSoftmax loss: {}'.format(epoch_val_score))
        else:
            print('------Epoch validation LovaszLossHinge loss: {}'.format(epoch_val_score))
        print('Optimizer learning rate : {:.15f}'.format(optimizer.param_groups[0]['lr']))

        if (epoch + 1) % 10 == 0:
            state = {
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'epoch_loss': epoch_loss,
                'epoch_val_score': epoch_val_score
            }
            savepath = os.path.join(stm.model_save_path, 'Checkpoint_epoch{}.pth'.format(epoch + 1))
            torch.save(state, savepath)
            print(f'######_Checkpoint {epoch + 1} saved!')

        if epoch_val_score < best_val_score:
            best_val_score = epoch_val_score
            best_model_wts = copy.deepcopy(model.state_dict())
            best_state = {
                'state_dict': model.state_dict(),
                'best_val_score': best_val_score,
                'optimizer': optimizer.state_dict(),
            }
            best_save_path = os.path.join(stm.model_save_path, 'Best_val_Checkpoint.pth')
            torch.save(best_state, best_save_path)
            print('@@@@@@_Best_val_score_model already save...')
        print('-----' * 20)


def eval_net(net, loader, device, n_val, stm):
    """
    Evaluation without the densecrf with the dice coefficient

    """
    net.eval()
    tot = 0
    if stm.n_classes > 1:
        criterion = LovaszLossSoftmax()
    else:
        criterion = LovaszLossHinge()

    for batch in loader:
        imgs = batch['image']
        true_masks = batch['mask']

        imgs = imgs.to(device=device, dtype=torch.float32)
        mask_type = torch.float32 if net.n_classes == 1 else torch.long
        true_masks = true_masks.to(device=device, dtype=mask_type)

        # compute loss
        if stm.deepsupervision:
            masks_preds = net(imgs)
            loss = 0
            for masks_pred in masks_preds:
                tot_cross_entropy = 0
                for true_mask, pred in zip(true_masks, masks_pred):
                    pred = (pred > stm.rate_threshold).float()
                    # if stm.n_classes > 1:
                    #     sub_cross_entropy = F.cross_entropy(pred.unsqueeze(dim=0),
                    #                                         true_mask.unsqueeze(dim=0).squeeze(1)).item()
                    # else:
                    #     sub_cross_entropy = dice_coeff(pred, true_mask.squeeze(dim=1)).item()
                    if stm.n_classes > 1:
                        loss += criterion(pred.unsqueeze(dim=0),
                                          true_mask.unsqueeze(dim=0).squeeze(1))
                        loss = (torch.tensor(0.0, requires_grad=True) if loss == 0 else loss).item()
                        loss /= len(pred)
                    else:
                        loss = criterion(pred, true_mask.squeeze(dim=1))
                        loss = (torch.tensor(0.0, requires_grad=True) if loss == 0 else loss).item()

                    tot_loss += loss
                tot_loss = tot_loss / len(masks_preds)
                tot += tot_loss
        else:
            masks_pred = net(imgs)
            for true_mask, pred in zip(true_masks, masks_pred):
                pred = (pred > stm.rate_threshold).float()
                if net.n_classes > 1:
                    loss = criterion(pred.unsqueeze(dim=0), true_mask.unsqueeze(dim=0).squeeze(1))
                    loss = (torch.tensor(0.0, requires_grad=True) if loss == 0 else loss).item()
                    tot += loss
                else:
                    loss = criterion(pred, true_mask.squeeze(dim=1))
                    loss = (torch.tensor(0.0, requires_grad=True) if loss == 0 else loss).item()
                    tot += loss

    return tot / n_val


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # model = NestedUNet(stm)
    model = UNet(stm)
    if stm.load:
        best_load_path = os.path.join(stm.model_save_path, 'Best_val_Checkpoint.pth')
        checkpoint = torch.load(best_load_path, map_location=device)
        best_val_score = checkpoint['best_val_score']
        model.load_state_dict(checkpoint['state_dict'])
        print('Model load successful, best_val_score:{}'.format(best_val_score))

    else:
        best_val_score = 100
        print('Do not load any model')

    model.to(device)

    train_net(model, stm, best_val_score)
