import torch.nn as nn
import torch
import torch.nn.functional as F
from utils.dice_score import multiclass_dice_coeff, dice_coeff,dice_loss

def evaluate(net, dataloader, device):
    net.eval()
    num_val_batches = len(dataloader)
    dice_score = 0

    criterion = nn.CrossEntropyLoss()

    # iterate over the validation set
    for batch in dataloader:
        image, mask_true = batch[0], batch[1]
        mask_true = torch.squeeze(mask_true,1)
        # move images and labels to correct device and type
        image = image.to(device=device, dtype=torch.float32)
        mask_true_1 = mask_true.to(device=device, dtype=torch.long)
        mask_true = F.one_hot(mask_true_1, net.n_classes).permute(0, 3, 1, 2).float()

        with torch.no_grad():
            # predict the mask
            mask_pred = net(image)
            
            #calculate loss - cross entropy and dice loss 
            loss_class = criterion(mask_pred, mask_true_1) 
            loss_dice = dice_loss(F.softmax(mask_pred, dim=1).float(),
                            F.one_hot(mask_true_1, net.n_classes).permute(0, 3, 1, 2).float(),multiclass=True)
            val_loss = loss_class +loss_dice
            
            # convert to one-hot format
            if net.n_classes == 1:
                mask_pred = (F.sigmoid(mask_pred) > 0.5).float()
                # compute the Dice score
                dice_score += dice_coeff(mask_pred, mask_true, reduce_batch_first=False)
            else:
                mask_pred = F.one_hot(mask_pred.argmax(dim=1), net.n_classes).permute(0, 3, 1, 2).float()
                # compute the Dice score, ignoring background
                dice_score += multiclass_dice_coeff(mask_pred[:, 1:, ...], mask_true[:, 1:, ...], reduce_batch_first=False)

    net.train()

    # Fixes a potential division by zero error
    if num_val_batches == 0:
        return dice_score
    return dice_score / num_val_batches,val_loss