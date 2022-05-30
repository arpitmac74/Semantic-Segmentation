import numpy as np
from utils import *
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torchvision.models.segmentation
from unet.unet_model import UNet
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from evaluate import evaluate
from utils.data_loading import *
from data import *
from utils.dice_score import *

def train_net(net,
            len_dataset,
            device,
            batch_size: int = 1,
            epochs: int = 5,
            learning_rate: float = 1e-5,
            amp: bool = False,
            visualize_training = False,
            val_percent: float = 0.1,
            ):

# 1. Create dataset
    n_val = int(len_dataset * val_percent)
    n_train = int(len_dataset - n_val)
    
    dat_1, lbl_1 = load(mode='train',n=n_train,get = 0.0,return_mask =False) #have tumor 
    dat_2, lbl_2 = load(mode='train',n=n_val,get = 1.0,return_mask = False) #dont have
    
    data = np.vstack((dat_1,dat_2))
    labels = np.vstack((lbl_1,lbl_2))

    

# 2. Split into train / validation partitions
    train_size = 1 - val_percent
    inputs_train,inputs_valid = train_test_split(data,random_state=42,train_size=train_size,shuffle=True)
    labels_train,labels_valid = train_test_split(labels,random_state=42,train_size=train_size,shuffle=True)
    
    training_dataset = SegmentationDataSet(inputs_train,labels_train,transform = False)
    val_dataset = SegmentationDataSet(inputs_valid,labels_valid)

    
# 3. Create data loaders
    training_dataloader = DataLoader(dataset=training_dataset,
                                        batch_size=batch_size,
                                        shuffle=False)
    validation_dataloader = DataLoader(dataset=val_dataset,
                                        batch_size=batch_size,
                                        shuffle=False)


# 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.RMSprop(net.parameters(), lr=learning_rate, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=2)  # goal: maximize Dice score
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()
    
    #variables for visualzation and storing
    j=0
    iter_step = n_train/batch_size
    training_loss = []
    loss_val = []
    score_val = []

# 5. Begin training
    for epoch in range(1, epochs+1):
        net.train()
        
        for batch in training_dataloader:
            j=j+1
            images = batch[0]
            true_masks = batch[1]
            true_masks = torch.squeeze(true_masks)

            assert images.shape[1] == net.n_channels, \
                f'Network has been defined with {net.n_channels} input channels, ' \
                f'but loaded images have {images.shape[1]} channels. Please check that ' \
                'the images are loaded correctly.'
            
            images = images.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=torch.long)
            with torch.cuda.amp.autocast(enabled=amp):
                masks_pred = net(images)
                loss_class = criterion(masks_pred, true_masks) 
                loss_dice = dice_loss(F.softmax(masks_pred, dim=1).float(),
                                F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),multiclass=True)
                loss = loss_class + loss_dice
            optimizer.zero_grad(set_to_none=True)
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()

            
            stats = [epoch, epochs, j, iter_step, loss]
            print('\n Train: Epoch: [{}/{}] Iter: [{}/{}] Training Loss: {}'.format(*stats))   
            if j >= iter_step:
                j=0
            
            # if epoch % 5== 0: 
            #     img = images[0].cpu()
            #     img = img.permute(1,2,0)
            #     mask = true_masks[0].float().cpu()
            #     pred = torch.softmax(masks_pred, dim=1).argmax(dim=1)[0].float().cpu()
            #     plt.imshow(mask)
            #     plt.show()
            #     plt.imshow(pred)
            #     plt.show()
            #     plt.imshow(img)
            #     plt.show()
        
        # Evaluation round
        val_score,val_loss = evaluate(net, validation_dataloader, device)
        scheduler.step(val_score)
        stats_val = [epoch, epochs, val_score,val_loss,optimizer.param_groups[0]['lr']]
        print('\n Validation: Epoch: [{}/{}] : score {} Loss {} Lr: {}'.format(*stats_val))  
        if visualize_training:
                training_loss.append(loss.cpu().item())
                loss_val.append(val_loss.cpu().item())
                score_val.append(val_score.cpu().item())

    torch.save(net.state_dict(),"./model/unet_trained.pt")
    
    #visualize training 
    epos = range(1,len(training_loss)+1)
    if visualize_training:
        plt.plot(epos, training_loss, 'g', label='Training loss')
        plt.plot(epos, loss_val, 'b', label='validation loss')
        plt.plot(epos, score_val, 'r', label='validation dice score')
        plt.title("Training loss vs Val loss vs Val dice score:" + str(batch_size) + " data length:" + str(len_dataset))
        plt.xlabel('Epochs')
        plt.ylabel('Loss/Score')
        plt.legend()
        plt.show()
    print("TRAINING DONE")
                    
                

### TRAINING ###
# if __name__ == '__main__':
#     domain = 'DomainA'
#     path = './data/Training/' + domain
#     set_root(path)
#     random_seed = 42
    
#     net = UNet(n_channels=1, n_classes=2, bilinear=True)

#     epochs = 15
#     lr = 1e-5
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     val = 20
#     amp = True
#     net.to(device=device)
#     visualize = True

#     train_net(net=net,
#                 len_dataset = 300,
#                 device=device,
#                 batch_size = 8,
#                 epochs=epochs,
#                 learning_rate=lr,
#                 amp=amp,
#                 visualize_training = visualize,                
#                 val_percent= val / 100,
#                 )