import numpy as np
from utils import *
import torch
import torch.nn.functional as F
from data import *
from utils.data_loading import *


def predict(net,input,device):
    net.eval()
    prediction_dataset = SegmentationDataSet(input,None,transform = False)
    result = []
    with torch.no_grad():
        for img in prediction_dataset:
            img = torch.unsqueeze(img,0)
            if device != 'cpu':
                img = img.to(device=device, dtype=torch.float32)
            
            output = net(img)
            probs = F.softmax(output,dim=1)[0]
            full_mask = probs.cpu()
            mask =  F.one_hot(full_mask.argmax(dim=0), net.n_classes).permute(2, 0, 1).numpy()
            pred_mask = np.argmax(mask, axis=0) * 255 / mask.shape[0]
            result.append(torch.from_numpy(pred_mask))
    result= torch.stack(result)
    return result









