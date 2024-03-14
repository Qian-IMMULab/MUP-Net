### Adapted from https://github.com/stefannc/GradCAM-Pytorch/blob/07fd6ece5010f7c1c9fbcc8155a60023819111d7/example.ipynb retrieved Mar 3 2021 #####

## cell 1: imports
import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
from torchvision.utils import make_grid, save_image
import torchvision.transforms as transforms
import PIL

from dataHelper import MyDataset
from c_gradcam_utils import visualize_cam, Normalize
from c_gradcam_core import GradCAM, GradCAMpp
from model import PPNet, VGG_features

def main():
    dev = 'cpu'
    if torch.has_cuda and torch.cuda.is_available():
        dev = 'cuda'
    elif torch.has_mps:
       dev = 'mps'
    print('using dev:', dev)

    ## get our mammo img
    target_size = 224
    test_dataset = MyDataset(
            file_list='./test345-2-sel1.xlsx',
            root_dir='./dataset',
            target_size=target_size,
            is_train=True,
        )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=True,
        num_workers=4, pin_memory=False)

    ## load model
    # pnet
    model_path = './path/to/model/40last12.pth'
    ppnet3 = torch.load(model_path, map_location=torch.device('cpu'))
    ppnet3.eval()
    ppnet3.to(dev)

    for i, batch_data in enumerate(test_loader):
        sample_img, target = batch_data
        torch_img3 = sample_img.to(dev)
        torch_img3.requires_grad = True
        save_gradcam(ppnet3, i, torch_img3, dev, target_size, model_path)

def save_gradcam(ppnet3, idx, input_img, dev, target_size, model_path):
    cam_dict = dict()
    for mi, pnet in enumerate(ppnet3.pnet123):
        print('----------------', mi, '----------------')
        pnet_model_dict = dict(type='resnet', arch=pnet.features, layer_name='layer4_basicblock0_relu',
                                input_size=(target_size, target_size))
        pnet_gradcam = GradCAM(pnet_model_dict, dev, True)
        pnet_gradcampp = GradCAMpp(pnet_model_dict, True)
        cam_dict['pnet'] = [pnet_gradcam, pnet_gradcampp]

        ## cell 5: make image grid
        images = []
        torch_img = input_img[:,mi,...]
        for gradcam, gradcam_pp in cam_dict.values():
            mask, _ = gradcam(torch_img)
            print("Min of mask is: ", torch.min(mask))
            print("Max of mask is: ", torch.max(mask))

            heatmap, result = visualize_cam(mask, torch_img)

            mask_pp, _ = gradcam_pp(torch_img)
            print("Max of mask_pp is: ", torch.max(mask_pp))
            heatmap_pp, result_pp = visualize_cam(mask_pp, torch_img)

            images.append(torch.stack([torch_img.squeeze().cpu(),
                                        heatmap, heatmap_pp,
                                        result, result_pp], 0))

        image_grid = make_grid(torch.cat(images, 0), nrow=5)

        ## cell 6: save image grid
        os.makedirs(model_path[:-4], exist_ok=True)
        output_path = os.path.join(model_path[:-4], str(idx)+'-'+str(mi)+'gradCAM_img.png')
        save_image(image_grid, output_path)

if __name__ == '__main__':
    main()