import torch
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import re
import numpy as np
import os
import copy
import argparse
import ast

from helpers import makedir, find_high_activation_crop
from helpers import create_logger

target_size = 300

# specify the test image to be analyzed
parser = argparse.ArgumentParser()
parser.add_argument('-test_img_name', nargs=1, type=str, default='0')
parser.add_argument('-test_img_dir', nargs=1, type=str, default='0')
parser.add_argument('-test_img_label', nargs=1, type=int, default='-1')
parser.add_argument('-test_model_dir', nargs=1, type=str, default='0')
parser.add_argument('-test_model_name', nargs=1, type=str, default='0')
args = parser.parse_args()

test_image_dir = args.test_img_dir[0]
test_image_name =  args.test_img_name[0]
test_image_label = args.test_img_label[0]

test_image_path1 = os.path.join(test_image_dir, test_image_name.format(1))
test_image_path2 = os.path.join(test_image_dir, test_image_name.format(2))
test_image_path3 = os.path.join(test_image_dir, test_image_name.format(3))

##### MODEL AND DATA LOADING
load_model_dir = args.test_model_dir[0]
load_model_name = args.test_model_name[0]

model_base_architecture = load_model_dir.split('/')[-3]
experiment_run = load_model_dir.split('/')[-2]

save_analysis_path = os.path.join(load_model_dir, test_image_name.format(123))
makedir(save_analysis_path)
print(save_analysis_path)

log, logclose = create_logger(\
    log_filename=os.path.join(save_analysis_path, 'local_analysis.log'))

load_model_path = os.path.join(load_model_dir, load_model_name)
epoch_number_str = re.search(r'\d+', load_model_name).group(0)
start_epoch_number = int(epoch_number_str)

log('load model from ' + load_model_path)
log('model base architecture: ' + model_base_architecture)
log('experiment run: ' + experiment_run)

dev = 'cpu'
if torch.has_cuda and torch.cuda.is_available():
    dev = 'cuda'
elif torch.has_mps:
    dev = 'mps'

ppnet = torch.load(load_model_path, map_location=torch.device('cpu'))
ppnet.eval()
ppnet = ppnet.to(dev)
ppnet_multi = torch.nn.DataParallel(ppnet)

img_size = target_size
max_dist_l = []
for prototype_shape in ppnet.prototype_shape_l:
    max_dist_l.append(prototype_shape[1] * prototype_shape[2] * prototype_shape[3])

class_specific = True

##### SANITY CHECK
# confirm prototype class identity
load_img_dir = os.path.join(load_model_dir, 'img')


def save_prototype(mi, fname, epoch, index):
    p_img = plt.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch), \
                                    str(mi)+'proto-img'+str(index)+'.png'))
    plt.imsave(fname, p_img)

def save_prototype_self_activation(mi, fname, epoch, index):
    p_img = plt.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch),
                                    str(mi)+'proto-img-original_with_self_act'+str(index)+'.png'))
    plt.imsave(fname, p_img)

def save_prototype_original_img_with_bbox(mi, fname, epoch, index,
                                          bbox_height_start, bbox_height_end,
                                          bbox_width_start, bbox_width_end, color=(0, 255, 255)):
    p_img_bgr = cv2.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch), \
                                        str(mi)+'proto-img-original'+str(index)+'.png'))
    cv2.rectangle(p_img_bgr, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1),
                  color, thickness=2)
    p_img_rgb = p_img_bgr[...,::-1]
    p_img_rgb = np.float32(p_img_rgb) / 255
    plt.imsave(fname, p_img_rgb)

def save_prototype_full_size(mi, fname, epoch, index, color=(0, 255, 255)):
    p_img_bgr = cv2.imread(os.path.join(load_img_dir, 'epoch-'+str(epoch), \
                                        str(mi)+'proto-img-original'+str(index)+'.png'))
    p_img_rgb = p_img_bgr[...,::-1]
    p_img_rgb = np.float32(p_img_rgb) / 255
    plt.imsave(fname, p_img_rgb)

def imsave_with_bbox(fname, img_rgb, bbox_height_start, bbox_height_end,
                     bbox_width_start, bbox_width_end, color=(0, 255, 255)):
    img_bgr_uint8 = cv2.cvtColor(np.uint8(255*img_rgb), cv2.COLOR_RGB2BGR)
    cv2.rectangle(img_bgr_uint8, (bbox_width_start, bbox_height_start), (bbox_width_end-1, bbox_height_end-1),
                  color, thickness=2)
    img_rgb_uint8 = img_bgr_uint8[...,::-1]
    img_rgb_float = np.float32(img_rgb_uint8) / 255
    plt.imsave(fname, img_rgb_float)

resize = transforms.Compose([
    transforms.Resize(target_size),
])
img_pil1 = resize(Image.open(test_image_path1).convert('RGB'))
img_pil2 = resize(Image.open(test_image_path2).convert('RGB'))
img_pil3 = resize(Image.open(test_image_path3).convert('RGB'))

# load image & label
original_img1 = np.array(img_pil1).astype(np.uint8)
plt.imsave(os.path.join(save_analysis_path, 'original_img1.png'), original_img1)
original_img2 = np.array(img_pil2).astype(np.uint8)
plt.imsave(os.path.join(save_analysis_path, 'original_img2.png'), original_img2)
original_img3 = np.array(img_pil3).astype(np.uint8)
plt.imsave(os.path.join(save_analysis_path, 'original_img3.png'), original_img3)
original_img_l = [original_img1, original_img2, original_img3]

def img_to_tensor(image):
    # convert PIL(image) to tensor (1,C,H,W)
    loader = transforms.Compose([transforms.ToTensor()])
    image = loader(image).unsqueeze(0)
    return image

img_tensor1 = img_to_tensor(img_pil1)# [1, 3, H, W]
img_tensor2 = img_to_tensor(img_pil2)
img_tensor3 = img_to_tensor(img_pil3)
img_tensor = torch.vstack([img_tensor1,img_tensor2,img_tensor3])# [3, 3, H, W]
img_tensor = img_tensor.unsqueeze(0).float().to(dev) # [1, 3, 3, H, W]

images_test = img_tensor
labels_test = torch.tensor([test_image_label])
# inference
logits, min_distances_l, _ = ppnet_multi(images_test)
prototype_activations_l, prototype_activation_patterns_l = [], []
for i,ppnet in enumerate(ppnet_multi.module.pnet123):
    _, distances = ppnet.push_forward(images_test[:,i,...])
    prototype_activations = ppnet.distance_2_similarity(min_distances_l[i])
    prototype_activation_patterns = ppnet.distance_2_similarity(distances)
    if ppnet.prototype_activation_function == 'linear':
        prototype_activations = prototype_activations + max_dist_l[i]
        prototype_activation_patterns = prototype_activation_patterns + max_dist_l[i]
    prototype_activations_l.append(prototype_activations)
    prototype_activation_patterns_l.append(prototype_activation_patterns)

tables = []
for i in range(logits.size(0)):
    tables.append((torch.argmax(logits, dim=1)[i].item(), labels_test[i].item()))
    log(str(i) + ' ' + str(tables[-1]))

idx = 0
predicted_cls = tables[idx][0]
correct_cls = tables[idx][1]
log('Predicted: ' + str(predicted_cls))
log('Actual: ' + str(correct_cls))

##### MOST ACTIVATED (NEAREST) 5 PROTOTYPES OF THIS IMAGE

log('Most activated 5 prototypes of this image:')
for mi,ppnet in enumerate(ppnet_multi.module.pnet123):
    n_protos_cum = sum(ppnet_multi.module.num_prototypes_l[:mi])
    # prepare
    prototype_info = np.load(os.path.join(load_img_dir, \
        'epoch-'+epoch_number_str, str(mi)+'bb'+epoch_number_str+'.npy'))
    prototype_img_identity = prototype_info[:, -1]
    num_classes = len(set(prototype_img_identity))
    log('Prototypes are chosen from ' + str(len(set(prototype_img_identity))) + ' number of classes.')
    log('Their class identities are: ' + str(prototype_img_identity))
    # confirm prototype connects most strongly to its own class
    last_layer_weight = ppnet_multi.module.last_layer.weight[ \
        :,n_protos_cum:n_protos_cum+ppnet.num_prototypes]
    prototype_max_connection = torch.argmax(last_layer_weight, dim=0)
    prototype_max_connection = prototype_max_connection.cpu().numpy()
    if np.sum(prototype_max_connection == prototype_img_identity) == ppnet.num_prototypes:
        log('All prototypes connect most strongly to their respective classes.')
    else:
        log('WARNING: Not all prototypes connect most strongly to their respective classes.')
    #
    array_act, sorted_indices_act = torch.sort(prototype_activations_l[mi][idx])
    dirname = '%dmost_activated_prototypes' % mi
    makedir(os.path.join(save_analysis_path, dirname))
    max_act = 0
    for i in range(1,6):
        log('top {0} activated prototype for this image:'.format(i))
        save_prototype(mi, os.path.join(save_analysis_path, dirname,
                                    'top-%d_activated_prototype.png' % i),
                        start_epoch_number, sorted_indices_act[-i].item())
        save_prototype_full_size(mi, fname=os.path.join(save_analysis_path, dirname,
                                                    'top-%d_activated_prototype_full_size.png' % i),
                                    epoch=start_epoch_number,
                                    index=sorted_indices_act[-i].item(),
                                    color=(0, 255, 255))
        save_prototype_original_img_with_bbox(mi, fname=os.path.join(save_analysis_path, dirname,
                                                                'top-%d_activated_prototype_in_original_pimg.png' % i),
                                                epoch=start_epoch_number,
                                                index=sorted_indices_act[-i].item(),
                                                bbox_height_start=prototype_info[sorted_indices_act[-i].item()][1],
                                                bbox_height_end=prototype_info[sorted_indices_act[-i].item()][2],
                                                bbox_width_start=prototype_info[sorted_indices_act[-i].item()][3],
                                                bbox_width_end=prototype_info[sorted_indices_act[-i].item()][4],
                                                color=(0, 255, 255))
        save_prototype_self_activation(mi, os.path.join(save_analysis_path, dirname,
                                                        'top-%d_activated_prototype_self_act.png' % i),
                                        start_epoch_number, sorted_indices_act[-i].item())
        log('prototype index: {0}'.format(sorted_indices_act[-i].item()))
        log('prototype class identity: {0}'.format(prototype_img_identity[sorted_indices_act[-i].item()]))
        if prototype_max_connection[sorted_indices_act[-i].item()] != prototype_img_identity[sorted_indices_act[-i].item()]:
            log('prototype connection identity: {0}'.format(prototype_max_connection[sorted_indices_act[-i].item()]))
        log('activation value (similarity score): {0}'.format(array_act[-i]))

        f = open(os.path.join(save_analysis_path, dirname, \
            'top-' + str(i) + '_activated_prototype.txt'), "w")
        f.write('similarity: {0:.3f}\n'.format(array_act[-i].item()))
        f.write('last layer connection with predicted class: {0} \n'.format(last_layer_weight[predicted_cls][sorted_indices_act[-i].item()]))
        f.write('proto index:')
        f.write(str(sorted_indices_act[-i].item()) + '\n')
        f.write('number of prototype classes: {0} \n'.format(num_classes))
        for class_id_ in range(num_classes):
            f.write(f'proto connection to class {class_id_}:')
            f.write(str(last_layer_weight[class_id_][sorted_indices_act[-i].item()]) + '\n')
        f.close()

        log('last layer connection with predicted class: {0}'.format(last_layer_weight[predicted_cls][sorted_indices_act[-i].item()]))

        activation_pattern = prototype_activation_patterns_l[mi][idx][sorted_indices_act[-i].item()] \
            .detach().cpu().numpy()
        upsampled_activation_pattern = cv2.resize(activation_pattern, dsize=(img_size, img_size),
                                                interpolation=cv2.INTER_CUBIC)

        # show the most highly activated patch of the image by this prototype
        original_img = original_img_l[mi]
        high_act_patch_indices = find_high_activation_crop(upsampled_activation_pattern)
        high_act_patch = original_img[high_act_patch_indices[0]:high_act_patch_indices[1],
                                      high_act_patch_indices[2]:high_act_patch_indices[3], :]
        log('most highly activated patch of the chosen image by this prototype:')
        #plt.axis('off')
        plt.imsave(os.path.join(save_analysis_path, dirname,
                                str(mi)+'most_highly_activated_patch_by_top-%d_prototype.png' % i),
                    high_act_patch)
        log('most highly activated patch by this prototype shown in the original image:')
        imsave_with_bbox(fname=os.path.join(save_analysis_path, dirname,
                                str(mi)+'most_highly_activated_patch_in_original_img_by_top-%d_prototype.png' % i),
                        img_rgb=original_img,
                        bbox_height_start=high_act_patch_indices[0],
                        bbox_height_end=high_act_patch_indices[1],
                        bbox_width_start=high_act_patch_indices[2],
                        bbox_width_end=high_act_patch_indices[3], color=(0, 255, 255))

        # show the image overlayed with prototype activation map
        rescaled_activation_pattern = upsampled_activation_pattern - np.amin(upsampled_activation_pattern)
        rescaled_activation_pattern = rescaled_activation_pattern / np.amax(rescaled_activation_pattern)
        heatmap = cv2.applyColorMap(np.uint8(255*rescaled_activation_pattern), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        heatmap = heatmap[...,::-1]
        overlayed_img = 0.5 * original_img / 255 + 0.3 * heatmap
        overlayed_img = np.clip(overlayed_img, 0, 1)
        log('prototype activation map of the chosen image:')
        #plt.axis('off')
        plt.imsave(os.path.join(save_analysis_path, dirname,
                    'prototype_activation_map_by_top-%d_prototype.png' % i), overlayed_img)

        # show the image overlayed with different normalized prototype activation map
        rescaled_activation_pattern = upsampled_activation_pattern - np.amin(upsampled_activation_pattern)

        # get the max activation of any proto on this image
        # (works because we start with highest act, must be on rescale)
        if np.amax(rescaled_activation_pattern) > max_act:
            max_act = np.amax(rescaled_activation_pattern)

        rescaled_activation_pattern = rescaled_activation_pattern / max_act
        heatmap = cv2.applyColorMap(np.uint8(255*rescaled_activation_pattern), cv2.COLORMAP_JET)
        heatmap = np.float32(heatmap) / 255
        heatmap = heatmap[...,::-1]
        overlayed_img = 0.5 * original_img / 255 + 0.3 * heatmap
        overlayed_img = np.clip(overlayed_img, 0, 1)
        plt.imsave(os.path.join(save_analysis_path, dirname,
                                'prototype_activation_map_by_top-%d_prototype_normed.png' % i),
                overlayed_img)
        log('--------------------------------------------------------------')


log('--------------------------------------------------------------')
##### PROTOTYPES FROM TOP-k CLASSES
log('--------------------------------------------------------------')
topk_k = 2

log('Prototypes from top-%d classes:' % topk_k)
topk_logits, topk_classes = torch.topk(logits[idx], k=topk_k)

for mi,ppnet in enumerate(ppnet_multi.module.pnet123):
    n_protos_cum = sum(ppnet_multi.module.num_prototypes_l[:mi])
    for i,c in enumerate(topk_classes.detach().cpu().numpy()):
        proto_act_normed_maps = []
        dirname = '%dtop-%d_class_prototypes' % (mi, (i+1))
        makedir(os.path.join(save_analysis_path, dirname))

        log('top %d predicted class: %d' % (i+1, c))
        log('logit of the class: %f' % topk_logits[i])
        class_prototype_indices = np.nonzero(\
            ppnet_multi.module.prototype_class_identity.detach().cpu().numpy() \
                [n_protos_cum:n_protos_cum+ppnet.num_prototypes, c])[0]
        # import ipdb; ipdb.set_trace()
        if class_prototype_indices.shape[0] == 0:
            continue
        class_prototype_activations = prototype_activations_l[mi][idx][class_prototype_indices]
        _, sorted_indices_cls_act = torch.sort(class_prototype_activations)

        prototype_info = np.load(os.path.join(load_img_dir, \
            'epoch-'+epoch_number_str, str(mi)+'bb'+epoch_number_str+'.npy'))
        prototype_img_identity = prototype_info[:, -1]
        num_classes = len(set(prototype_img_identity))

        prototype_cnt = 1
        for j in reversed(sorted_indices_cls_act.detach().cpu().numpy()):
            prototype_index = class_prototype_indices[j]
            save_prototype(mi, os.path.join(save_analysis_path, dirname,
                                            'top-%d_activated_prototype.png' % prototype_cnt),
                                            start_epoch_number,
                                            prototype_index)
            save_prototype_full_size(mi, fname=os.path.join(save_analysis_path, dirname,
                                                'top-%d_activated_prototype_full_size.png' % prototype_cnt),
                                                epoch=start_epoch_number,
                                                index=prototype_index,
                                                color=(0, 255, 255))
            save_prototype_original_img_with_bbox(mi, fname=os.path.join(save_analysis_path, dirname,
                                                    'top-%d_activated_prototype_in_original_pimg.png' % prototype_cnt),
                                                    epoch=start_epoch_number,
                                                    index=prototype_index,
                                                    bbox_height_start=prototype_info[prototype_index][1],
                                                    bbox_height_end=prototype_info[prototype_index][2],
                                                    bbox_width_start=prototype_info[prototype_index][3],
                                                    bbox_width_end=prototype_info[prototype_index][4],
                                                    color=(0, 255, 255))
            save_prototype_self_activation(mi, os.path.join(save_analysis_path, dirname,
                                                            'top-%d_activated_prototype_self_act.png' % prototype_cnt),
                                                            start_epoch_number,
                                                            prototype_index)
            log('prototype index: {0}'.format(prototype_index))
            log('prototype class identity: {0}'.format(prototype_img_identity[prototype_index]))

            last_layer_weight = ppnet_multi.module.last_layer.weight[ \
                :,n_protos_cum:n_protos_cum+ppnet.num_prototypes]
            log('last layer connection: {0}'.format(last_layer_weight[c][prototype_index]))
            prototype_max_connection = torch.argmax(last_layer_weight, dim=0)
            prototype_max_connection = prototype_max_connection.cpu().numpy()

            if prototype_max_connection[prototype_index] != prototype_img_identity[prototype_index]:
                log('prototype connection identity: {0}'.format(prototype_max_connection[prototype_index]))
            log('activation value (similarity score): {0}'.format(prototype_activations_l[mi][idx][prototype_index]))

            activation_pattern = prototype_activation_patterns_l[mi][idx][prototype_index].detach().cpu().numpy()
            upsampled_activation_pattern = cv2.resize(activation_pattern, dsize=(img_size, img_size),
                                                    interpolation=cv2.INTER_CUBIC)
            # logging
            f = open(os.path.join(save_analysis_path, dirname, \
                            'top-' + str(prototype_cnt) + '_activated_prototype.txt'), "w")
            f.write('similarity: {0:.3f}\n'.format(prototype_activations_l[mi][idx][prototype_index]))
            f.write('last layer connection: {0:.3f}\n'.format(last_layer_weight[c][prototype_index]))
            f.write('proto index: ' + str(prototype_index) + '\n')
            for class_id_ in range(num_classes):
                f.write(f'proto connection to class {class_id_}:')
                f.write(str(last_layer_weight[class_id_][prototype_index]) + '\n')
            f.close()
            # show the most highly activated patch of the image by this prototype
            high_act_patch_indices = find_high_activation_crop(upsampled_activation_pattern)
            high_act_patch = original_img[high_act_patch_indices[0]:high_act_patch_indices[1],
                                        high_act_patch_indices[2]:high_act_patch_indices[3], :]
            log('most highly activated patch of the chosen image by this prototype:')
            #plt.axis('off')
            plt.imsave(os.path.join(save_analysis_path, dirname,
                                    'most_highly_activated_patch_by_top-%d_prototype.png' % prototype_cnt),
                        high_act_patch)
            log('most highly activated patch by this prototype shown in the original image:')
            imsave_with_bbox(fname=os.path.join(save_analysis_path, dirname,
                                                'most_highly_activated_patch_in_original_img_by_top-%d_prototype.png' % prototype_cnt),
                             img_rgb=original_img,
                             bbox_height_start=high_act_patch_indices[0],
                             bbox_height_end=high_act_patch_indices[1],
                             bbox_width_start=high_act_patch_indices[2],
                             bbox_width_end=high_act_patch_indices[3], color=(0, 255, 255))

            # show the image overlayed with prototype activation map
            rescaled_activation_pattern = upsampled_activation_pattern - np.amin(upsampled_activation_pattern)
            rescaled_activation_pattern = rescaled_activation_pattern / np.amax(rescaled_activation_pattern)
            heatmap = cv2.applyColorMap(np.uint8(255*rescaled_activation_pattern), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            heatmap = heatmap[...,::-1]
            overlayed_img = 0.5 * original_img / 255 + 0.3 * heatmap
            overlayed_img = np.clip(overlayed_img, 0, 1)
            log('prototype activation map of the chosen image:')
            #plt.axis('off')
            plt.imsave(os.path.join(save_analysis_path, dirname,
                                    'prototype_activation_map_by_top-%d_prototype.png' % prototype_cnt),
                    overlayed_img)

            # show the image overlayed with differently normed prototype activation map
            rescaled_activation_pattern = upsampled_activation_pattern - np.amin(upsampled_activation_pattern)
            rescaled_activation_pattern = rescaled_activation_pattern / max_act
            heatmap = cv2.applyColorMap(np.uint8(255*rescaled_activation_pattern), cv2.COLORMAP_JET)
            heatmap = np.float32(heatmap) / 255
            heatmap = heatmap[...,::-1]
            overlayed_img = 0.5 * original_img / 255 + 0.3 * heatmap
            overlayed_img = np.clip(overlayed_img, 0, 1)
            log('prototype activation map of the chosen image:')
            #plt.axis('off')
            plt.imsave(os.path.join(save_analysis_path, dirname,
                                    'prototype_activation_map_by_top-%d_prototype_normed.png' % prototype_cnt),
                    overlayed_img)
            proto_act_normed_maps.append(rescaled_activation_pattern)
            log('--------------------------------------------------------------')
            prototype_cnt += 1

        class_heatmap = np.average(np.asarray(proto_act_normed_maps), axis=0)
        class_heatmap = class_heatmap - np.amin(class_heatmap)
        class_heatmap = class_heatmap / np.amax(class_heatmap)
        class_heatmap = cv2.applyColorMap(np.uint8(255*class_heatmap), cv2.COLORMAP_JET)
        class_heatmap = np.float32(class_heatmap) / 255
        class_heatmap = class_heatmap[...,::-1]

        overlayed_img = 0.5 * original_img / 255 + 0.3 * class_heatmap
        overlayed_img = np.clip(overlayed_img, 0, 1)
        plt.imsave(os.path.join(save_analysis_path, dirname, \
                    'prototype_activation_map_by_whole_%d_class.png' % c), overlayed_img)


if predicted_cls == correct_cls:
    log('Prediction is correct.')
else:
    log('Prediction is wrong.')
print("saved in ", save_analysis_path)

logclose()

