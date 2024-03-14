import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")
import torch
import torch.nn as nn
import numpy as np
import argparse
from datetime import datetime

import tnt
import push
from helpers import create_logger, makedir, save_model_w_condition
from dataHelper import MyDataset

from model import resnet18_features, PPNet3

target_size = 300
prototype_shape = None
topk_k=5
prototype_activation_function = "log"
add_on_layers_type = 'regular'
class_specific = True
num_classes = 2

def construct_PPNet(pretrained=True):
    feature_extractor = resnet18_features
    features1 = feature_extractor(pretrained=pretrained)
    features2 = feature_extractor(pretrained=pretrained)
    features3 = feature_extractor(pretrained=pretrained)
    layer_filter_sizes, layer_strides, layer_paddings = features1.conv_info()
    print('layer_filter_sizes, layer_strides, layer_paddings: ', layer_filter_sizes, layer_strides, layer_paddings)
    print('len of layers: ', len(layer_filter_sizes))

    return PPNet3(features=[features1,features2,features3],
                 img_size=target_size,
                 prototype_shape=prototype_shape,
                 topk_k=topk_k,
                 num_classes=num_classes,
                 init_weights=True,
                 prototype_activation_function=prototype_activation_function,
                 add_on_layers_type=add_on_layers_type,
                 class_specific=class_specific)

def main():
    #################################################################
    # PPNet
    #################################################################
    dev = 'cpu'
    if torch.has_cuda and torch.cuda.is_available():
        dev = 'cuda'
    elif torch.has_mps:
        dev = 'mps'
    print('using dev:', dev)
    ppnet = construct_PPNet().to(dev)
    ppnet_multi = torch.nn.DataParallel(ppnet)

    #################################################################
    # Optimizers
    #################################################################
    joint_optimizer_lrs = {'features': 2e-4,
                        'add_on_layers': 3e-3,
                        'prototype_vectors': 3e-3}
    joint_lr_step_size = 5
    warm_optimizer_lrs = {'add_on_layers': 2e-3,
                        'prototype_vectors': 3e-3}
    last_layer_optimizer_lr = 1e-3
    def merge_params(params):
        ret=[]
        for p in params:
            ret += list(p)
        return ret
    # joint
    joint_optimizer_specs = \
    [{'params': merge_params([p.features.parameters() for p in ppnet.pnet123]),
        'lr': joint_optimizer_lrs['features'], 'weight_decay': 1e-3}, # bias are now also being regularized
    {'params': merge_params([p.add_on_layers.parameters() for p in ppnet.pnet123]),
        'lr': joint_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
    {'params': [p.prototype_vectors for p in ppnet.pnet123],
        'lr': joint_optimizer_lrs['prototype_vectors']},
    ]
    joint_optimizer = torch.optim.Adam(joint_optimizer_specs)
    joint_lr_scheduler = torch.optim.lr_scheduler.StepLR(joint_optimizer, \
                                step_size=joint_lr_step_size, gamma=0.1)
    # warm
    warm_optimizer_specs = \
    [{'params': merge_params([p.add_on_layers.parameters() for p in ppnet.pnet123]),
        'lr': warm_optimizer_lrs['add_on_layers'], 'weight_decay': 1e-3},
    {'params': [p.prototype_vectors for p in ppnet.pnet123],
        'lr': warm_optimizer_lrs['prototype_vectors']},
    ]
    warm_optimizer = torch.optim.Adam(warm_optimizer_specs)
    # last
    last_layer_optimizer_specs = \
    [{'params': ppnet.last_layer.parameters(),
        'lr': last_layer_optimizer_lr}
    ]
    last_layer_optimizer = torch.optim.Adam(last_layer_optimizer_specs)
    #################################################################
    # Training
    #################################################################
    train_batch_size = 40
    test_batch_size = 40
    train_push_batch_size = 40
    tar_size = [target_size, target_size]
    # train set
    train_dataset = MyDataset(
            file_list='./train345-2-aug.xlsx',
            root_dir='./dataset',
            target_size=tar_size,
            is_train=True,
            )
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True,
        num_workers=4, pin_memory=False)
    # push set
    train_push_dataset = MyDataset(
            file_list='./train345-2-push.xlsx',
            root_dir='./dataset',
            target_size=tar_size,
            is_train=False,
            )
    train_push_loader = torch.utils.data.DataLoader(
        train_push_dataset, batch_size=train_push_batch_size, shuffle=False,
        num_workers=4, pin_memory=False)
    train_push_dataset2 = MyDataset(
            file_list='./train345-2-push3.xlsx',
            root_dir='./dataset',
            target_size=tar_size,
            is_train=False,
            )
    train_push_loader2 = torch.utils.data.DataLoader(
        train_push_dataset2, batch_size=train_push_batch_size, shuffle=False,
        num_workers=4, pin_memory=False)
    # test set
    test_dataset = MyDataset(
            file_list='./val345-2.xlsx',
            root_dir='./dataset',
            target_size=tar_size,
            is_train=False,
            )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch_size, shuffle=False,
        num_workers=4, pin_memory=False)

    print('len(train_loader):', len(train_loader))

    #################################################################
    # Training
    #################################################################
    model_dir = './model_dir' + str(prototype_shape[0]) +'_'+ datetime.now().strftime('%m.%d-%H.%M')
    makedir(model_dir)
    print("saving models to: ", model_dir)
    protos_img_dir = os.path.join(model_dir, 'img')
    makedir(protos_img_dir)
    print("saving protos to: ", protos_img_dir)

    prototype_activation_function_in_numpy = prototype_activation_function
    prototype_img_filename_prefix = 'proto-img'
    prototype_self_act_filename_prefix = 'proto-self-act'
    proto_bound_boxes_filename_prefix = 'bb'

    coefs = {
        'crs_ent': 1,
        'clst': 0.8,
        'sep': -0.08,
        'l1': 1e-4,
    }
    num_train_epochs = 40
    num_warm_epochs = 10
    push_start = 10
    push_epochs = [i for i in range(1,num_train_epochs+1) if i % 10 == 0]

    log, logclose = create_logger(log_filename=os.path.join(model_dir, 'train.log'))
    train_auc = []
    test_auc = []
    push_auc = np.zeros([0,2])
    last_train_auc = np.zeros([0,2])
    last_test_auc = np.zeros([0,2])
    currbest, best_epoch = 0, -1
    #
    def savefig():
        cnt = len(train_auc)+1
        plt.plot(range(1,cnt), train_auc, "b", lw=1, label="train")
        plt.plot(range(1,cnt), test_auc, "r", lw=1, label="test")
        plt.scatter(push_auc[:,0], push_auc[:,1], marker="x", c="g", label="push_test")
        plt.scatter(last_train_auc[:,0], last_train_auc[:,1], marker=".", c="c", label="last_train")
        plt.scatter(last_test_auc[:,0], last_test_auc[:,1], marker=".", c="y", label="last_test")
        plt.ylim(0.4, 1)
        plt.xticks([1]+push_epochs)
        plt.legend()
        plt.grid(True)
        plt.savefig(os.path.join(model_dir, 'train_test_auc.png'))
        plt.close()
    #
    best_acc=0
    for epoch in range(1, num_train_epochs+1):
        log('epoch: \t{0}'.format(epoch))

        if epoch <= num_warm_epochs:
            tnt.warm_only(model123=ppnet_multi, log=log)
            _ = tnt.train(model123=ppnet_multi, dataloader=train_loader, optimizer=warm_optimizer, dev=dev,
                        class_specific=class_specific, coefs=coefs, log=log)
        else:
            tnt.joint(model123=ppnet_multi, log=log)
            _ = tnt.train(model123=ppnet_multi, dataloader=train_loader, optimizer=joint_optimizer, dev=dev,
                        class_specific=class_specific, coefs=coefs, log=log)
            joint_lr_scheduler.step()

        auc = tnt.test(model123=ppnet_multi, dataloader=test_loader, dev=dev,
                        class_specific=class_specific, log=log)
        best_acc = auc if auc > best_acc else best_acc

        train_auc.append(_)
        if currbest < auc:
            currbest = auc
            best_epoch = epoch
        log("\tcurrent best auc is: \t\t{} at epoch {}".format(currbest, best_epoch))
        test_auc.append(auc)
        savefig()

        if epoch >= push_start and epoch in push_epochs:
            for mi,proto_net in enumerate(ppnet_multi.module.pnet123):
                push_loader = train_push_loader if mi in [0,1] else train_push_loader2
                offset = sum(ppnet.num_prototypes_l[0:mi])
                num_prototypes = ppnet.num_prototypes_l[mi]
                class_identity = ppnet_multi.module.prototype_class_identity[offset:offset+num_prototypes,:]
                push.push_prototypes(
                    mi,
                    dev,
                    class_identity,
                    push_loader, # pytorch dataloader (must be unnormalized in [0,1])
                    prototype_network=proto_net, # pytorch network with prototype_vectors
                    class_specific=class_specific,
                    preprocess_input_function=None, # normalize if needed
                    prototype_layer_stride=1,
                    root_dir_for_saving_prototypes=protos_img_dir, # if not None, prototypes will be saved here
                    epoch_number=epoch, # if not provided, prototypes saved previously will be overwritten
                    prototype_img_filename_prefix=prototype_img_filename_prefix,
                    prototype_self_act_filename_prefix=prototype_self_act_filename_prefix,
                    proto_bound_boxes_filename_prefix=proto_bound_boxes_filename_prefix,
                    save_prototype_class_identity=True,
                    log=log,
                    prototype_activation_function_in_numpy=prototype_activation_function_in_numpy)
            auc = tnt.test(model123=ppnet_multi, dataloader=test_loader, dev=dev,
                            class_specific=class_specific, log=log)
            push_auc = np.row_stack((push_auc, np.array([[epoch,auc]])))
            save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'push', accu=auc,
                                        target_accu=best_acc, log=log)
            best_acc = auc if auc > best_acc else best_acc

            if prototype_activation_function != 'linear':
                tnt.last_only(model123=ppnet_multi, log=log)
                iters = 20 if epoch == push_epochs[-1] else 10
                for i in range(iters):
                    log('iteration: \t{0}'.format(i))
                    _ = tnt.train(model123=ppnet_multi, dataloader=train_loader, optimizer=last_layer_optimizer, dev=dev,
                                class_specific=class_specific, coefs=coefs, log=log)
                    auc = tnt.test(model123=ppnet_multi, dataloader=test_loader, dev=dev,
                                    class_specific=class_specific, log=log)
                    save_model_w_condition(model=ppnet, model_dir=model_dir,
                                            model_name=str(epoch) + '_' + str(i) + 'push', accu=auc,
                                            target_accu=best_acc, log=log)
                    best_acc = auc if auc > best_acc else best_acc
                    last_train_auc = np.row_stack((last_train_auc, np.array([[epoch,_]])))
                    last_test_auc = np.row_stack((last_test_auc, np.array([[epoch,auc]])))

                    if currbest < auc:
                        currbest = auc
                        best_epoch = epoch

                    savefig()
            save_model_w_condition(model=ppnet, model_dir=model_dir, model_name=str(epoch) + 'last', accu=auc,
                                    target_accu=0.5, log=log)
    logclose()
    return model_dir, auc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-nprotos', default=6, type=int)
    args = parser.parse_args()

    prototype_shape = (args.nprotos, 512, 1, 1)
    model_dir, test_auc = main()
    if test_auc < 0.91:
        import shutil
        shutil.rmtree(model_dir)