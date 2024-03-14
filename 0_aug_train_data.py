import os
import argparse
import pandas as pd
import torchvision.transforms as transforms
from PIL import Image

def aug_train_dat(root_dir, t2set, start_idx, inc):
    pos = t2set[t2set['label(pos:1, neg:0)'] == 1]
    npos = pos.shape[0]
    sampled = None
    if inc < npos:
        sampled = pos.sample(n=inc, replace=False, random_state=345)
    elif inc == npos:
        sampled = pos
    else:
        sampled = pos
        sampled = pd.concat([sampled, pos.sample(n=inc-npos, replace=False, random_state=345)])
    print(sampled)
    i=0
    for _, row in sampled.iterrows():
        num = row['num']
        img1 = Image.open(os.path.join(root_dir, str(int(num)), 'ROI_1.png')).convert('RGB')
        img2 = Image.open(os.path.join(root_dir, str(int(num)), 'ROI_2.png')).convert('RGB')
        img3 = Image.open(os.path.join(root_dir, str(int(num)), 'ROI_3.png')).convert('RGB')

        transform1 = transforms.Compose([
                        transforms.RandomHorizontalFlip(p=1),
                        transforms.ColorJitter(brightness=0.1, contrast=0.1),
                        transforms.RandomRotation(10),
                        transforms.GaussianBlur(kernel_size=3),
                        transforms.RandomAutocontrast(p=1)
                     ])
        img1 = transform1(img1)
        img2 = transform1(img2)
        img3 = transform1(img3)

        new_num = start_idx + i
        print('add {} -> {}'.format(num, new_num))
        os.makedirs(os.path.join(root_dir, str(int(new_num))))
        img1.save(os.path.join(root_dir, str(int(new_num)), 'ROI_1.png'))
        img2.save(os.path.join(root_dir, str(int(new_num)), 'ROI_2.png'))
        img3.save(os.path.join(root_dir, str(int(new_num)), 'ROI_3.png'))
        row['num'] = new_num
        i=i+1

    return pd.concat([t2set, sampled])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-root', default='./dataset', type=str)
    parser.add_argument('-start', default=1441, type=int)
    parser.add_argument('-inc', default=310, type=int)
    args = parser.parse_args()

    t2 = pd.read_excel('train345-2.xlsx')
    new_t2 = aug_train_dat(args.root, t2, args.start, args.inc)
    new_t2.to_excel('train345-2-aug.xlsx', index=False)
