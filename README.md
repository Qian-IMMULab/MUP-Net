# MUP-Net
This repository contains code for paper `A domain-knowledge based interpretable deep learning system for improving clinical breast ultrasound diagnosis`.

## Requirements
torch>=2.0.0, torchvision>=0.16.0, PIL>=10.0.0, cv2>=4.0.0

## Preparing dataset
The directory containing dataset should be organized like this:
```
/path/to/dataset
    /1
        /ROI_1.png
        /ROI_2.png
        /ROI_3.png
    /2
        /ROI_1.png
        /ROI_2.png
        /ROI_3.png
    ...
```
Each number named folder represent a case. Images ROI_1/2/3.png corresponding to B-mode, colour Doppler, and elastography modalities, respectively.

We also need `.xlsx` files to record the list of train, push, and validation subset. Each file has three columns:
```
num: patient id
age: age of patient (not used in this work)
label: biopsy-confirmed pathology label (pos:1, neg:0)
```

## Data augmentation
We use augmentation techniques such as horizontal flipping, random rotation, and Gaussian blurring to increase the size of malignant cases. Run the following command to perform this augmentation.
```
python3 0_aug_train_data.py -start XXX -inc YYY
```
The `-start` is the start patient id of augmented samples. The `-inc` is the number of positive samples to be added.

## Training the model
Run the following command to train MUP-Net. The `-nprotos` parameter represents the number of prototypes assigned to each class.
```
python3 1_train_ppnet.py -nprotos 10
```
The output folder will be named as `model_dir{#nprotos}-{$datetime}`. It contains saved weights, prototype images, and training log.

## Generating GradCAM results
In `2_run_gradcam.py`, set `model_path` pointing to your trained model weights. Then, run the following command to generate GradCAM and GradCAM++ output.
```
python3 2_run_gradcam.py
```
The results will be saved into the same folder of your model weights.

## Analyzing a case
First, preparing `3_run_local_analysis.sh` file by: 1. Modifying `MODELFOLDER` and `MODELNAME` to the directory of training output and the file name of the saved model weights. 2. Modifying `TESTCASE` and `TESTCASE_LABEL` to the directory of a test case and the groundtruth label of this case. Then, run `3_run_local_analysis.sh` to visualizing the reasoning cues of model.
