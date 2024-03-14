#!/bin/bash

MODELFOLDER=./model_dir10_02-10.10
MODELNAME=40last0.9118.pth

TESTCASE='./dataset/1'
TESTCASE_LABEL=1

FILENAME=ROI_{}.png
DIRNAME=ROI_123.png

rm -rf "$MODELFOLDER/$DIRNAME/"

python3 3_local_analysis.py -test_img_name "$FILENAME" \
                            -test_img_dir "$TESTCASE" \
                            -test_img_label "$TESTCASE_LABEL" \
                            -test_model_dir "$MODELFOLDER/" \
                            -test_model_name "$MODELNAME"

python3 3_local_analysis_vis.py -local_analysis_directory "$MODELFOLDER/$DIRNAME/"
