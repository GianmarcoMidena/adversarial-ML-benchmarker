#!/bin/sh
python3 download_images.py \
    -dataset_name_original "ImageNet" \
    -n_images 100 \
    -output_dir "data/original datasets/ImageNet"
python3 download_images.py \
     -dataset_name_original "CIFAR10" \
     -split "test" \
     -n_images 100 \
     -output_dir "data/original datasets/CIFAR10"
python3 download_images.py \
    -dataset_name_original "MNIST" \
    -split "test" \
    -n_images 100 \
    -output_dir "data/original datasets/MNIST"