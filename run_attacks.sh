#!/bin/sh
python3 run_attacks.py \
    -attack_methods "FGM" "BIM" "DeepFool" "C&W" "BIM" "PGD" "SaliencyMap"\
    -tools "ART" "CleverHans" "Foolbox"\
    -data_sets "MNIST" "CIFAR10" "ImageNet"\
    -input_data_root "data/original datasets"\
    -output_data_root "data/adversarial examples"\
    -batch_size 50
