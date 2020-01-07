#!/bin/sh
python3 compute_perturbation_magnitude.py \
    -original_data_root "data/original datasets"\
    -adversarial_data_root "data/adversarial examples"\
    -output_dir "data/stats"