#!/bin/sh
python3 run_defenses.py \
    -original_data_root "data/original datasets"\
    -adversarial_data_root "data/adversarial examples"\
    -output_dir "data/stats"\
    -batch_size 50