from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import pandas as pd
import os
import pathlib
from glob import glob

from utils.utils import set_up_logger, calc_mean_perturbation_magnitude


def main(original_data_root, adversarial_data_root, output_dir):
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    perturbations_path = os.path.join(output_dir, "perturbations.csv")
    if os.path.isfile(perturbations_path):
        perturbations = pd.read_csv(perturbations_path)
        perturbations = perturbations[['dataset', 'attack', 'targeted', 'tool', 'sample size', 'perturbation']]
    else:
        perturbations = pd.DataFrame(columns=['dataset', 'attack', 'targeted', 'tool', 'sample size', 'perturbation'])
    for attack_path in glob(os.path.join(adversarial_data_root, "*")):
        sample_size = len(glob(os.path.join(attack_path, "*.png")))
        if not os.path.isdir(attack_path) or sample_size == 0:
            continue
        is_targeted, attack_name, dataset_name, tool_name = os.path.basename(attack_path).split("_")

        _logger.info(f"tool: {tool_name}, attack: {attack_name}, targeted: {is_targeted}, dataset: {dataset_name}")

        is_targeted = True if is_targeted == "targeted" else False
        if not perturbations[perturbations['dataset'].eq(dataset_name) &
                             perturbations['tool'].eq(tool_name) &
                             perturbations['attack'].eq(attack_name) &
                             perturbations['targeted'].eq(False) &
                             perturbations['sample size'].ge(sample_size)].empty:
            continue
        perturbations = perturbations[~(perturbations['dataset'].eq(dataset_name) &
                                        perturbations['tool'].eq(tool_name) &
                                        perturbations['attack'].eq(attack_name) &
                                        perturbations['targeted'].eq(False))]

        dataset_dir = os.path.join(original_data_root, dataset_name)

        mean_perturbation_magnitude = calc_mean_perturbation_magnitude(image_dir=attack_path, dataset_dir=dataset_dir,
                                                                       logger=_logger)

        perturbation_magnitude = pd.DataFrame({'dataset': [dataset_name], 'attack': [attack_name],
                                               'tool': [tool_name], 'targeted': [is_targeted],
                                               'sample size': [sample_size],
                                               'perturbation': [mean_perturbation_magnitude]})
        perturbation_magnitude['targeted'] = perturbation_magnitude['targeted'].astype(bool)
        perturbations = perturbations.append(perturbation_magnitude, ignore_index=True, sort=False)
        perturbations = perturbations.drop_duplicates()
        perturbations = perturbations.sort_values(by=["dataset", "attack", "targeted", "tool"])
        perturbations.to_csv(perturbations_path, index=False, float_format="%.0E")


if __name__ == '__main__':
    _logger = set_up_logger(__name__)
    parser = argparse.ArgumentParser()
    parser.add_argument("-original_data_root", required=True,
                        help="path to the root containing the original data directories")
    parser.add_argument("-adversarial_data_root", required=True,
                        help="path to the root containing the adversarial data directories")
    parser.add_argument("-output_dir", required=True,
                        help="path to the output directory")
    args = parser.parse_args()
    original_data_root = args.original_data_root
    if not os.path.isdir(original_data_root):
        raise Exception(f"Attention: \"{original_data_root}\" path does not exist!")
    adversarial_data_root = args.adversarial_data_root
    if not os.path.isdir(adversarial_data_root):
        raise Exception(f"Attention: \"{adversarial_data_root}\" path does not exist!")
    output_dir = args.output_dir
    if not os.path.isdir(output_dir):
        raise Exception(f"Attention: \"{output_dir}\" path does not exist!")
    main(original_data_root=original_data_root,
         adversarial_data_root=adversarial_data_root,
         output_dir=output_dir)
