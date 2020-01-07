from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pathlib
import numpy as np
import pandas as pd
from glob import glob
import json
from utils.utils import set_up_logger

from models.ready_models.inception_v3_model import InceptionV3Model
from models.ready_models.madry_model import MadryModel
from models.ready_models.wide_resnet_model import WideResNetModel
from utils.utils import score_attacks_models

MODEL_CKPTS_ROOT = "model checkpoints"
MODEL_DIR_BY_KEY = {
    "ImageNetInceptionV3": os.path.join(MODEL_CKPTS_ROOT, "inception_v3_ckpt"),
    "ImageNetInceptionV3Adversarial": os.path.join(MODEL_CKPTS_ROOT,
                                                   "adv_inception_v3_ckpt/adv_inception_v3.ckpt"),
    "Cifar10ResNet": os.path.join(MODEL_CKPTS_ROOT, "wide_resnet_ckpt"),
    "Cifar10ResNetAdversarial": os.path.join(MODEL_CKPTS_ROOT, "adv_wide_resnet_ckpt"),
    "MNISTMadry": os.path.join(MODEL_CKPTS_ROOT, "madry_ckpt"),
    "MNISTMadryAdversarial": os.path.join(MODEL_CKPTS_ROOT, "adv_madry_ckpt"),
}
MODELS_BY_DATASET = {
    'CIFAR10': {
        "Cifar10ResNet": WideResNetModel,
        "Cifar10ResNetAdversarial": WideResNetModel
    },
    'MNIST': {
        "MNISTMadry": MadryModel,
        "MNISTMadryAdversarial": MadryModel
    },
    'ImageNet': {
        "ImageNetInceptionV3": InceptionV3Model,
        "ImageNetInceptionV3Adversarial": InceptionV3Model
    }
}


def defend(original_data_root, adversarial_data_root, output_dir, batch_size):
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    attack_model_scores_path = os.path.join(output_dir, "attack_model_scores.csv")
    if os.path.isfile(attack_model_scores_path):
        attack_model_scores = pd.read_csv(attack_model_scores_path, na_values=[np.nan])
    else:
        attack_model_scores = pd.DataFrame(columns=["dataset", "attack", "targeted",
                                                    "model", "tool", "n_samples"])

    for filepath in glob(os.path.join(adversarial_data_root, "*")):
        n_samples = len(glob(os.path.join(filepath, "*.png")))
        if not os.path.isdir(filepath) or n_samples == 0:
            continue
        is_targeted, attack_name, dataset_name, tool_name = os.path.basename(filepath).split("_")

        if dataset_name not in MODELS_BY_DATASET:
            continue

        is_targeted = True if is_targeted == "targeted" else False

        dataset_metadata_path = os.path.join(original_data_root, dataset_name,
                                             "{}_metadata.json".format(dataset_name.lower()))
        with open(dataset_metadata_path, 'r') as f:
            dataset_metadata = json.load(f)

        for model_name, model in MODELS_BY_DATASET[dataset_name].items():
            if not attack_model_scores[attack_model_scores['dataset'].eq(dataset_name) &
                                       attack_model_scores['tool'].eq(tool_name) &
                                       attack_model_scores['targeted'].eq(is_targeted) &
                                       attack_model_scores['attack'].eq(attack_name) &
                                       attack_model_scores['model'].eq(model_name) &
                                       attack_model_scores['n_samples'].ge(n_samples)].empty:
                continue
            attack_model_scores = attack_model_scores[~(attack_model_scores['dataset'].eq(dataset_name) &
                                                        attack_model_scores['tool'].eq(tool_name) &
                                                        attack_model_scores['targeted'].eq(is_targeted) &
                                                        attack_model_scores['attack'].eq(attack_name) &
                                                        attack_model_scores['model'].eq(model_name))]

            _logger.info(f"run model {model_name}, tool: {tool_name}, attack: {attack_name}, "
                         f"targeted: {is_targeted}, dataset: {dataset_name}")

            model = model(image_height=dataset_metadata["image_height"], image_width=dataset_metadata["image_width"],
                          n_channels=dataset_metadata["n_channels"], n_classes=dataset_metadata["n_classes"],
                          batch_size=batch_size, checkpoint_path=MODEL_DIR_BY_KEY[model_name])

            predictions = model.predict(input_dir=filepath, logger=_logger)
            predictions["model"] = model_name

            dataset_labels_path = os.path.join(original_data_root, dataset_name,
                                               "{}_labels.csv".format(dataset_name.lower()))
            dataset_true_labels = pd.read_csv(dataset_labels_path, dtype={"image_id": str, "label": str}) \
                .rename(columns={"image_name": "image_id", "label": "label_true"})

            predictions = predictions.set_index("image_id")\
                                     .join(dataset_true_labels.set_index("image_id"), sort=False)\
                                     .reset_index(drop=False)
            # predictions.loc[predictions['targeted'].eq(False), "label_target"] = np.nan

            attack_model_scores = score_attacks_models(predictions, attack_model_scores)
            attack_model_scores.to_csv(os.path.join(output_dir, "attack_model_scores.csv"), index=False)


if __name__ == '__main__':
    _logger = set_up_logger(__name__)
    parser = argparse.ArgumentParser()
    parser.add_argument("-original_data_root", required=True,
                        help="path to the root containing the original data directories")
    parser.add_argument("-adversarial_data_root", required=True,
                        help="path to the root containing the adversarial data directories")
    parser.add_argument("-output_dir", required=True,
                        help="path to the output directory")
    parser.add_argument("-batch_size", type=int, required=False, default=50,
                        help="batch size")
    args = parser.parse_args()
    original_data_root = args.original_data_root
    if not os.path.isdir(original_data_root):
        raise Exception(f"Attention: \"{original_data_root}\" path does not exist!")
    adversarial_data_root = args.adversarial_data_root
    if not os.path.isdir(adversarial_data_root):
        raise Exception(f"Attention: \"{adversarial_data_root}\" path does not exist!")
    output_dir = args.output_dir
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    defend(original_data_root=original_data_root,
           adversarial_data_root=adversarial_data_root,
           output_dir=output_dir,
           batch_size=args.batch_size)
