import csv

import os
import numpy as np
from glob import glob
from imageio import imread
import tensorflow as tf
import logging


def load_defense_output(filename):
    """Loads output of defense from given file."""
    result = {}
    with open(filename) as f:
        for row in csv.reader(f):
            try:
                image_filename = row[0]
                if image_filename.endswith('.png') or image_filename.endswith('.jpg'):
                    image_filename = image_filename[:image_filename.rfind('.')]
                label = int(row[1])
            except (IndexError, ValueError):
                continue
            result[image_filename] = label
    return result


def calc_mean_perturbation_magnitude(image_dir, dataset_dir, logger):
    distances = []
    paths = glob(os.path.join(image_dir, "*.png"))
    sample_size = len(paths)
    for i, filepath in enumerate(paths):
        filename = os.path.basename(filepath)
        clean_image = imread(os.path.join(dataset_dir, filename)) / 255.
        adv_image = imread(filepath) / 255.
        distance = np.mean(np.square(clean_image - adv_image))
        distances += [distance]
        if ((i + 1) % 50) == 0 or (i + 1) == sample_size:
            logger.info("{}/{} current mean={:.0E}".format(i + 1, sample_size, np.mean(distances)))
    return np.mean(distances)


def score_attacks_models(predictions, starting_attack_scores):
    """
  Args:
      :param predictions: outputs of defenses. Dictionary of dictionaries, key in
      outer dictionary is name of the defense, key of inner dictionary is
      name of the image, value of inner dictionary is classification label.
      :param starting_attack_scores:
  """
    predictions['labels_pred_eq_true'] = predictions['label_pred'].eq(predictions['label_true'])
    # predictions['n_labels_pred_eq_target'] = predictions['label_pred'].eq(predictions['label_target'])
    predictions['labels_pred_eq_true'] = predictions['labels_pred_eq_true'].astype(int)
    # predictions['n_labels_pred_eq_target'] = predictions['n_labels_pred_eq_target'].astype(int)

    attack_model_scores = predictions.groupby(['dataset', 'targeted', 'attack', 'tool', 'model'], sort=False) \
                                     .agg(n_labels_pred_eq_true=('labels_pred_eq_true', 'sum'),
                                          # n_labels_pred_eq_target=('n_labels_pred_eq_target', 'sum'),
                                          n_samples=('image_id', 'count'))\
                                     .reset_index(drop=False)

    # attack_model_scores.loc[attack_model_scores['targeted'].eq(False), 'n_labels_pred_eq_target'] = np.nan

    if starting_attack_scores is not None:
        attack_model_scores = starting_attack_scores.append(attack_model_scores, sort=False)

    attack_model_scores = attack_model_scores[["dataset", "attack", "targeted", "model", "tool", "n_samples",
                                               "n_labels_pred_eq_true"
                                               # 'n_labels_pred_eq_target'
                                               ]]

    attack_model_scores = attack_model_scores.sort_values(by=["dataset", "attack", "targeted", "model", "tool"])

    return attack_model_scores


def score_attacks(attack_model_scores):
    """
  Args:
      :param attack_model_scores: outputs of defenses. Dictionary of dictionaries, key in
      outer dictionary is name of the defense, key of inner dictionary is
      name of the image, value of inner dictionary is classification label.
  """
    attack_scores = attack_model_scores.groupby(['dataset', 'targeted', 'attack', 'tool'], sort=False) \
                                       .agg(n_labels_pred_eq_true=('n_labels_pred_eq_true', 'sum'),
                                            # n_labels_pred_eq_target=('n_labels_pred_eq_target', 'sum'),
                                            n_models=('model', 'count'),
                                            n_samples=('n_samples', 'sum')) \
                                       .reset_index(drop=False)

    attack_scores['n_labels_pred_ne_true'] = attack_scores['n_samples'] - attack_scores['n_labels_pred_eq_true']
    # attack_scores.loc[attack_scores['targeted'].eq(False), 'n_labels_pred_eq_target'] = np.nan

    attack_scores = attack_scores[["dataset", "attack", "targeted", "tool", "n_models", "n_samples",
                                   "n_labels_pred_ne_true",
                                   # 'n_labels_pred_eq_target'
                                   ]]
    attack_scores = attack_scores.sort_values(by=["dataset", "attack", "targeted", "tool"])
    return attack_scores


def score_models(attack_model_scores):
    """
  Args:
      :param attack_model_scores: outputs of defenses. Dictionary of dictionaries, key in
      outer dictionary is name of the defense, key of inner dictionary is
      name of the image, value of inner dictionary is classification label.
  """
    attack_model_scores['targeted'] = attack_model_scores['targeted'].astype(int)
    attack_model_scores['n_targeted_samples'] = attack_model_scores['n_samples'] * attack_model_scores['targeted']
    model_scores = attack_model_scores.groupby(['dataset', 'tool', 'model'], sort=False)\
                                      .agg(n_labels_pred_eq_true=('n_labels_pred_eq_true', 'sum'),
                                           # n_labels_pred_eq_target=('n_labels_pred_eq_target', 'sum'),
                                           n_attacks=('attack', 'count'),
                                           n_targeted_attacks=('targeted', 'sum'),
                                           n_samples=('n_samples', 'sum'),
                                           n_targeted_samples=('n_targeted_samples', 'sum'))\
                                      .reset_index(drop=False)

    # model_scores.loc[model_scores['n_targeted_samples'] > 0, 'n_labels_pred_ne_target'] = \
    #     model_scores.loc[model_scores['n_targeted_samples'] > 0, 'n_targeted_samples'] \
    #     - model_scores.loc[model_scores['n_targeted_samples'] > 0, 'n_labels_pred_eq_target']
    # model_scores.loc[model_scores['n_targeted_samples'] == 0, 'n_labels_pred_ne_target'] = np.nan

    model_scores = model_scores[["dataset", "model", "tool", "n_attacks", "n_targeted_attacks",
                                 "n_samples", "n_targeted_samples", "n_labels_pred_eq_true",
                                 # 'n_labels_pred_ne_target'
                                ]]

    model_scores = model_scores.sort_values(by=["dataset", "model", "tool"])
    return model_scores


def load_checkpoint(checkpoint_path):
    if os.path.isdir(checkpoint_path):
        loaded_checkpoint = tf.compat.v1.train.latest_checkpoint(checkpoint_path)
        if loaded_checkpoint is None:
            found_checkpoints = glob(os.path.join(checkpoint_path, "*.ckpt"))
            if len(found_checkpoints) > 0:
                loaded_checkpoint = found_checkpoints[0]
            else:
                raise Exception("Attention: \"{}\" folder does not contain checkpoints!".format(checkpoint_path))
    elif os.path.isfile(checkpoint_path):
        loaded_checkpoint = checkpoint_path
    else:
        loaded_checkpoint = checkpoint_path
    return loaded_checkpoint


def set_up_logger(logger_name):
    # create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    # create console handler and set level to debug
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)

    # create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # add formatter to ch
    ch.setFormatter(formatter)

    # add ch to logger
    logger.addHandler(ch)

    return logger
