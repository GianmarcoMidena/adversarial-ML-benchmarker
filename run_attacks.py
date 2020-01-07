from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import os
import pathlib
import json
import argparse
from itertools import product
import traceback
from utils.utils import set_up_logger


from utils.normalized_image_io import NormalizedImageIO
from tools.tool_finder import find_tool

MODEL_CHECKPOINTS_ROOT = "model checkpoints"
MADRY_CKPT_PATH = os.path.join(MODEL_CHECKPOINTS_ROOT, "madry_ckpt")
assert (os.path.isdir(MADRY_CKPT_PATH))
RESNET_CKPT_PATH = os.path.join(MODEL_CHECKPOINTS_ROOT, "wide_resnet_ckpt")
assert (os.path.isdir(RESNET_CKPT_PATH))
INCEPTION_V3_CKPT_PATH = os.path.join(MODEL_CHECKPOINTS_ROOT, "inception_v3_ckpt")
assert (os.path.isdir(INCEPTION_V3_CKPT_PATH))
MODEL_CKPTS_BY_DATA_SET = {
    "mnist": MADRY_CKPT_PATH,
    "cifar10": RESNET_CKPT_PATH,
    "imagenet": INCEPTION_V3_CKPT_PATH
}


def attack(attack_methods, tools, data_sets, batch_size, input_data_root, output_data_root, model_ckpt_path):
    for data_set_name in data_sets:
        try:
            _attack_data_set(attack_methods, batch_size, data_set_name, input_data_root, model_ckpt_path,
                             output_data_root, tools)
        except:
            traceback.print_exc()


def _attack_data_set(attack_methods, batch_size, data_set_name, input_data_root, model_ckpt_path, output_data_root,
                     tools):
    if model_ckpt_path is None:
        model_ckpt_path = _find_default_model_checkpoint_path(data_set_name)
    data_set_path = os.path.join(input_data_root, data_set_name)
    image_height, image_width, n_channels, n_classes = _extract_data_set_info(data_set_name, data_set_path)
    for tool_name, attack_name in product(tools, attack_methods):
        try:
            _attack_data_set_by_tool_and_method(attack_name, batch_size, data_set_name, data_set_path, image_height,
                                                image_width, model_ckpt_path, n_channels, n_classes, output_data_root,
                                                tool_name)
        except:
            traceback.print_exc()


def _attack_data_set_by_tool_and_method(attack_name, batch_size, data_set_name, data_set_path, image_height,
                                        image_width, model_ckpt_path, n_channels, n_classes, output_data_root,
                                        tool_name):
    tool = find_tool(tool_name)
    attack = tool.create_attack(attack_name=attack_name, dataset_name=data_set_name,
                                image_height=image_height, image_width=image_width, n_channels=n_channels,
                                n_classes=n_classes, checkpoint_path=model_ckpt_path)
    tool_name, attack_name = attack.get_name().split("_")
    attack_dir = "{}_{}_{}_{}".format("untargeted", attack_name, data_set_name, tool_name)
    _logger.info(f"Executing an {attack_dir} attack...")
    output_path = os.path.join(output_data_root, attack_dir)
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)
    for images, info in _read_images(input_dir=data_set_path,
                                     image_height=image_height, image_width=image_width, n_channels=n_channels,
                                     exclude_dir=output_path, batch_size=batch_size):
        start = time.time()
        adv_images = attack.execute(images)
        end = time.time()
        _logger.info("{:.2f} sec".format(end - start))
        _write(adv_images, info['image_id'].to_list(), output_dir=output_path)


def _read_images(input_dir, batch_size, image_height, image_width, n_channels, exclude_dir=None):
    return NormalizedImageIO.read(input_dir, batch_size=batch_size, image_height=image_height,
                                  image_width=image_width, n_channels=n_channels, exclude_dir=exclude_dir)


def _write(perturbed_images, filenames, output_dir):
    NormalizedImageIO.write(perturbed_images, filenames, output_dir)


def _find_default_model_checkpoint_path(data_set_name):
    if data_set_name.lower() in MODEL_CKPTS_BY_DATA_SET:
        return MODEL_CKPTS_BY_DATA_SET[data_set_name.lower()]
    raise Exception(f"Attention: \"{data_set_name}\" is not an available data set!")


def _extract_data_set_info(data_set_name, data_set_path):
    with open(os.path.join(data_set_path, "{}_metadata.json".format(data_set_name.lower())), 'r') as f:
        dataset_metadata = json.load(f)
    image_height = dataset_metadata['image_height']
    image_width = dataset_metadata['image_width']
    n_channels = dataset_metadata['n_channels']
    n_classes = dataset_metadata['n_classes']
    return image_height, image_width, n_channels, n_classes


if __name__ == '__main__':
    _logger = set_up_logger(__name__)
    parser = argparse.ArgumentParser()
    parser.add_argument("-attack_methods", nargs='+', required=True,
                        help="adversarial attack methods")
    parser.add_argument("-tools", nargs='+', required=True,
                        help="tools for adversarial machine learning")
    parser.add_argument("-data_sets", nargs='+', required=True,
                        help="data sets")
    parser.add_argument("-batch_size", type=int, required=False, default=50,
                        help="batch size")
    parser.add_argument("-input_data_root", required=True,
                        help="path to the root containing the input data directories")
    parser.add_argument("-output_data_root", required=True,
                        help="path to the root containing the output data directories")
    parser.add_argument("-model_ckpt_path", required=False,
                        help="path to a model checkpoint")
    args = parser.parse_args()
    attack(attack_methods=args.attack_methods,
           tools=args.tools,
           data_sets=args.data_sets,
           batch_size=args.batch_size,
           input_data_root=args.input_data_root,
           output_data_root=args.output_data_root,
           model_ckpt_path=args.model_ckpt_path)