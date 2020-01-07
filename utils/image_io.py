"""Implementation of sample attack."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from glob import glob
import numpy as np
import pandas as pd

from imageio import imread, imsave


class ImageIO:
    @classmethod
    def read(cls, input_dir, batch_size, image_height, image_width, n_channels, recursive=False, exclude_dir=None):
        """Read png images from input directory in batches.

      Args:
        input_dir: input directory
        batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

      Yields:
        filenames: list file names without path of each image
          Length of this list could be less than batch_size, in this case only
          first few images of the result are elements of the minibatch.
        images: array with all images from this batch
        :param exclude_dir:
        :param recursive:
        :param input_dir:
        :param batch_size:
        :param image_height:
        :param image_width:
        :param n_channels:
      """
        exclusions = set(os.listdir(exclude_dir)) if exclude_dir is not None else set()
        image_shape = [image_height, image_width, n_channels]
        batch_shape = [0] + image_shape
        images = np.empty(batch_shape)
        info = pd.DataFrame()
        images_counter = 0

        path_pattern = input_dir
        if recursive:
            path_pattern = os.path.join(path_pattern, "**")
        path_pattern = os.path.join(path_pattern, "*.png")
        for filepath in glob(path_pattern, recursive=recursive):
            if os.path.basename(filepath) in exclusions:
                continue
            image = np.reshape(imread(filepath,  # {'pilmode': 'RGB'} if n_channels == 3 else {}
                                      ).astype(np.float), [1] + image_shape)
            images = np.append(images, cls.transform_image(image), axis=0)
            image_file_name = os.path.basename(filepath)
            image_name = os.path.splitext(image_file_name)[0]
            single_info = pd.DataFrame({"image_id": [image_name]})
            single_info['image_id'] = single_info['image_id'].astype(str)

            # if recursive is True:
            try:
                is_targeted, attack_name, dataset_name, tool_name = os.path.basename(os.path.dirname(filepath))\
                                                                      .split("_")
                is_targeted = True if is_targeted == "targeted" else False
                single_info["targeted"] = is_targeted
                single_info["targeted"] = single_info["targeted"].astype(bool)
                single_info["attack"] = attack_name
                single_info["attack"] = single_info["attack"].astype(str)
                single_info["tool"] = tool_name
                single_info["tool"] = single_info["tool"].astype(str)
                single_info["dataset"] = dataset_name
                single_info["dataset"] = single_info["dataset"].astype(str)
            except:
                pass

            info = info.append(single_info, ignore_index=True, sort=False)
            images_counter += 1
            if images_counter == batch_size:
                yield images, info
                info = pd.DataFrame()
                images = np.empty(batch_shape)
                images_counter = 0
        if images_counter > 0:
            yield images, info

    @classmethod
    def write(cls, images, filenames, output_dir):
        """Saves images to the output directory.

      Args:
        images: array with minibatch of images
        filenames: list of filenames without path
          If number of file names in this list less than number of images in
          the minibatch then only first len(filenames) images will be saved.
        output_dir: directory where to save images
      """
        for i, filename in enumerate(filenames):
            output_path = os.path.join(output_dir, filename)
            image = cls.inverse_transform_image(images[i, :, :, :]).astype(np.uint8)
            if output_path[-4:] != ".png":
                output_path += ".png"
            imsave(output_path, image, format='PNG')

    @classmethod
    def transform_image(cls, image):
        return image / 255.

    @classmethod
    def inverse_transform_image(cls, image):
        return image * 255.
