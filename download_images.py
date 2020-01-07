import os
import pandas as pd
from imageio import imsave
import tensorflow as tf
import tensorflow_datasets as tfds
import json
import argparse
from imagenet_downloader import download_images as imagenet
from utils.utils import set_up_logger
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
tf.compat.v1.enable_eager_execution()


def download(dataset_name_original, split, download_dir, n_images_to_download=100):
    tf.compat.v1.logging.info(f"Downloading {n_images_to_download} images from the \"{dataset_name_original}\" data set...")

    if not os.path.isdir(download_dir):
        os.makedirs(download_dir)

    if str.lower(dataset_name_original) == 'imagenet':
        imagenet.download_images(download_dir=download_dir, n_images_to_download=n_images_to_download, logger=_logger)
    else:
        # Construct a tf.data.Dataset
        dataset, info = tfds.load(name=dataset_name_original.lower(), split=split, with_info=True)

        height, width, n_channels = info.features['image'].shape
        n_classes = info.features['label'].num_classes
        sample_size = info.splits[split].num_examples

        # Build your input pipeline
        dataset = dataset.shuffle(sample_size)\
                         .prefetch(tf.data.experimental.AUTOTUNE)

        labels = pd.DataFrame(columns=['image_name', 'label'])
        dataset_name = os.path.basename(download_dir)
        image_counter = 0
        for features in dataset.take(n_images_to_download):
            image, label = features["image"], features["label"]
            image_counter += 1
            image_name = "{}_{:03d}".format(dataset_name, image_counter)
            imsave(os.path.join(download_dir, image_name + ".png"), image, format='PNG')
            labels = labels.append({'image_name': image_name, 'label': label.numpy()}, ignore_index=True, sort=False)

        labels.to_csv(os.path.join(download_dir, "{}_labels.csv".format(dataset_name)), index=False)

        with open(os.path.join(download_dir, "{}_metadata.json".format(dataset_name)), 'w') as f:
            metadata = {
                'image_height': height, 'image_width': width, 'n_channels': n_channels, 'n_classes': n_classes
            }
            json.dump(metadata, f)


if __name__ == '__main__':
    _logger = set_up_logger(__name__)
    parser = argparse.ArgumentParser()
    parser.add_argument("-dataset_name_original", required=True,
                        help="the original name of the dataset to download")
    parser.add_argument("-split", required=False, help='name of the split of the data to download')
    parser.add_argument("-output_dir", required=True,
                        help="path to the directory where to save data")
    parser.add_argument("-n_images", type=int, required=False, default=100,
                        help="number of images to download")
    args = parser.parse_args()
    download(dataset_name_original=args.dataset_name_original, split=args.split, download_dir=args.output_dir,
             n_images_to_download=args.n_images)
