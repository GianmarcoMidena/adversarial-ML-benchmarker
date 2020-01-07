from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import csv
import os
import sys
from io import BytesIO
import json
from PIL import Image
import pandas as pd
import random

try:
    from urllib.request import urlopen
except ImportError:
    from urllib2 import urlopen


DATASET_PATH = "imagenet_downloader/imagenet_dataset.csv"
IMG_WIDTH = 299
IMG_HEIGHT = 299
IMG_CHANNELS = 3
N_CLASSES = 1000
SEED = 3


def get_image(url, x1, y1, x2, y2):
    try:
        # Download image
        url_file = urlopen(url)
        if url_file.getcode() != 200:
            return False
        image_buffer = url_file.read()
        # Crop, resize and save image
        image = Image.open(BytesIO(image_buffer)).convert('RGB')
        w = image.size[0]
        h = image.size[1]
        image = image.crop((int(x1 * w), int(y1 * h), int(x2 * w),
                            int(y2 * h)))
        image = image.resize((IMG_WIDTH, IMG_HEIGHT), resample=Image.ANTIALIAS)
    except IOError:
        return None
    return image


def download_images(download_dir, n_images_to_download, logger):
    dataset_name = os.path.basename(download_dir)

    with open(DATASET_PATH) as input_file:
        reader = csv.reader(input_file)
        header_row = next(reader)
        rows = list(reader)
    try:
        row_idx_url = header_row.index('URL')
        row_idx_x1 = header_row.index('x1')
        row_idx_y1 = header_row.index('y1')
        row_idx_x2 = header_row.index('x2')
        row_idx_y2 = header_row.index('y2')
        row_idx_label = header_row.index('TrueLabel')
    except ValueError as e:
        logger.error('One of the columns was not found in the source file: ', e.message)

    rows = [{'url': row[row_idx_url],
             'x1': float(row[row_idx_x1]),
             'y1': float(row[row_idx_y1]),
             'x2': float(row[row_idx_x2]),
             'y2': float(row[row_idx_y2]),
             'label': int(row[row_idx_label])} for row in rows]

    image_counter = 0
    labels = pd.DataFrame(columns=['image_name', 'label'])
    indices = [i for i in range(len(rows))]
    random.seed(SEED)
    random.shuffle(indices)
    for idx in indices:
        row = rows[idx]
        image = get_image(url=row['url'], x1=row['x1'], y1=row['y1'], x2=row['x2'], y2=row['y2'])
        if image is not None:
            image_counter += 1
            image_name = "{}_{:03d}".format(dataset_name, image_counter)
            image.save(os.path.join(download_dir, image_name+".png"))
            labels = labels.append({'image_name': image_name, 'label': row['label']}, ignore_index=True, sort=False)
            if image_counter == n_images_to_download:
                break

    labels.to_csv(os.path.join(download_dir, "{}_labels.csv".format(dataset_name)), index=False)

    with open(os.path.join(download_dir, "{}_metadata.json".format(dataset_name)), 'w') as f:
        metadata = {
            'image_height': IMG_HEIGHT,
            'image_width': IMG_WIDTH,
            'n_channels': IMG_CHANNELS,
            'n_classes': N_CLASSES
        }
        json.dump(metadata, f)
