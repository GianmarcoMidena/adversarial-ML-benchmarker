#!/bin/sh
#
# Scripts which download model checkpoints.
#

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Download inception v3 checkpoint
echo "Downloading a checkpoint of inception v3 model..."
rm -rf "${SCRIPT_DIR}/model checkpoints/inception_v3_ckpt/"
mkdir -p "${SCRIPT_DIR}/model checkpoints/inception_v3_ckpt/"
cd "${SCRIPT_DIR}/model checkpoints/inception_v3_ckpt/"
wget http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz
tar -xvzf inception_v3_2016_08_28.tar.gz
rm inception_v3_2016_08_28.tar.gz

# Download adversarially trained inception v3 checkpoint
echo "Downloading a checkpoint of an adversarially trained inception v3 model..."
rm -rf "${SCRIPT_DIR}/model checkpoints/adv_inception_v3_ckpt/"
mkdir -p "${SCRIPT_DIR}/model checkpoints/adv_inception_v3_ckpt/"
cd "${SCRIPT_DIR}/model checkpoints/adv_inception_v3_ckpt/"
wget http://download.tensorflow.org/models/adv_inception_v3_2017_08_18.tar.gz
tar -xvzf adv_inception_v3_2017_08_18.tar.gz
rm adv_inception_v3_2017_08_18.tar.gz

# Download trained wide resnet checkpoint
echo "Downloading a checkpoint of a trained wide resnet model..."
rm -rf "${SCRIPT_DIR}/model checkpoints/wide_resnet_ckpt/"
mkdir -p "${SCRIPT_DIR}/model checkpoints/wide_resnet_ckpt/"
cd "${SCRIPT_DIR}/model checkpoints/wide_resnet_ckpt/"
wget https://www.dropbox.com/s/cgzd5odqoojvxzk/natural.zip?dl=1 -O natural.zip
unzip -j natural.zip
rm natural.zip

# Download adversarially trained wide resnet checkpoint
echo "Downloading a checkpoint of a adversarially trained wide resnet model..."
rm -rf "${SCRIPT_DIR}/model checkpoints/adv_wide_resnet_ckpt/"
mkdir -p "${SCRIPT_DIR}/model checkpoints/adv_wide_resnet_ckpt/"
cd "${SCRIPT_DIR}/model checkpoints/adv_wide_resnet_ckpt/"
wget https://www.dropbox.com/s/g4b6ntrp8zrudbz/adv_trained.zip?dl=1 -O adv_trained.zip
unzip -j adv_trained.zip
rm adv_trained.zip

# Download madry checkpoint
echo "Downloading a checkpoint of madry model..."
rm -rf "${SCRIPT_DIR}/model checkpoints/madry_ckpt/"
mkdir -p "${SCRIPT_DIR}/model checkpoints/madry_ckpt/"
cd "${SCRIPT_DIR}/model checkpoints/madry_ckpt/"
wget https://github.com/MadryLab/mnist_challenge_models/raw/master/natural.zip
unzip -j natural.zip
rm natural.zip

# Download adversarially trained madry checkpoint
echo "Downloading a checkpoint of an adversarially trained madry model..."
rm -rf "${SCRIPT_DIR}/model checkpoints/adv_madry_ckpt/"
mkdir -p "${SCRIPT_DIR}/model checkpoints/adv_madry_ckpt/"
cd "${SCRIPT_DIR}/model checkpoints/adv_madry_ckpt/"
wget https://github.com/MadryLab/mnist_challenge_models/raw/master/adv_trained.zip
unzip -j adv_trained.zip
rm adv_trained.zip

# Download ensemble adversarially trained inception resnet v2 checkpoint
#echo "Downloading a checkpoint of an ensemble adversarially trained inception resnet v2 model..."
#rm -rf "${SCRIPT_DIR}/model checkpoints/ens_adv_inception_resnet_v2_ckpt/"
#mkdir -p "${SCRIPT_DIR}/model checkpoints/ens_adv_inception_resnet_v2_ckpt/"
#cd "${SCRIPT_DIR}/model checkpoints/ens_adv_inception_resnet_v2_ckpt/"
#wget http://download.tensorflow.org/models/ens_adv_inception_resnet_v2_2017_08_18.tar.gz
#tar -xvzf ens_adv_inception_resnet_v2_2017_08_18.tar.gz
#rm ens_adv_inception_resnet_v2_2017_08_18.tar.gz