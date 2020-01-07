# Adversarial machine learning tools benchmarker

## Use
1) download images (`./download_images.sh`)
2) download model checkpoints (`./download_model_checkpoints.sh`)
3) run attacks (`./run_attacks.sh`)
4) run defenses (`./run_defenses.sh`)
5) compute perturbation magnitude (`./computer_perturbation_magnitude.sh`)

## Supported attacks
**Evasion Attacks:**
* Fast gradient method (FGM) ([Goodfellow et al., 2014](https://arxiv.org/abs/1412.6572))
* DeepFool ([Moosavi-Dezfooli et al., 2015](https://arxiv.org/abs/1511.04599))
* Basic iterative method (BIM) ([Kurakin et al., 2016](https://arxiv.org/abs/1607.02533))
* Carlini & Wagner (C&W) ([Carlini and Wagner, 2016](https://arxiv.org/abs/1608.04644))
* Projected gradient descent (PGD) ([Madry et al., 2017](https://arxiv.org/abs/1706.06083))
* Jacobian saliency map ([Papernot et al., 2016](https://arxiv.org/abs/1511.07528))

## Supported tools
- [CleverHans](https://github.com/tensorflow/cleverhans)
- [Foolbox](https://github.com/bethgelab/foolbox)
- [Adversarial Robustness 360 Toolbox](https://github.com/IBM/adversarial-robustness-toolbox)

## Supported data sets
- [ImageNet](http://www.image-net.org/)
- [Cifar10](https://www.cs.toronto.edu/~kriz/cifar.html)
- [MNIST](http://yann.lecun.com/exdb/mnist/)

## Supported models
- Inception V3 ([Szegedy et al.](https://arxiv.org/abs/1512.00567))
- Wide Resnet ([Zagoruyco and Komodakis](https://arxiv.org/abs/1605.07146))
- [Madry et al.](https://arxiv.org/abs/1706.06083)

## Dependencies
- cleverhans 3.0.1
- foolbox 2.3.0
- adversarial-robustness-toolbox 1.0.1
- tensorflow 1.15.0
- tensorflow-datasets 1.3.2
- pandas 0.25.0
- numpy 1.18.0