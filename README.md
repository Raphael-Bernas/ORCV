## ORCV : Assignment 3 - Raphael BERNAS
You can test the following models :

1. `Net`: A basic convolutional neural network model. - *basic_cnn*
2. `BetterNet` : A multilayer convolutional neural network model. - *better_cnn*
3. `EfficientNetB7`: A larger version of EfficientNet optimized for better performance. - *efficient_net*
4. `DINOv2Model`: A model using DINOv2-small. - *dinov2*
5. `DINOv2LModel`: A model using DINOv2-large. - *dinov2L*
6. `DINOv2XLModel`: A model using DINOv2-giant. - *dinov2XL*
7. `DeiTModel`: A model using DeiT-base. - *deit*

And you can test different training method for those : 

1. `Adversarial` : Perturb the data to increase robustness [-- training_method ADV] [-- epsilon 0.01]
- `Gaussian` : Add a gaussian noise [-- attack_method Gaussian]
- `FGSM` : Add a noise based on gradient sign [-- attack_method FGSM]
- `PGD` : Add a perturbation projected for gradient descent [-- attack_method PGD]

2. `FLIP` : Regularization training on the lipschitz constant [-- training_method FLIP] [-- lamda 0.01]















## ORIGINAL READ ME :
## Object recognition and computer vision 2024/2025
This is the code from https://github.com/willowsierra/recvis24_a3


### Assignment 3: Sketch image classification
[![Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1PxshEMwNm4tLu8f_Bz_Z0emUlC1TPob4?usp=sharing)
#### Requirements
1. Install PyTorch from http://pytorch.org

2. Run the following command to install additional dependencies

```bash
pip install -r requirements.txt
```

#### Dataset
We will be using a dataset containing 500 different classes of sketches adapted from the [ImageNet Sketch](https://github.com/HaohanWang/ImageNet-Sketch).
Download the training/validation/test images from [here](https://www.kaggle.com/competitions/mva-recvis-2024/data). The test image labels are not provided.

#### Training and validating your model
Run the script `main.py` to train your model.

Modify `main.py`, `model.py` and `data.py` for your assignment, with an aim to make the validation score better.

- By default the images are loaded and resized to 64x64 pixels and normalized to zero-mean and standard deviation of 1. See data.py for the `data_transforms`.

- When changing models, you should also add support for your model in the `ModelFactory` class in `model_factory.py`. This allows to not having to modify the evaluation script after the model has finished training.

#### Evaluating your model on the test set

As the model trains, model checkpoints are saved to files such as `model_x.pth` to the current working directory.
You can take one of the checkpoints and run:

```
python evaluate.py --data [data_dir] --model [model_file] --model_name [model_name]
```

That generates a file `kaggle.csv` that you can upload to the private kaggle competition website.


#### Logger

We recommend you use an online logger like [Weights and Biases](https://wandb.ai/site/experiment-tracking) to track your experiments. This allows to visualise and compare every experiment you run. In particular, it could come in handy if you use google colab as you might easily loose track of your experiments when your sessions ends.

Note that currently, the code does not support such a logger. It should be pretty straightforward to set it up.

#### Acknowledgments
Adapted from Rob Fergus and Soumith Chintala https://github.com/soumith/traffic-sign-detection-homework.<br/>
Origial adaptation done by Gul Varol: https://github.com/gulvarol<br/>
New Sketch dataset and code adaptation done by Ricardo Garcia and Charles Raude: https://github.com/rjgpinel, http://imagine.enpc.fr/~raudec/
