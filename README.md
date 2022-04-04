# 3d-neuro-seg
Repository to our 3d neuro-anatomic segmentation method presented at the MIDL 2022 conference

Roy S*, Kügler D*, Reuter M (*equal contribution). Are 2.5D approaches superior to 3D deep networks in whole brain segmentation? In: Medical Imaging with Deep Learning. Proceedings of Machine Learning Research (2022), [PDF](https://openreview.net/forum?id=Ob62JPB_CDF), [Bibtex](res/citation.bib)

## Configs
The params dictionary for each model is provided in `models/config.py`

## Training

- Run `train_model.py` to train a model.

  `CUDA_VISIBLE_DEVICES=0,1 python train_model.py`. 2 GPUs are necessary if you have a separate `loss_device`.

- Even though the micro/block architecture file (eg. VNet.py) is copied into the checkpoints folder, it is for reference. 
During Evaluation the micro/block architectures are taken from ./model/inference. This is to ensure compatibility with older trained models. Fix this if you are 
changing the micro/block archs.

## Evaluation of final architectures

### Names of model architectures
For evaluation, users should use `eval_w_mgz.py`. See documentation in the help.

| Name in Paper             | Class Name   | Source File                 | Weights File                |
|---------------------------|--------------|-----------------------------|-----------------------------|
| Ours                      | QuadNet      | `model/QuadNet.py`          | `ensembled_model_<index>`   |
| Ours (w/o SL)             | RCVNet       | `model/inference/RCVNet.py` | `unensembled_model`         |

### Model files corresponding to paper
While we have included code to automatically download model files, these model files can also be [manually downloaded](https://b2share.fz-juelich.de/records/67dfccf54c75492388f038128aa4c687).


# Evaluating old archived models
These architectures did not provide sufficient performance improvements over the chosen architectures (see above).

## Architecture modifications 

| Model Description (Backbone VNet with ...)                                                        | Source File               |
|---------------------------------------------------------------------------------------------------|---------------------------|
| GroupNorm and 3^3 Kernels instead of 5^3                                                          | `model/RCVNet.py`         |
| Symmetric filter factorization with two 3^3 kernels instead of 5^3                                | `model/VNetSym.py`        |
| Asymmetric filter factorization with two sets of (3x1x1), (1x3x1), (1x1x3) filters instead of 5^3 | `model/VNetAsym.py`       |
| Dense residual block as in DenseNet, with additive residuals                                      | `model/VNetDenseAdd.py`   |
| Attention Block as in Attention UNet                                                              | `model/inference/VNet.py` |
| Squeeze and Excite block as in SqueezeNet                                                         | `model/VNetSE.py`         |
| Concurrent 3 axis 2.5D and 3D paths with 75% of feature maps in 3D path                           | `model/VNet_2D_3D.py`     |
| only 75% of feature maps included in each block                                                   | `model/MultiResVNet.py`   |

Checkpoint directory:
1. Copy the model folder to evaluate into your present working 3dseg's checkpoint folder. Also, change the absolute path to the checkpoints folder to the one inside your present working 3dseg checkpoint folder and create a `__init__.py` (if it does not already exist). This is because model classes will be directly read from here
2. Go into the right file containing the model macro architecture (as specified by `file_string`) named RCVNet.py / CropNet3d.py / LResFCNet.py inside the copied folder and change the import on line 5 from model.module to .module (basically forcing it to use the `module.py` file inside this folder as the layer/micro architecture definition). `module.py` is the common name I gave to VNet.py, VNet_2D_3D.py etc when I copied them over during training, e.g.:
    - OLD: `from model.module import EncoderBlock, BottleNeck, DecoderBlock `
    - NEW: `from .module import EncoderBlock, BottleNeck, DecoderBlock` 
3. Select the correct file and class name in the eval*.py file and run.

## Notes

1. There is no named retrieval of classes for quadnet eval. Please make sure that a standard vnet is retrieved from the `./models` folder. Use the models in the `./models/inference` folder if necessary - I created that folder to keep copies incase one edited the py files inside the models folder.
2. Please find details of the model folders in the 3dseg-saved-models.xlsx file