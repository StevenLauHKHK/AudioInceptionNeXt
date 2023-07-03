# Auditory AudioInceptionNeXt

This repository implements the model proposed in the paper:

Kin Wai Lau, Yasar Abbas Ur Rehman, Yuyang Xie, Lan Ma, **AudioInceptionNeXt: TCL AI LAB Submission to EPIC-SOUND Audio-Based-Interaction-Recognition Challenge 2023**

[[arXiv paper]](Coming Soon)

The implementation code is based on the **Slow-Fast Auditory Streams for Audio Recognition**, ICASSP, 2021. For more information, please refer to the [link](https://github.com/ekazakos/auditory-slow-fast).


## Citing

When using this code, kindly reference:

```
Coming Soon
```



## Pretrained models

You can download our pretrained models on VGG-Sound and EPIC-Sounds:
- AudioInceptionNeXt (VGG-Sound) [link](https://portland-my.sharepoint.com/:u:/g/personal/kinwailau6-c_my_cityu_edu_hk/Ef-RPDEZYGpChEp4D8DDnBABrMQonXi309p24PTID1kWqQ)
- AudioInceptionNeXt (EPIC-Sound) [link](https://portland-my.sharepoint.com/:u:/g/personal/kinwailau6-c_my_cityu_edu_hk/ETftrpxDnq1Og_l2oCcKJ2cB2abpmw0JzlJS99wPKZao7A)

## Preparation

* Requirements:
  * [PyTorch](https://pytorch.org) 1.7.1
  * [librosa](https://librosa.org): `conda install -c conda-forge librosa`
  * [h5py](https://www.h5py.org): `conda install h5py`
  * [wandb](https://wandb.ai/site): `pip install wandb`
  * [fvcore](https://github.com/facebookresearch/fvcore/): `pip install 'git+https://github.com/facebookresearch/fvcore'`
  * simplejson: `pip install simplejson`
  * psutil: `pip install psutil`
  * tensorboard: `pip install tensorboard` 
* Add this repository to $PYTHONPATH.
```
export PYTHONPATH=/path/to/auditory-slow-fast/slowfast:$PYTHONPATH
```
* VGG-Sound:
  See the instruction in Auditory Slow-Fast repository [link](https://github.com/ekazakos/auditory-slow-fast)
* EPIC-KITCHENS:
  See the instruction in Auditory Slow-Fast repository [link](https://github.com/ekazakos/auditory-slow-fast)
* EPIC-Sounds
  See the instruction in Epic-Sounds annotations repository [link](https://github.com/epic-kitchens/epic-sounds-annotations) and [link](https://github.com/epic-kitchens/epic-sounds-annotations/tree/main/src)

## Training/validation on VGG-Sound
To train the model run:
```
python tools/run_net.py --cfg configs/VGG-Sound/AudioInceptionNeXt.yaml --init_method tcp://localhost:9996 \
NUM_GPUS num_gpus \
OUTPUT_DIR /path/to/output_dir \
VGGSOUND.AUDIO_DATA_DIR /path/to/dataset 
VGGSOUND.ANNOTATIONS_DIR /path/to/annotations 
```

To validate the model run:
```
python tools/run_net.py --cfg configs/VGG-Sound/AudioInceptionNeXt.yaml --init_method tcp://localhost:9998 \
NUM_GPUS num_gpus \
OUTPUT_DIR /path/to/experiment_dir \
VGGSOUND.AUDIO_DATA_DIR /path/to/dataset \
VGGSOUND.ANNOTATIONS_DIR /path/to/annotations \
TRAIN.ENABLE False \
TEST.ENABLE True \
TEST.CHECKPOINT_FILE_PATH /path/to/experiment_dir/checkpoints/checkpoint_best.pyth
```

## Fine Tune/validation on EPIC-Sounds
To fine-tuning from VGG-Sound pretrained model:
```
python tools/run_net.py --cfg configs/EPIC-SOUND-416x128/AudioInceptionNeXt.yaml --init_method tcp://localhost:9996 \
NUM_GPUS num_gpus \
OUTPUT_DIR /path/to/output_dir \
EPICSOUND.AUDIO_DATA_FILE /path/to/EPIC-KITCHENS-100_audio.hdf5 \
EPICSOUND.ANNOTATIONS_DIR /path/to/annotations \
TRAIN.CHECKPOINT_FILE_PATH /path/to/VGG-Sound/pretrained/model
```

To validate the model run:
```
python tools/run_net.py --cfg configs/EPIC-SOUND-416x128/AudioInceptionNeXt.yaml --init_method tcp://localhost:9997 \
NUM_GPUS num_gpus \
OUTPUT_DIR /path/to/experiment_dir \
EPICKITCHENS.AUDIO_DATA_FILE /path/to/EPIC-KITCHENS-100_audio.hdf5 \
EPICKITCHENS.ANNOTATIONS_DIR /path/to/annotations \
TRAIN.ENABLE False \
TEST.ENABLE True \
TEST.CHECKPOINT_FILE_PATH /path/to/experiment_dir/checkpoints/checkpoint_best.pyth
```



