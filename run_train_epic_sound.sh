export PYTHONPATH=/data1/steven/audio/AudioInceptionNeXt:$PYTHONPATH

CUDA_VISIBLE_DEVICES=0,1,2,3 python tools/run_net.py --cfg configs/EPIC-SOUND-416x128/AudioInceptionNeXt.yaml --init_method tcp://localhost:9996 \
NUM_GPUS 4 \
OUTPUT_DIR /data1/steven/audio/AudioInceptionNeXt/checkpoints_epic_sound_416x128/AudioInceptionNeXt_exp1 \
EPICSOUND.AUDIO_DATA_FILE /data_ssd/DATA/EPIC-Kitchens-100-hdf5/EPIC-KITCHENS-100_audio.hdf5 \
EPICSOUND.ANNOTATIONS_DIR  /data1/steven/audio/AudioInceptionNeXt \
TRAIN.CHECKPOINT_FILE_PATH /data1/steven/audio/AudioInceptionNeXt/checkpoints_vgg/AudioInceptionNeXt/checkpoints/checkpoint_epoch_00050.pyth



