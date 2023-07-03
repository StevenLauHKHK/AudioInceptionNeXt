export PYTHONPATH=/data1/steven/audio/AudioInceptionNeXt:$PYTHONPATH

CUDA_VISIBLE_DEVICES=0 python tools/run_net.py --cfg configs/EPIC-SOUND-416x128/AudioInceptionNeXt.yaml --init_method tcp://localhost:9997 \
NUM_GPUS 1 \
OUTPUT_DIR /data1/steven/audio/AudioInceptionNeXt/eval_epic_sound_416x128/AudioInceptionNeXt/epoch_30 \
EPICSOUND.AUDIO_DATA_FILE /data_ssd/DATA/EPIC-Kitchens-100-hdf5/EPIC-KITCHENS-100_audio.hdf5 \
EPICSOUND.ANNOTATIONS_DIR /data1/steven/audio/AudioInceptionNeXt \
TRAIN.ENABLE False \
TEST.ENABLE True \
TEST.CHECKPOINT_FILE_PATH /data1/steven/audio/AudioInceptionNeXt/checkpoints_epic_sounds/AudioInceptionNeXt/checkpoints/checkpoint_epoch_00030.pyth