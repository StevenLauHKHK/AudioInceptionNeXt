export PYTHONPATH=/data1/steven/audio/AudioInceptionNeXt:$PYTHONPATH

python tools/run_net.py --cfg configs/VGG-Sound/AudioInceptionNeXt.yaml --init_method tcp://localhost:9998 \
NUM_GPUS 4 \
OUTPUT_DIR /data1/steven/audio/AudioInceptionNeXt/eval/AudioInceptionNeXt/epoch_50 \
VGGSOUND.AUDIO_DATA_DIR /data_ssd/DATA/VGGSound/wav_sound \
VGGSOUND.ANNOTATIONS_DIR /data1/steven/audio/AudioInceptionNeXt \
TRAIN.ENABLE False \
TEST.ENABLE True \
TEST.CHECKPOINT_FILE_PATH /data1/steven/audio/AudioInceptionNeXt/checkpoints_vgg/AudioInceptionNeXt/checkpoints/checkpoint_epoch_00050.pyth
