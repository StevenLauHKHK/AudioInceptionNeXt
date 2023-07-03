export PYTHONPATH=/data1/steven/audio/AudioInceptionNeXt:$PYTHONPATH

python tools/run_net.py --cfg configs/VGG-Sound/AudioInceptionNeXt.yaml --init_method tcp://localhost:9996 \
NUM_GPUS 4 \
OUTPUT_DIR /data1/steven/audio/AudioInceptionNeXt/checkpoints/AudioInceptionNeXt_exp1 \
VGGSOUND.AUDIO_DATA_DIR /data_ssd/DATA/VGGSound/wav_sound \
VGGSOUND.ANNOTATIONS_DIR /data1/steven/audio/AudioInceptionNeXt




