#!/bin/bash
# requirement: python version over 3.10
# (required 3.9 for networkx used in torch, 3.10 for shiboken6 used in pyqtgraph)

# change this configuration according to your environment
python=python3.10
stage=0
enable_espnet=true #false
enable_whisper=true #false

# 
if [ $stage -le 0 ]; then
    echo "-- stage 0 ------ "
    sudo apt update
    sudo apt install -y ${python}-venv emacs wavesurfer alsa-utils libasound2-dev libportaudio2 libxcb-cursor0 \
	 libgl1-mesa-dev libfontconfig1 libxkbcommon-x11-0 \
	 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 libxcb-xinerama0
fi

# 
if [ $stage -le 1 ]; then
    echo "-- stage 1 ------ "
    ${python} -m venv venv/main/
    . venv/main/bin/activate
    ${python} -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
    ${python} -m pip install numpy pyyaml sounddevice huggingface_hub safetensors
fi

# for waveform-plot
if [ $stage -le 2 ]; then
    echo "-- stage 2 ------ "
    . venv/main/bin/activate
    ${python} -m pip install PySide6 pyqtgraph
fi

# for silero vad
if [ $stage -le 3 ]; then
    echo "-- stage 3 ------ "
    . venv/main/bin/activate
    ${python} -m pip install silero-vad
fi

# for ESPnet install
if "${enable_espnet}"; then
    echo "-- espnet ------- "
    python3 -m venv venv/espnet/
    
    . venv/espnet/bin/activate
    python3 -m pip install espnet torchaudio
    python3 -m pip install -U espnet_model_zoo
fi

# for faster Whisper
if "${enable_whisper}"; then
    echo "-- whisper ------ "
    python3 -m venv venv/whisper
    . ./venv/whisper/bin/activate
    python3 -m pip install faster_whisper
fi
