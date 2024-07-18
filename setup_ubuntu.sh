# requirement: python version over 3.10 (required 3.9 for networkx used in torch, 3.10 for shiboken6 used in pyqtgraph)
python=python3.10
stage=0

if [ $stage -le 0 ]; then
  sudo apt update
  sudo apt install -y ${python}-venv emacs wavesurfer alsa-utils libasound2-dev libportaudio2 libxcb-cursor0 \
       libgl1-mesa-dev libfontconfig1 libxkbcommon-x11-0 \
       libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-render-util0 libxcb-xinerama0
fi

# 
if [ $stage -le 1 ]; then
  ${python} -m venv venv
  . venv/bin/activate
  ${python} -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
  ${python} -m pip install numpy pyyaml sounddevice huggingface_hub safetensors
  #exit;
fi

# for waveform-plot
if [ $stage -le 2 ]; then
  . venv/bin/activate
  ${python} -m pip install PySide6 pyqtgraph
fi
