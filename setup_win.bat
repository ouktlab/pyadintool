rem python3 -m venv venv
rem .\venv\Scripts\activate
python3 -m pip install --upgrade pip
python3 -m pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
python3 -m pip install numpy==1.26.4 pyyaml sounddevice huggingface_hub safetensors
python3 -m pip install PySide6 pyqtgraph
