# Pyadintool #

Pyadintool is a pre-processing toolkit covering voice activity detection, recording, splitting and sending of audio stream. 
This toolkit has been developed as a simple python clone of [adintool](https://github.com/julius-speech/julius/blob/master/adintool/README.md) in [Julius](https://github.com/julius-speech/julius) Japanese ASR. 

## Key Features ##
### Interface ###
* Real-time processing on CPU using multi-threading (requires at least two or three cores)
* Support sending segmented audio data to servers, such as Julius
* Support long recording and saving it to files
* Support gui plot of results
* Support batch processing using filelist

### Supported Voice Activity Detection (VAD) ###
* Simple Power-based VAD in time-domain
* VAD based on machine learning (ML)
    * Block-wise transformer-encoder VAD with HMM in STFT domain

### ML-VAD oriented for spoken dialogue system in real environments ###
* Scale-invariant processing: less influenced by the gain setting of microphone
    * Good portability
* Robustness against disturbances (non-speech and noise signals)
    * Suitable for noisy environments

### Limitations of current ML-based VAD model ###
* Language: Japanese is main target (due to the training data)
    * But it's expected to work for different languages to some extent because VAD does not use language information explicitly
* Number of speakers: only one (single speaker)
* Audio conditions: noisy input
    * Unsuitable for clean input
* Detection tendency
    * The latter part of long vowels tend nto to be detected
    * Cough is sometimed detected (it was not included in training data)


## Installation ##
### Supported System Requirements ###
* Ubuntu (22.04)
    * Windows WSL is avairable
    * alsa-utils
    * libasound2-dev
    * libportaudio2
    * ..., and other GUI and audio libraries
* Python3.10 (Python3.9 if gui plot is not used) and libraries
    * torch
    * torchaudio
    * numpy
    * pyyaml
    * sounddevice
    * huggingface_ub
    * savetensors
    * pyqtgraph (for realtime plot)
    * PySide6 (for realtime plot)

### Setup ###
* Install libraries automatically for ubuntu environment using "apt" and "pip install" commands 
```
sh setup.sh
```

* We have not specified required Ubuntu libaries. Therefore, some unnecessary libaries may be installed by "apt install" in the setup.sh. 

## Usage and Examples ##
### Run default setting ###
* Pyadintool requires configuration file for run
```
python3 pyadintool.py [conf]
```
* Check avairable sound device in advance
```
python3 pyadintool.py devinfo
``` 
* Run adinnet server (example) before running pyadintool.py. This server receive segmented audio data from pyadintool.py. 
```
python3 lib/egs.server.py
``` 

* Use default configuration using ML-VAD: input stream is mic, output stream is adinnet, sampling frequency is 16k Hz, and the number of mic. channels is 1. 
```
python3 pyadintool.py conf/default.yaml
``` 

* Switch to power-and-thresholding VAD configuration
```
python3 pyadintool.py conf/power.yaml
``` 

* Another choice to check the "adinnet" function, run Julius ASR with adinnet mode as follows 
```
sudoapt install git-lfs
git lfsinstall
git clone https://github.com/julius-speech/dictation-kit
cd dictation-kit
sh run-linux-dnn.sh -input adinnet -adport 5530
```


### Example-01: Input from file  ###
```
python3 pyadintool.py conf/default.yaml --in file
```
```
echo auido.wav | python3 pyadintool.py conf/default.yaml --in file
```

### Example-02: Save segmented audio to files  ###
```
python3 pyadintool.py conf/default.yaml --out file
```
```
python3 pyadintool.py conf/default.yaml --out file --filename segs/result_%Y%m%d_%H%M_%R.wav --startid 0
```

### Example-03: Send segmented audio to servers  ###
```
python3 pyadintool.py conf/default.yaml --out adinnet
```
```
python3 pyadintool.py conf/default.yaml --out adinnet --server localhost --port 5530
```
```
python3 pyadintool.py conf/default.yaml --out adinnet --server localhost,l92.168.1.30 --port 5530,5530
```

### Example-04: Set multiple outputs ###
```
python3 pyadintool.py conf/default.yaml --out adinnet-file
```

### Example-05: Save timesamps of VAD to a file  ###
```
python3 pyadintool.py conf/default.yaml --enable_timestamp --timestampfile result.lab
```

### Example-06: Logging  ###
```
python3 pyadintool.py conf/default.yaml --enable_logsave
```
```
python3 pyadintool.py conf/default.yaml --enable_logsave --logfilefmt log_%Y%d%m.log
```

### Example-07: Filelist batch processing ###
```
python3 pyadintool.py conf/default.yaml --enable_list --inlist wavlist.txt --tslist tslist.txt
```

Filenames of Audio and label data are listed in "wavlist.txt" and "tslist.txt"
```
data001.wav
data002.wav
```
```
data001.lab
data002.lab
```


### Example-08: Set input device name ###
```
python3 pyadintool.py conf/default.yaml --device default
```

### Example-09: Realtime plot ###
```
python3 pyadintool.py conf/default.yaml --enable_plot
```

![figure of realtime plot](images/fig_realtimeplot.png)

## Options ##
### --in [IN] ###
* Set input stream. "mic" or "file".
### --out [OUT] ### 
* Set output stream. "file", "adinnet" and both "adinnet-file"
* The data format of "adinnet"
    * send("bytes of data (int: 4-byte)")
    * send("binary data")
    * send(0 (int: 4-byte)): for end of segment


### --filename [FILENAME] ###

## Citations ##
