# Pyadintool #

Pyadintool is a pre-processing toolkit covering voice activity detection, recording, splitting and sending of audio stream. 
This toolkit has been developed as a simple python clone of [adintool](https://github.com/julius-speech/julius/blob/master/adintool/README.md) in [Julius](https://github.com/julius-speech/julius) Japanese ASR. 

## Key Features ##
### Interface ###
* Real-time processing on CPU using multi-threading (requires at least two or three cores)
* Support sending segmented audio data to servers
* Support long recording and saving it to files
* Support gui plot of results
* Support batch processing using filelist

### Supported Voice Activity Detection (VAD) ###
* Power-based VAD in time-domain
    * suitable for silent environment. e.g., head-set microphone.
* ML-based VAD in STFT domain
    * Scale-invariant 

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
### System Requirements ###
* Ubuntu 22.04 and Ubuntu 22.04 on Windows WSL
    * alsa-utils
    * libasound2-dev
    * libportaudio2
    * ..., and other GUI and audio libraries
* Windows 11
* Python3.10~3.12 (Python3.9 if gui plot is not used) and libraries
    * torch
    * torchaudio
    * numpy
    * pyyaml
    * sounddevice
    * huggingface_ub
    * savetensors
    * pyqtgraph (for realtime plot)
    * PySide6 (for realtime plot)

### Setup on Ubuntu ###
* Copy and edit shell script: change the python version
```
cp setup_ubuntu.sh setup_ubuntu_local.sh
```
* Install libraries automatically for ubuntu environment using "sudo apt install" and "pip install" commands 
```
sh setup_ubuntu_local.sh
```
* The above script also create "venv" environment (venv/) in the current directory. Activate venv when you run our python scripts.

* We have not specified required Ubuntu libaries. Therefore, some unnecessary libaries may be installed by "apt install" in the setup.sh. 

### Setup on Windows ###
* Create virtual environemnt
```
python3 -m venv venv
.\venv\Scripts\activate
```
* Install python libraries by using batch file
```
setup_win.bat
```
* Sometimes edit the batch file to change the python version

## Usage and Examples ##
### Run default setting ###
* Activate virtual environment
```
. venv/bin/activate
```
```
.\venv\Scripts\activate
```
* Pyadintool requires configuration file for run
```
python3 pyadintool.py [conf]
```
* Check avairable sound devices (device list) in advance. 
```
python3 pyadintool.py devinfo
``` 

* Use default configuration using ML-VAD: input stream is "mic", output stream is "file", sampling frequency is 16k Hz, and the number of mic. channels is 1. 
```
python3 pyadintool.py conf/default.yaml
``` 

* Change the audio device by using --device option in the case of Windows. The device ID must be selected from the device list. 
```
python3 pyadintool.py conf/default.yaml --device 1
``` 


* Switch to power-and-thresholding VAD configuration
```
python3 pyadintool.py conf/power.yaml
``` 

### Example-01: Set input stream to a audio file  ###
```
python3 pyadintool.py conf/default.yaml --in file
```
```
echo auido.wav | python3 pyadintool.py conf/default.yaml --in file
```

### Example-02: Save segmented audio signals to files  ###
```
python3 pyadintool.py conf/default.yaml --out file
```
```
python3 pyadintool.py conf/default.yaml --out file --filename segs/result_%Y%m%d_%H%M_%R.wav --startid 0
```

### Example-03: Send segmented audio signals to servers  ###
* Run adinnet server (example) before running pyadintool.py. This server receive segmented audio data from pyadintool.py. 
```
python3 lib/egs.server.py
``` 
* Or another choice to check the "adinnet" function, run Julius ASR with adinnet mode as follows 
```
sudoapt install git-lfs
git lfsinstall
git clone https://github.com/julius-speech/dictation-kit
cd dictation-kit
sh run-linux-dnn.sh -input adinnet -adport 5530
```
* Then, run the main script with adinnet option. Stop it by Ctrl-C. 
```
python3 pyadintool.py conf/default.yaml --out adinnet
```
```
python3 pyadintool.py conf/default.yaml --out adinnet --server localhost --port 5530
```
```
python3 pyadintool.py conf/default.yaml --out adinnet --server localhost,l92.168.1.30 --port 5530,5530
```

### Example-04: Set multiple output streams ###
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
python3 pyadintool.py conf/default.yaml --enable_logsave --logfilefmt log_%Y%m%d.log
```
* Available format
    * %Y: year
    * %m: month
    * %d: day
    * %H: hour
    * %M: minutes
    * %S: second
    * %u: host name
    * %R: rotation id

### Example-07: Switch mode to filelist batch processing ###
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
* Clikc the close button of the window to stop
![figure of realtime plot](images/fig_realtimeplot.png)

### Example-10: Save raw input signals to files
```
python3 pyadintool.py conf/default.yaml --enable_rawsave
```
```
python3 pyadintool.py conf/default.yaml --enable_rawsave --rawfilefmt raw/%Y%m%d/record_%u_%R_%H%M%S.wav
```

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
