import os
import platform
import socket
import struct
import datetime
import numpy as np
import wave
import time
import queue
import logging

from lib.pipeline import Source, Sink


"""
"""
def path_embed_tag(path, rotateid=None):
    now = datetime.datetime.now()
    if '%Y' in path:
        path = path.replace('%Y', now.strftime('%Y'))
    if '%m' in path:
        path = path.replace('%m', now.strftime('%m'))
    if '%d' in path:
        path = path.replace('%d', now.strftime('%d'))
    if '%H' in path:
        path = path.replace('%H', now.strftime('%H'))
    if '%M' in path:
        path = path.replace('%M', now.strftime('%M'))
    if '%S' in path:
        path = path.replace('%S', now.strftime('%S'))

    if os.name == 'posix':        
        uname = os.uname()[1]
    elif os.name == 'nt':
        uname = platform.node()
        
    if '%u' in path:
        path = path.replace('%u', uname)
        pass

    if rotateid is not None and '%R' in path:
        path = path.replace('%R', f'{rotateid:05d}')

    return path

'''
'''
def setup_logger(enable_logsave, logfilefmt):
    import os
    import sys
    from datetime import datetime
    from logging import StreamHandler, FileHandler, Formatter
    from logging import INFO, DEBUG, NOTSET

    #
    handlers = []

    # set up stream handler
    stream_handler = StreamHandler()
    stream_handler.setLevel(INFO)
    stream_handler.setFormatter(Formatter("%(message)s"))
    handlers.append(stream_handler)

    #
    if enable_logsave is True:
        logfilename = path_embed_tag(logfilefmt)
        dirname = os.path.dirname(logfilename)
        if len(dirname) > 0:
            os.makedirs(dirname, exist_ok=True)
            
        # set up file handler
        file_handler = FileHandler(
            logfilename,
            encoding='utf-8'
        )
        file_handler.setLevel(DEBUG)
        file_handler.setFormatter(
            Formatter("%(asctime)s@ %(name)s [%(levelname)s] %(funcName)s: %(message)s")
        )
        handlers.append(file_handler)

    ###
    logging.basicConfig(level=NOTSET, handlers=handlers)


"""
a microphone input signal must be set in the first channel of audio file.
a loopback (reference) signal must be set in the second channel of audio file.
"""
class AudioSourceFile(Source):
    def __init__(self, filename, fs, nch, nframe, block=True):
        self.filename = filename
        self.fs = fs
        self.nch = nch
        self.nframe = nframe
        self.block = block
        
    def open(self):
        import torchaudio
        s, fs = torchaudio.load(self.filename)
        nch = s.shape[0]

        if self.fs != fs or self.nch != nch:
            raise Exception(f'Different settings: '
                            f'sampling frequency {self.fs}, {fs} '
                            f'or number of channels {self.nch}, {nch}')
        self.s = s.T
        self.n_len = s.shape[1]
        self.s_len = self.n_len / fs
        self.pos_start = 0

    def close(self):
        pass

    def read(self):
        if self.pos_start == self.n_len:
            return None
        
        pos_start = self.pos_start
        pos_end = pos_start + self.nframe

        if pos_end > self.n_len:
            pos_end = self.n_len
        self.pos_start = pos_end

        if self.block is True:
            time.sleep(self.nframe/self.fs)

        return self.s[pos_start:pos_end,:].numpy()

    def progress(self):
        logger = logging.getLogger(__name__)
        logger.info(f'{self.pos_start}/{self.n_len} {self.pos_start/self.fs:.2f}/{self.n_len/self.fs:.2f}')


"""
  constant
"""
_SEG_STATE_NONACTIVE = 0
_SEG_STATE_ACTIVE = 1
_SEG_STATE_END = 2

"""
"""
class AdinnetSinkSocket(Sink):
    def __init__(self, IPs, PORTs):
        self.IPs = str(IPs).split(',')
        self.PORTs = [int(x) for x in str(PORTs).split(',')]
        self.adinsocks = []
        self.scale = 36727.0
        self.state = _SEG_STATE_NONACTIVE
        pass

    def open(self):
        logger = logging.getLogger(__name__)

        for IP, PORT in zip(self.IPs, self.PORTs):
            adinsock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                adinsock.connect((IP, PORT))
                logger.info(f'[LOG]: connection suceeded to {IP}:{PORT}')
                self.adinsocks.append(adinsock)
            except ConnectionRefusedError:
                logger.info(f'[LOG]: failed to connect to {IP}:{PORT}')
                quit()

    def close(self):
        for adinsock in self.adinsocks:
            adinsock.shutdown(1)
            adinsock.close()
        self.adinsocks = []
        pass

    """
    """
    def write(self, data):
        state = data['state']

        # non-active packet
        if state == _SEG_STATE_NONACTIVE:
            return

        # 
        audio = data.get('audio')
        if audio is not None and state != _SEG_STATE_NONACTIVE:
            # speech section
            audio = np.ravel((audio * self.scale).astype('int16'))
            n_len = len(audio) * 2

            ##
            try:
                bdata = struct.pack('<i', n_len)
                for adinsock in self.adinsocks:
                    adinsock.sendall(bdata)
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.info(f'[LOG]: failed to send the number of bytes of data.')
                logger.info(f'[LOG]: {e}')
                quit()
            
            ##
            try:
                bdata = audio.tobytes()
                for adinsock in self.adinsocks:
                    adinsock.sendall(bdata)
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.info(f'[LOG]: failed to send audio data')
                logger.info(f'[LOG]: {e}')
                quit()

        # end of segment
        if state == _SEG_STATE_END:
            try:
                eos = struct.pack('<i', 0)
                for adinsock in self.adinsocks:
                    adinsock.sendall(eos)
            except Exception as e:
                logger = logging.getLogger(__name__)
                logger.info(f'[LOG]: failed to send "end of segment"')
                logger.info(f'[LOG]: {e}')
                quit()

        ##
        pass


class SegmentedAudioSinkFile(Sink):
    def __init__(self, filename, startid, freq=16000, nch=1, btype='int16'):
        self.filename = filename
        self.fileid = startid
        self.state = _SEG_STATE_NONACTIVE
        self.stream = None
        self.freq = freq
        self.nch = 1

        ###
        self.btype = btype
        if btype == 'int16':
            self.bwidth = 2
            self.scale = 32767
            
        pass

    def open(self):
        pass

    def close(self):
        if self.stream is not None:
            self.file_close()

    def getfilename(self):
        filename = path_embed_tag(self.filename, self.fileid)
        dirname = os.path.dirname(filename)
        if len(dirname) > 0:
            os.makedirs(dirname, exist_ok=True)            
        self.fileid += 1
        return filename
    
    def file_open(self):
        filename = self.getfilename()
        self.stream = wave.open(filename, 'wb')
        self.stream.setnchannels(self.nch)
        self.stream.setsampwidth(self.bwidth)
        self.stream.setframerate(self.freq)

    def file_close(self):
        self.stream.close()
        self.stream = None

    """
    """
    def write(self, data):
        state = data['state']

        # non-active packet
        if state == _SEG_STATE_NONACTIVE:
            return

        audio = data.get('audio')
        if audio is not None and state != _SEG_STATE_NONACTIVE:
            if self.stream is None:
                self.file_open()            
            audio = data['audio']
            self.stream.writeframes((audio*self.scale).flatten().astype(self.btype).tobytes())

        # end of segment
        if state == _SEG_STATE_END:
            self.file_close()

        ##
        pass

class AudioSinkFileRotate(Sink):
    def __init__(self, filenamefmt, rotate_min, fs, nch, btype='int16'):
        self.filenamefmt = filenamefmt
        self.rotate_min = rotate_min
        self.fs = fs
        self.nch = nch

        self.btype = btype
        if btype == 'int16':
            self.bwidth = 2
            self.scale = 32767
        elif btype == 'int32':
            self.bwidth = 4
            self.scale = 2**32-1

        ##
        self.fileid = 0
        self.total_size = 0
        self.current_size = 0
        self.maxsize_per_file = self.bwidth * self.fs * self.nch * 60 * self.rotate_min

    def getfilename(self):
        filename = path_embed_tag(self.filenamefmt, self.fileid)
        dirname = os.path.dirname(filename)
        if len(dirname) > 0:
            os.makedirs(dirname, exist_ok=True)            
        self.fileid += 1
        return filename

    def open(self):
        filename = self.getfilename()

        logger = logging.getLogger(__name__)
        logger.info(f'[LOG]: now opening "{filename}"')

        self.stream = wave.open(filename, 'wb')
        self.stream.setnchannels(self.nch)
        self.stream.setsampwidth(self.bwidth)
        self.stream.setframerate(self.fs)
        logger.info(f'[LOG]: raw audio file is opened"')

    def close(self):
        self.stream.close()
        pass


    def write(self, data):
        n_remained_frame = int((self.maxsize_per_file - self.current_size) / (self.bwidth * self.nch))

        if len(data) >= n_remained_frame:
            rdata = data[:n_remained_frame]
            self.stream.writeframes((rdata*self.scale).flatten().astype(self.btype).tobytes())
            self.total_size += rdata.size * self.bwidth
            self.current_size += rdata.size * self.bwidth

            logger = logging.getLogger(__name__)
            logger.info(f'[LOG]: close the file and create a new file.')
            logger.info(f'[LOG]: current data sizes (bytes): '
                             f'saved file {self.current_size}, '
                             f'limit {self.maxsize_per_file}, '
                             f'total {self.total_size}.')
            self.current_size = 0
            self.close()
            
            logger.info(f'[LOG]: file closed')
            self.open()

            data = data[n_remained_frame:]

        self.stream.writeframes((data*self.scale).flatten().astype(self.btype).tobytes())
        self.total_size += data.size * self.bwidth
        self.current_size += data.size * self.bwidth

"""
"""
class TimestampTextSinkFile(Sink):
    def __init__(self, filename):
        self.filename = filename
        self.stream = None
        self.state = 0
        pass

    def open(self):
        self.stream = open(self.filename, 'w')
        pass
    
    def close(self):
        self.stream.close()
        pass

    """
    """
    def write(self, data):
        state = data['state']

        # end of segment
        if state == _SEG_STATE_END:
            print(f"{data['start']:.3f}\t{data['end']:.3f}\tspeech", file=self.stream, flush=True)
                        
        ##
        pass

try:
    import pyaudio
    import struct

    class PyAudioSource(Source):
        def __init__(self, fs, nch, nframe):
            self.nframe = nframe
            self.fs = fs
            self.nch = nch
            self.audio = pyaudio.PyAudio()

        def open(self):
            self.stream = self.audio.open(format=pyaudio.paInt16,
                                          channels=self.nch,
                                          rate=self.fs,
                                          input=True,
                                          frames_per_buffer=800)
        
        def close(self):
            if self.stream is not None:
                self.stream.stop_stream()
                self.stream.close()
            self.audio.terminate()
        
        def read(self):
            data = self.stream.read(self.nframe)
            data = np.frombuffer(data, dtype="int16")/32768.0
            return data.reshape(-1, self.nch)
        
except:
    pass
        
try:
    import sounddevice as sd

    class SoundDeviceSource(Source):
        def query_device():
            return sd.query_devices()

        def get_default_device():
            return sd.default.device[0]

        def callback(self, indata, frames, time, status):
            if status:
                self.logger.info(f'[ERROR]: {status}')
                self.logger.info(f'[LOG]: callback-info: frame: {self.total_frames}, queue-size: {self.q.qsize()}')
                
            self.q.put(indata.copy())

            self.total_frames += frames
            self.count_frames += frames

            if self.count_frames >= self.cycle_frames:
                self.logger.info(f'[LOG]: callback-info: time: buffer: {time.inputBufferAdcTime:.4f}, '
                                 f' current: {time.currentTime:.4f} -- {self.total_frames},'
                                 f' queue-size: {self.q.qsize()}, arg frames: {frames}, arg indata.shape: {indata.shape}')
                self.count_frames -= self.cycle_frames
        
        def __init__(self, device, fs, nch, nlimit=-1):
            self.device = device
            self.nch = nch
            self.fs = fs
            self.nlimit = nlimit

            self.cycle_frames = self.fs * 10
            self.total_frames = 0
            self.count_frames = 0

            import logging
            self.logger = logging.getLogger(__name__)
            
            self.stream = sd.InputStream(device=self.device,
                                         channels=self.nch,
                                         samplerate=self.fs,
                                         callback=self.callback)
            self.q = queue.Queue()

        def open(self):
            self.stream.start()

        def close(self):
            self.stream.stop()
            self.stream.close()

        def read(self):
            if self.nlimit > 0 and self.total_frames > self.nlimit:
                return None
            return self.q.get()

    class SoundDeviceSink(Sink):
        def query_device():
            return sd.query_devices()
        
        def __init__(self, device, fs, nch, nframe):
            self.device = device
            self.nch = nch
            self.fs = fs
            self.nframe = nframe
            
            self.stream = sd.OutputStream(device=self.device,
                                          channels=self.nch,
                                          samplerate=self.fs,
                                          blocksize=int(nframe*2.0))

        def open(self):
            self.stream.start()

        def close(self):
            self.stream.stop()
            self.stream.close()

        """
        data: [Len, Ch]
        """
        def write(self, data):
            self.stream.write(data)
            return data
except:
    pass


if __name__ == "__main__":
    pass
