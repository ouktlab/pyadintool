"""
  VAD in Time Domain
"""
import numpy as np
import logging
import lib.pipeline
from lib.io import _SEG_STATE_NONACTIVE, _SEG_STATE_ACTIVE, _SEG_STATE_END

'''
'''
class HMMSmoother(lib.pipeline.Processor):
    def __init__(self, ptrans_self=0.99):
        self.post = np.zeros(2, dtype='float32')
        
        self.prior = np.zeros(2, dtype='float32')        
        self.prior[0] = 1

        self.p11 = ptrans_self
        self.p12 = 1 - ptrans_self
        self.p21 = 1 - ptrans_self
        self.p22 = ptrans_self
        pass

    def update(self, data):
        n_data = len(data)

        labels = np.zeros(data.shape, dtype='float32')
        for t in range(n_data):
            self.post[0] = self.p11 * self.prior[0] + self.p12 * self.prior[1]
            self.post[1] = self.p21 * self.prior[0] + self.p22 * self.prior[1]

            self.post[0] *= (1 - data[t,0])
            self.post[1] *= data[t,0]

            self.post = self.post / np.sum(self.post)
            self.prior = self.post

            labels[t] = self.post[1]

        return labels

"""
  Power-based Simple VAD
"""
class SimpleVAD(lib.pipeline.Processor):
    def __init__(self, n_buf, n_win, flramp=300,
                 n_skip=80, thre=0.5, n_ch=1, nbits=16):
        self.n_buf = n_buf
        self.buf = np.zeros((n_buf, n_ch), dtype='float32')
        self.n_win = n_win

        self.flrpower = ((flramp / (2**(nbits-1)-1))**2)
        self.thre = thre

        self.n_skip = n_skip
        self.skip_offset = 0
        self.last_label = 0
        self.smoother = HMMSmoother()

    def update(self, data):
        n_data = len(data)

        # shift
        self.buf = np.roll(self.buf, shift=self.n_buf - n_data)
        self.buf[-n_data:,:] = (data ** 2)

        # moving average
        power = np.zeros(data.shape, dtype='float32')
        power[0] = np.sum(self.buf[(-n_data-self.n_win):-n_data,:])
        for i in range(1,n_data):
            power[i] = power[i-1] - self.buf[-n_data+i-self.n_win-1,:] + self.buf[-n_data+i-1,:]
        power /= self.n_win
        
        #
        power = power[self.skip_offset::self.n_skip,:]

        #
        labels_ = np.zeros(power.shape, dtype='float32')
        labels_[power >= self.flrpower] = 1        
        labels_ = self.smoother.update(labels_)

        #
        labels_[labels_ >= self.thre] = 1.0
        labels_[labels_ < self.thre] = 0.0

        labels = np.ones(data.shape, dtype='float32') * self.last_label

        for i in range(n_data - self.skip_offset):
            labels[self.skip_offset + i] = labels_[int(i/self.n_skip)]
        q, mod = divmod(n_data - self.skip_offset, self.n_skip)
        self.skip_offset = (self.n_skip -  mod) % self.n_skip

        self.last_label = labels[-1]

        return np.concatenate([data, labels], axis=1)

"""
"""
class PostProc(lib.pipeline.Processor):
    def __init__(self, fs, margin_begin, margin_end, shift_time=0.2):
        self.n_buf = int(fs * (margin_begin + margin_end))
        self.n_margin_begin = int(fs * margin_begin)
        self.n_margin_end = int(fs * margin_end)
        
        self.buf = np.zeros((self.n_buf, 1), dtype=float)
        self.state = 0
        self.prev_lab = 0

        self.fs = fs
        self.shift_frames = int(shift_time * self.fs)

        self.total_frames = 0
        self.time_start = 0
        self.time_end = 0
        pass

    def reset(self):
        self.buf = np.zeros((self.n_buf, 1), dtype=float)
        self.state = 0
        self.prev_lab = 0
        self.total_frames = 0
        self.time_start = 0
        self.time_end = 0

    #  data: [len, ch]
    def update(self, data):
        # 
        if data is None:
            return None
        
        # binarize label
        data[(data[:,1] >= 0.5),1] = 1
        data[(data[:,1] < 0.5),1] = 0

        #
        packet_state = _SEG_STATE_NONACTIVE
        labels = np.zeros((len(data),1), dtype=float)
        audio = np.zeros((len(data)+self.n_margin_begin, 1), dtype=float)
        n_audio = 0

        # naive implementation
        for n in range(len(data)):
            
            # non-speech section
            if self.state == 0:
                print('[LOG]: now non-speech section\r', end='')
                
                # detection of speech frame: move to state 1 (speech)
                if data[n,1] - self.prev_lab > 0:
                    self.state = 1
                    labels[n] = 1
                    
                    audio[0:self.n_margin_begin-n] = self.buf[-(self.n_margin_begin-n):]
                    n_audio += self.n_margin_begin - n

                    for nn in range(n):
                        audio[n_audio] = data[nn,0]
                        n_audio += 1
                        
                    packet_state = _SEG_STATE_ACTIVE
                    self.time_start = (self.total_frames + n - self.n_margin_begin - self.shift_frames) / self.fs
                    
                pass
            
            # speech section (actual)
            elif self.state == 1:
                packet_state = 1
                labels[n] = 1
                audio[n_audio] = data[n,0]
                n_audio += 1
                
                print('[LOG]: now speech section (actual)\r', end='')
                # detection of non-speech frame: move to state 2 (margin)
                if data[n,1] - self.prev_lab < 0:
                    self.state = 2
                    self.cnt_margin_frame = 0
                pass

            # non-speech section (margin)
            elif self.state == 2:
                packet_state = _SEG_STATE_ACTIVE
                labels[n] = 1
                audio[n_audio] = data[n,0]
                n_audio += 1
                
                print('[LOG]: now speech section (margin)\r', end='')
                self.cnt_margin_frame += 1

                # detection of speech section: move to state 1 (speech)
                if data[n,1] - self.prev_lab > 0:
                    self.state = 1

                # end of maring: move to state 0 (non-speech)
                if self.cnt_margin_frame > self.n_margin_end:
                    self.state = 0
                    packet_state = _SEG_STATE_END
                    self.time_end = (self.total_frames + n - self.shift_frames) / self.fs

                    logger = logging.getLogger(__name__)
                    logger.info(f'[LOG]: segment: {self.time_start:.3f} {self.time_end:.3f}             ')
                    
                pass
        
            self.prev_lab = data[n,1]
            pass
        
        # data shift
        self.buf = np.roll(self.buf, shift=-len(data))
        self.buf[-len(data):] = data[:,0:1]
        self.total_frames += len(data)

        return labels, {
            'state':packet_state,
            'audio':audio[:n_audio],
            'start':self.time_start,
            'end':self.time_end
        }

