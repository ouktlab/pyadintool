"""
  VAD in Time Domain
"""
import numpy as np
import logging
import lib.pipeline as pl
from lib.io import _SEG_STATE_NONACTIVE, _SEG_STATE_ACTIVE, _SEG_STATE_END

'''
'''
class PairedBuffer:
    def __init__(self, n_init, dtype='float32'):
        self.n_init = n_init
        self.dtype = dtype
        self.reset()

    def reset(self):
        self.n_labeled = self.n_init
        self.buf_data = np.zeros((self.n_init, 1), dtype=self.dtype)
        self.buf_label = np.zeros((self.n_init, 1), dtype=self.dtype)        

    def push(self, data):
        self.buf_data = np.concatenate([self.buf_data, data], axis=0)

    def get_unlabeled(self, n_len):
        pos_end = self.n_labeled + n_len
        if pos_end > len(self.buf_data):
            return None        
        return self.buf_data[self.n_labeled:self.n_labeled+n_len]

    def set_label(self, labels):
        self.buf_label = np.concatenate([self.buf_label, labels], axis=0)
        self.n_labeled += len(labels)

    def pop(self, n_len):
        if n_len > self.n_labeled:
            print(f'[ERROR]: required size {n_len} is larger than labeled data size {self.n_labeled}')
            quit()
        
        data = self.buf_data[:n_len]
        label = self.buf_label[:n_len]

        self.buf_data = self.buf_data[n_len:]
        self.buf_label = self.buf_label[n_len:]
        self.n_labeled -= n_len

        return np.concatenate([data, label], axis=1)
        

'''
'''
class HMMSmoother(pl.Processor):
    def __init__(self, ptrans_self=0.99, dtype='float32'):
        self.dtype = dtype
        self.prior = np.array([1, 0], dtype=dtype)
        self.trans_prob = np.array([
            [ptrans_self, 1 - ptrans_self],
            [1 - ptrans_self, ptrans_self]
        ], dtype=dtype)

    def reset(self):
        self.prior = np.array([1, 0], dtype=dtype)

    def update(self, data):
        n_data = len(data)
        labels = np.zeros(data.shape, dtype=self.dtype)

        for t, x in enumerate(data):
            post = np.matmul(self.trans_prob, self.prior)
            post[0] *= (1 -x)
            post[1] *= x

            post = post / np.sum(post)
            self.prior = post

            labels[t] = post[1]

        return labels

"""
  Power-based Simple VAD
"""
class SimpleVAD(pl.Processor):
    def __init__(self, n_win, flramp=300,
                 n_skip=80, thre=0.5, nbits=16, dtype='float32'):
        self.dtype = dtype
        
        self.n_win = n_win
        self.flrpower = ((flramp / (2**(nbits-1)-1))**2)
        self.n_skip = n_skip
        self.thre = thre

        self.pairbuf = PairedBuffer(n_skip)
        self.smoother = HMMSmoother()
        
        self.buf = np.zeros((n_win, 1), dtype=dtype)

    def reset(self):
        self.buf = np.zeros((self.n_win, 1), dtype=self.dtype)
        self.smoother.reset()
        self.pairbuf.reset()

    def update(self, data):
        n_data = len(data)

        # 
        self.pairbuf.push(data.astype(self.dtype))

        # 
        while True:
            chunk = self.pairbuf.get_unlabeled(self.n_skip)
            if chunk is None:
                break

            # shift buffer for power
            self.buf = np.roll(self.buf, shift=self.n_win - self.n_skip)
            self.buf[-self.n_skip:,:] = (chunk ** 2)

            # average
            power = np.mean(self.buf[-self.n_win:])

            label = 1 if power >= self.flrpower else 0
            label = self.smoother.update(np.array([label]))
            
            self.pairbuf.set_label(
                np.ones((self.n_skip, 1),
                        dtype=self.dtype)
                * label
            )

        return self.pairbuf.pop(n_data)


"""
"""
class PostProc(pl.Processor):
    def __init__(self, fs, margin_begin, margin_end,
                 shift_time=0.2, dtype='float32'):
        self.dtype = dtype
        self.n_buf = int(fs * (margin_begin + margin_end))
        self.n_margin_begin = int(fs * margin_begin)
        self.n_margin_end = int(fs * margin_end)
        
        self.buf = np.zeros((self.n_buf, 1), dtype=dtype)
        self.state = 0
        self.prev_lab = 0

        self.fs = fs
        self.shift_frames = int(shift_time * self.fs)

        self.total_frames = 0
        self.time_start = 0
        self.time_end = 0
        pass

    def reset(self):
        self.buf = np.zeros((self.n_buf, 1), dtype=self.dtype)
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
        labels = np.zeros((len(data),1), dtype=self.dtype)
        audio = np.zeros((len(data)+self.n_margin_begin, 1), dtype=self.dtype)
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
                packet_state = _SEG_STATE_ACTIVE
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

