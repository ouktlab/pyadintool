"""
  VAD in Time Domain
"""
import numpy as np
import logging
import lib.pipeline as pl
import lib.io as io


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
            print(f'[ERROR]: required size, {n_len}, is larger than '
                  f'the size of labeled data, {self.n_labeled}.')
            quit()
        
        data = self.buf_data[:n_len]
        label = self.buf_label[:n_len]

        self.buf_data = self.buf_data[n_len:]
        self.buf_label = self.buf_label[n_len:]
        self.n_labeled -= n_len

        return np.concatenate([data, label], axis=1)
        

'''
'''
class npBinaryProbFilter(pl.Processor):
    def __init__(self, ptrans_self=0.99, weight=0.5, dtype='float32'):
        self.dtype = dtype
        self.weight = np.array([1 - weight, weight])
        self.trans_prob = np.array([
            [ptrans_self, 1 - ptrans_self],
            [1 - ptrans_self, ptrans_self]
        ], dtype=dtype)

        self.reset()

    def reset(self):
        self.prior = np.array([1, 0], dtype=self.dtype)

    def update(self, data, isEOS=False):
        n_data = len(data)
        
        labels = np.zeros((n_data,1), dtype=self.dtype)
        lh = np.stack([1 - data, data], axis=1)

        for t in range(n_data):
            post = np.matmul(self.trans_prob, self.prior) * lh[t,:] / self.weight
            self.prior = post / np.sum(post)
            labels[t] = self.prior[1]

        return labels

"""
  Power-based Simple VAD
"""
class SimpleVAD(pl.Processor):
    def __init__(self, n_win, flramp=300,
                 n_skip=80, thre=0.5, nbits=16, dtype='float32'):
        # fixed parameters
        self.dtype = dtype
        self.n_win = n_win
        self.flrpower = ((flramp / (2**(nbits-1)-1))**2)
        self.n_skip = n_skip
        self.thre = thre

        self.pairbuf = PairedBuffer(n_skip)
        self.smoother = npBinaryProbFilter()

        self.reset()

    def reset(self):
        self.buf = np.zeros((self.n_win, 1), dtype=self.dtype)
        self.smoother.reset()
        self.pairbuf.reset()

    def update(self, data, isEOS):
        if data is None:
            return None
        
        n_data = len(data)

        # 
        self.pairbuf.push(data.astype(self.dtype))

        # 
        while (chunk := self.pairbuf.get_unlabeled(self.n_skip)) is not None:
            # shift buffer for power calculation
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
    _STATE_NONACTIVE = 0
    _STATE_ACTIVE = 1
    _STATE_MARGIN = 2

    def __init__(self, freq, margin_begin, margin_end,
                 thre=0.5, shift_time=0.2, dtype='float32'):
        self.dtype = dtype

        # fixed parameters
        self.freq = freq
        self.n_buf = int(freq * (margin_begin + margin_end))
        self.n_margin_begin = int(freq * margin_begin)
        self.n_margin_end = int(freq * margin_end)
        self.thre = thre
        self.shift_frames = int(shift_time * freq)

        # 
        self.reset()

    def reset(self):
        self.buf = np.zeros((self.n_buf,1), dtype=self.dtype)
        self.state = self._STATE_NONACTIVE
        self.plab = 0
        self.total_frames = 0
        self.time_start = 0
        self.time_end = 0

    def update(self, data, isEOS):
        if data is None:
            return None, None

        n_data = len(data)
        base_nframe = self.total_frames - self.shift_frames
        
        # end of source input
        if isEOS is True and self.state != self._STATE_NONACTIVE:
            self.time_end = base_nframe / self.freq
            logger = logging.getLogger(__name__)
            logger.info(f'[LOG]: segment: {self.time_start:.3f} {self.time_end:.3f}             ')
            return None, [{
                'state':io._SEG_STATE_END, 'audio':None,
                'start':self.time_start, 'end':self.time_end
            }]

        # separation
        oaudio, ilabel = data[:,0], data[:,1]

        # binarization
        ilabel = np.reshape(
            np.where(ilabel >= self.thre, 1.0, 0.0),
            (-1,1)
        ).astype(np.float32)

        # buffering
        self.buf = np.append(self.buf, oaudio)
        self.total_frames += n_data

        #
        olabel = np.copy(ilabel)
        n_oaudio = 0
        packets = []

        #
                    
        # very naive implementation (high computational cost)
        for n, lab in enumerate(ilabel):
            
            # non-speech section
            if self.state == self._STATE_NONACTIVE:
                #print('[LOG]: now non-speech section\r', end='')
                packet_state = io._SEG_STATE_NONACTIVE
                
                # detection of speech frame: move to state ACTIVE (speech)
                if lab - self.plab > 0:
                    print('[LOG]: now speech section (actual)\r', end='')
                    self.state = self._STATE_ACTIVE
                    
                    n_oaudio = self.n_margin_begin
                    oaudio = self.buf[self.n_buf+n-self.n_margin_begin:]
                    self.time_start = (base_nframe + n - self.n_margin_begin) / self.freq
            
            # speech section (actual)
            if self.state == self._STATE_ACTIVE:
                #print('[LOG]: now speech section (actual)\r', end='')
                packet_state = io._SEG_STATE_ACTIVE
                
                # detection of non-speech frame: move to state MARGIN (margin)
                if lab - self.plab < 0:
                    print('[LOG]: now speech section (margin)\r', end='')
                    self.state = self._STATE_MARGIN
                    self.cnt_margin_frame = 0
                else:
                    n_oaudio += 1                

            # non-speech section (margin)
            if self.state == self._STATE_MARGIN:
                #print('[LOG]: now speech section (margin)\r', end='')
                self.cnt_margin_frame += 1
                packet_state = io._SEG_STATE_ACTIVE

                olabel[n] = 1
                n_oaudio += 1

                # detection of speech section: move to state ACTIVE (speech)
                if lab - self.plab > 0:
                    print('[LOG]: now speech section (actual)\r', end='')
                    self.state = self._STATE_ACTIVE

                # end of margin: move to state NONACTIVE (non-speech)
                if self.cnt_margin_frame >= self.n_margin_end:
                    self.state = self._STATE_NONACTIVE
                    self.time_end = (base_nframe + n) / self.freq

                    logger = logging.getLogger(__name__)
                    logger.info(f'[LOG]: segment: {self.time_start:.3f} {self.time_end:.3f}             ')

                    packets.append({
                        'state':io._SEG_STATE_END, 'audio':oaudio[:n_oaudio],
                        'start':self.time_start, 'end':self.time_end
                    })
                    packet_state = io._SEG_STATE_NONACTIVE
                    print('[LOG]: now non-speech section\r', end='')
        
            self.plab = lab

        # buffer update
        self.buf = self.buf[-self.n_buf:]

        # remained data packet
        packets.append({
            'state':packet_state, 'audio':oaudio[:n_oaudio],
            'start':self.time_start, 'end':self.time_end
        })

        return olabel, packets
