"""
  VAD in Time Domain
"""
import numpy as np
import lib.pipeline

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

