from lib.pipeline import Processor

def bwd_padding(data, nlen):
    n = int(np.ceil(np.log2(nlen)))
    return np.concatenate([np.zeros((2**n-len(data))), data])

class LMSblockFFT(Processor):
    """
    Fast least mean square using FFT
    """
    def __init__(self, L, mu):
        self.L = L
        self.w = np.zeros(L, dtype='float32')
        self.buf_ = np.zeros(L, dtype='float32')
        self.mu = mu
        pass

    # data: [Len, CH]
    def update(self, data, isEOS):
        X, D = data[:,0], data[:,1]        
        n_len = len(X)

        # fft convolution
        s = np.concatenate([self.buf_, D])
        w_ = bwd_padding(self.w, self.L + n_len)
        s_ = bwd_padding(s, self.L + n_len)

        W = np.fft.rfft(np.flipud(w_))
        S = np.fft.rfft(s_)
        y = np.fft.irfft(W * S, len(s_)).real[-n_len:]

        # error
        e = X - y
        for i in range(n_len):
            self.w = self.w + self.mu * 2 * e[i] * s[i:i+self.L] / n_len

        # buffer-shift
        self.buf_ = s[-self.L:]

        return e.reshape(-1,1)

    def load(self, filename):
        self.w = np.loadtxt(filename)
        
    def save(self, filename):
        np.savetxt(filename, self.w)


class ChannelSelector(Processor):
    """
    Select channels
    """
    def __init__(self, tgt_chs=[0]):
        self.tgt_chs = tgt_chs

    def update(self, data, isEOS):
        return data[:,self.tgt_chs]


