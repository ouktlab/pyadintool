"""
  VAD in Time Domain
"""
import numpy as np
import torch
from silero_vad import load_silero_vad, read_audio

import lib.pipeline as pl
import usr.tdvad as tdvad

"""
  Silero VAD
"""
class SileroVAD(pl.Processor):
    def __init__(self, freq, thre, device='cpu', dtype='float32'):
        self.dtype = dtype
        self.model = load_silero_vad()
        self.model.to(device)

        self.freq = freq
        self.thre = thre
        self.window_size_samples = 512 if freq == 16000 else 256

        self.pairbuf = tdvad.PairedBuffer(self.window_size_samples)

    # It is better to reset the internal state of Silero model
    # for filelist processing
    def reset(self):
        pass

    '''
    '''
    def update(self, data, isEOS):
        if data is None:
            return None
        
        n_data = len(data)

        # push to buffer
        self.pairbuf.push(data.astype(self.dtype))

        # estimate label
        while (chunk := self.pairbuf.get_unlabeled(self.window_size_samples)) is not None:
            # 
            chunk = torch.from_numpy(chunk).clone().squeeze(dim=1)
            speech_prob = self.model(chunk, self.freq).item()

            #
            speech_prob = 1 if speech_prob > self.thre else 0

            self.pairbuf.set_label(
                np.ones((self.window_size_samples,1),
                        dtype=self.dtype)
                * speech_prob
            )

        # pop labeled paired data
        return self.pairbuf.pop(n_data)

