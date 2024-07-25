"""
  VAD in Time Domain
"""
import numpy as np
import lib.pipeline

import torch
from silero_vad import load_silero_vad, read_audio

"""
  Silero VAD
"""
class SileroVAD(lib.pipeline.Processor):
    def __init__(self, freq, thre, device='cpu'):
        self.model = load_silero_vad()
        self.model.to(device)
        self.buf = np.zeros((0, 1), dtype='float32')

        self.freq = freq
        self.thre = thre
        self.window_size_samples = 512 if freq == 16000 else 256
        self.last_label = 0

    def update(self, data):
        n_data = len(data)

        # concate
        self.buf = np.concatenate([self.buf, data], axis=0)

        #
        if len(self.buf) < self.window_size_samples:            
            labels = np.ones(data.shape, dtype='float32') * self.last_label
            return np.concatenate([data, labels], axis=1)
    
        #
        labels = np.zeros(data.shape, dtype='float32')
        n_chunk = 0
        for i in range(0, len(self.buf), self.window_size_samples):            
            chunk = self.buf[i:i+self.window_size_samples,:]
            chunk = torch.from_numpy(chunk.astype(np.float32)).clone().squeeze(dim=1)
            
            if len(chunk) < self.window_size_samples:
                break
            
            speech_prob = self.model(chunk, self.freq).item()
            
            # 
            if speech_prob > self.thre:
                speech_prob = 1
            else:
                speech_prob = 0

            self.last_label = speech_prob
            labels[i:i+self.window_size_samples] = self.last_label


        labels[i+self.window_size_samples:] = self.last_label 
        self.buf = self.buf[i+self.window_size_samples:]
        
        return np.concatenate([data, labels], axis=1)
