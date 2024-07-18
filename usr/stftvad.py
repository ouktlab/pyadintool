"""
  VAD in STFT domain
"""

import os
import yaml
import logging
import torch
import numpy as np

from os.path import dirname
from importlib import import_module
from huggingface_hub import PyTorchModelHubMixin

import lib.pipeline
from lib.io import _SEG_STATE_NONACTIVE, _SEG_STATE_ACTIVE, _SEG_STATE_END

"""
"""
def _load_model(package, classname, params, path=None):
    module = import_module(package)
    if path is None:
        model =  getattr(module, classname)(**params)
        return model

    if '.pt' in path:
        model = getattr(module, classname)(**params)
        ckp = torch.load(path, map_location=torch.device('cpu'))
        model.load_state_dict(ckp['model_state_dict'], strict=False)
    else:
        model = getattr(module, classname).from_pretrained(path)
        
    return model


'''
  recursive HMM filtering
'''
class BinaryProbFilter:
    def __init__(self, trp1_self=0.99, trp2_self=0.99, pw_1=0.5):
        trans = [[trp1_self, 1 - trp1_self],[1 - trp2_self, trp2_self]]
        self.trans = torch.tensor(trans)
        self.weight = torch.tensor([pw_1, 1 - pw_1])
        self.prior = torch.tensor([0.0, 1.0])

    def __call__(self, binlabels):
        return self.forward(binlabels)
        
    def forward(self, binlabels):
        n_len = len(binlabels)
        
        lh = torch.zeros((2,n_len))
        lh[0,:] = binlabels
        lh[1,:] = 1.0 - binlabels
        lh = (lh.T / self.weight).T

        post = torch.zeros(n_len)

        for t in range(n_len):
            p_trans = torch.mv(self.trans, self.prior)
            self.prior = p_trans * lh[:,t]
            self.prior = self.prior / torch.sum(self.prior)
            post[t] = self.prior[0]

        return post

    def reset(self):
        self.prior = torch.tensor([0.0, 1.0])

'''
'''
class DNNHMMFilter(torch.nn.Module):
    def __init__(self, n_fwd, n_bwd, n_offset, classifier, probfilter):
        super(DNNHMMFilter, self).__init__()
        self.classifier = _load_model(**classifier)
        self.probfilter = _load_model(**probfilter)

        self.n_fwd = n_fwd
        self.n_bwd = n_bwd
        self.n_offset = n_offset

    def __call__(self, pspec):
        return self.forward(pspec)

    def set_device(self, device):
        self.classifier = self.classifier.to(device)

    def reset(self):
        self.probfilter.reset()


    def forward(self, pspec):
        prob = self.classifier(pspec, probs=None)[:,-1]
        post = self.probfilter(prob.to('cpu'))            
        return prob, post


####
class BufferedWav2AmpSpec:
    def __init__(self, n_fwd=0, n_bwd=100, device='cpu', n_fft=512, n_shift=160):
        self.n_fwd = n_fwd
        self.n_bwd = n_bwd
        self.n_bwdpad = (n_bwd) * n_shift + n_fft
        self.n_fwdpad = (n_fwd) * n_shift + n_fft
                
        self.n_fft = n_fft
        self.n_shift = n_shift
        self.n_block = (self.n_fwd + self.n_bwd + 1)

        # buffers [Ch, Len]
        self.buf_wav = torch.randn(self.n_bwdpad).reshape(1, self.n_bwdpad) * 1.0e-5 
        self.buf_aspec = torch.zeros((0, 1, int(self.n_fft/2)))
        self.buf_feats = torch.zeros((0, self.n_block, int(self.n_fft/2)))

        #
        self.window = torch.hann_window(n_fft, device=device)

        self.debug = False
        pass

    def reset(self):
        self.buf_wav = torch.randn(self.n_bwdpad).reshape(1, self.n_bwdpad) * 1.0e-5 # [Ch, Len]
        self.buf_aspec = torch.zeros((0, 1, int(self.n_fft/2)))
        self.buf_feats = torch.zeros((0, self.n_block, int(self.n_fft/2)))
        
    def status(self):
        print(f'{self.buf_wav.shape}')
        print(f'{self.buf_aspec.shape}')
        print(f'{self.buf_feats.shape}')

    def noise_append(self):
        self.buf_wav = torch.randn(self.n_fwdpad).reshape(1, self.n_fwdpad) * 1.0e-8

    """
    wavs: torch.tensor [Ch, Len]
    """
    def wav_append(self, wavs):
        if self.debug:
            print(f'[LOG]: wav_append: input - {wavs.shape}, wavbuf - {self.buf_wav.shape}')
        self.buf_wav = torch.concat([self.buf_wav, wavs], dim=1)
        
    """
    
    """
    def wav2pspec_in_buf_if_possible(self):
        [n_ch, n_wav] = self.buf_wav.shape
        
        if self.debug:
            print(f'[LOG]: wav2spec: wavbuf - {self.buf_wav.shape}, specbuf - {self.buf_aspec.shape}')

        # 
        if n_wav >= self.n_fft:
            n_frame = (n_wav - self.n_fft) // self.n_shift + 1
            n_span = self.n_fft + self.n_shift * (n_frame - 1)

            spec = torch.stft(self.buf_wav[:,:n_span], n_fft=self.n_fft, hop_length=self.n_shift,
                              win_length=self.n_fft, window=self.window,
                              center=False, onesided=True, return_complex=True)
            aspec = torch.abs(spec.permute(2,0,1))[:,:,1:]

            # set aspec-buffer
            self.buf_aspec = torch.concat([self.buf_aspec, aspec], dim=0)
            
            # shift wav-buffer (erase old waves)
            self.buf_wav = self.buf_wav[:,n_span - (self.n_fft - self.n_shift):]

    def framesplice_in_buf_if_possible(self):
        ## 
        [n_frame, n_ch, n_freq] = self.buf_aspec.shape
        if self.debug:
            print(f'[LOG]: framesplice: specbuf - {self.buf_pspec.shape}, featbuf - {self.buf_feats.shape} ')

        ## 
        if n_frame >= self.n_block:
            # 
            n_span = n_frame - self.n_block
            #
            feats = [self.buf_feats]
            for t in range(n_span):
                feats.append(self.buf_aspec[t:t+self.n_block,:,:].reshape(1,self.n_block,-1))
            self.buf_feats = torch.concat(feats, dim=0)
            self.buf_aspec = self.buf_aspec[n_span:,:,:]

    def get_feats_if_possible(self, req_frame):
        [n_frame, n_ch, n_freq] = self.buf_feats.shape
        if self.debug:
            print(f'[LOG]: fet_feats: {self.buf_feats.shape}')

        if n_frame >= req_frame:
            feats = self.buf_feats[:,:,:]
            self.buf_feats = torch.zeros((0, self.n_block, int(self.n_fft/2)))
            return feats
        else:
            return None

    def get(self, blockwav, req_frame=1):
        if blockwav is None:
            self.noise_append()
        else:
            self.wav_append(blockwav)    
        self.wav2pspec_in_buf_if_possible()
        self.framesplice_in_buf_if_possible()
        return self.get_feats_if_possible(req_frame)

#####
#
class stftSlidingVAD(lib.pipeline.Processor):
    def __init__(self, yamlfile, min_frame,
                 nshift=160, nbuffer=12000, device='cpu', nthread=4):

        with open(yamlfile, 'r') as yml:
            try:
                config = yaml.safe_load(yml)
            except yaml.YAMLError as exc:
                print(exc)
                quit()
            
        self.model = DNNHMMFilter(**config)
        
        #self.model = HMMFilter.from_pretrained('my-hmm-model')
        #self.model.from_pretrained('my-hmm-model')

        #
        #if '.yaml' in yamlfile:
        #    self.model.load_from_config(yamlfile)
        #else:
        #    self.model.from_pretrained(yamlfile)

        # 
        
        self.wav2aspec = BufferedWav2AmpSpec(self.model.n_fwd, self.model.n_bwd)
        self.model.set_device(device)

        self.min_frame = min_frame
        self.nshift = nshift
        self.device = device

        self.nbuffer = nbuffer
        self.delayed_sample = self.model.n_offset * self.nshift
        self.delayed_data = torch.zeros(self.nbuffer)

        self.prev_lab = 0

        torch.set_num_threads(nthread)

        pass

    def reset(self):
        self.prev_lab = 0
        self.model.reset()
        self.wav2aspec.reset()
        self.delayed_data = torch.zeros(self.nbuffer)

    def update(self, data):

        if data is not None:
            # 
            data = torch.from_numpy(data).T.clone()
            n_len = len(data.T)

            # data shift
            self.delayed_data = torch.roll(self.delayed_data, shifts=-n_len)
            self.delayed_data[-n_len:] = data[0,:]

            # 
            feats = self.wav2aspec.get(data, self.min_frame)
        else:
            feats = self.wav2aspec.get(data, 1)            
            n_len = feats.shape[0] * self.nshift

        # 
        outputs = torch.zeros((n_len, 2))
        outputs[:,0] = self.delayed_data[-(self.delayed_sample+n_len):-self.delayed_sample]
        outputs[:,1] = self.prev_lab

        # 
        if feats is not None:
            with torch.no_grad():
                self.model.eval()
                prob_raw, prob_smooth = self.model(feats.to(self.device))
            
            for i, lab in enumerate(prob_smooth):
                outputs[i*self.nshift:(i+1)*self.nshift, 1] = lab

            self.prev_lab = lab

        return outputs.detach().cpu().numpy()

"""
"""
class PostProc(lib.pipeline.Processor):
    def __init__(self, fs, margin_begin, margin_end, shift_time=0.2):
        self.n_buf = int(fs * (margin_begin + margin_end))
        self.n_margin_begin = int(fs * margin_begin)
        self.n_margin_end = int(fs * margin_begin)
        
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
                    #print(f'[LOG]: segment: {self.time_start:.2f} {self.time_end:.2f}              ')
                    
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

