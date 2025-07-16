import math
import torch
from huggingface_hub import PyTorchModelHubMixin


class PositionalEncoding(torch.nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = torch.nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        pe = pe.permute(1,0,2)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:,:x.size(1),:]
        return self.dropout(x)


class SITEVoiceClassifier(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(self, n_fbin=256, n_fwd=0, n_bwd=50, n_smooth=1, n_decimate=2,
                 tf_dims=512, tf_nhead=8, tf_num_layers=2):
        """
        Scale-invariant (SI) Transformer-Encoder (TE) Voice Classifier
        """
        super(SITEVoiceClassifier, self).__init__()

        #
        self.n_fbin = n_fbin
        self.n_fwd = n_fwd
        self.n_bwd = n_bwd
        self.n_frame = (1 + n_fwd + n_bwd)
        self.n_label = n_smooth
        
        self.n_decimate = n_decimate
        self.n_half = math.ceil(self.n_frame / self.n_decimate)
        self.n_quarter = math.ceil(self.n_half / self.n_decimate)
        
        # 
        self.norm = torch.nn.LayerNorm(self.n_fbin)
        self.posenc = PositionalEncoding(self.n_fbin)

        # 
        self.enclayer_full = torch.nn.TransformerEncoderLayer(self.n_fbin,
                                                              dim_feedforward=tf_dims, nhead=tf_nhead,
                                                              batch_first=True)
        self.transformer_full = torch.nn.TransformerEncoder(self.enclayer_full, num_layers=1)

        self.enclayer_half = torch.nn.TransformerEncoderLayer(self.n_fbin,
                                                              dim_feedforward=tf_dims, nhead=tf_nhead,
                                                              batch_first=True)
        self.transformer_half = torch.nn.TransformerEncoder(self.enclayer_half, num_layers=1)

        self.enclayer_quarter = torch.nn.TransformerEncoderLayer(self.n_fbin,
                                                                 dim_feedforward=tf_dims,
                                                                 nhead=tf_nhead,
                                                                 batch_first=True)
        self.transformer_quarter = torch.nn.TransformerEncoder(self.enclayer_quarter, num_layers=tf_num_layers)

        self.linear = torch.nn.Linear(self.n_fbin * self.n_quarter, self.n_label)
 
        
    def forward(self, pspec, **kwargs):
        """
        pspec: block-wise amplitude spectrogram [Batch, TimeFrame, FreqBin]
        """
        [B, C, D] = pspec.shape
        
        # block-wise explicit scale-normalization
        scales = torch.mean(pspec, dim=(1,2)).reshape(B,1,1)
        pspec = pspec / (scales + 1.0e-6)

        # discrimination
        predicts = self.norm(pspec)
        predicts = self.transformer_full(self.posenc(predicts))
        predicts = self.transformer_half(predicts[:,0::self.n_decimate,:])
        predicts = self.transformer_quarter(predicts[:,0::self.n_decimate,:])
        predicts = self.linear(predicts.reshape(B, self.n_quarter * self.n_fbin))

        #####
        if 'probs' in kwargs:
            probs = torch.sigmoid(predicts)          
            return probs

        return predicts


class MixedFilterbank(torch.nn.Module):
    def __init__(self, input_dim, output_dim, eps=1.0e-16):
        super(MixedFilterbank, self).__init__()
        output_dim_h = int(output_dim/2)
        self.eps = eps
        
        # linear-complexabs-pow
        self.bank_c = torch.nn.Linear(input_dim, output_dim)
        torch.nn.init.xavier_normal_(self.bank_c.weight)
        self.pow = torch.nn.Parameter(torch.ones(output_dim_h) * 1.0e-1)
        
        # log-linear
        self.bank_r = torch.nn.Linear(input_dim, output_dim_h)
        torch.nn.init.xavier_normal_(self.bank_r.weight)
        
    def forward(self, inputs, **kwargs):
        fbank_c = self.bank_c(inputs)
        fbank_c = torch.complex(fbank_c[:,:,0::2], fbank_c[:,:,1::2])
        fbank_c = torch.pow(torch.abs(fbank_c)+self.eps, torch.abs(self.pow))
        
        fbank_r = self.bank_r(torch.log(torch.abs(inputs)+self.eps))
        return torch.cat([fbank_c, fbank_r], dim=2)


class SITEVoiceClassifierV2(torch.nn.Module, PyTorchModelHubMixin):
    def __init__(self, n_fbin, fbin_l=1, fbin_h=193, n_fbank=128, n_fwd=0, n_bwd=50, n_decimate=2,
                 tf_dims=256, tf_nhead=8, tf_num_layers=1, n_dense=32):
        super(SITEVoiceClassifierV2, self).__init__()

        #
        self.n_fbin = n_fbin
        self.n_fbank = n_fbank
        self.fbin_l = fbin_l
        self.fbin_h = fbin_h
        self.n_fwd = n_fwd
        self.n_bwd = n_bwd
        self.n_frame = (1 + n_fwd + n_bwd)
        self.n_class = 1
        self.n_dense = n_dense

        self.n_decimate = n_decimate
        self.n_half = math.ceil(self.n_frame / self.n_decimate)
        self.n_quarter = math.ceil(self.n_half / self.n_decimate)
        
        # 
        self.mfbank = MixedFilterbank(self.fbin_h - self.fbin_l, self.n_fbank)
        self.norm = torch.nn.LayerNorm(self.n_fbank)
        self.posenc = PositionalEncoding(self.n_fbank)

        # 
        enclayer_full = torch.nn.TransformerEncoderLayer(self.n_fbank, dim_feedforward=tf_dims, nhead=tf_nhead, batch_first=True)
        self.transformer_full = torch.nn.TransformerEncoder(enclayer_full, num_layers=1)
        
        enclayer_half = torch.nn.TransformerEncoderLayer(self.n_fbank, dim_feedforward=tf_dims, nhead=tf_nhead, batch_first=True)
        self.transformer_half = torch.nn.TransformerEncoder(enclayer_half, num_layers=1)
        
        enclayer_quarter = torch.nn.TransformerEncoderLayer(self.n_fbank, dim_feedforward=tf_dims, nhead=tf_nhead, batch_first=True)
        self.transformer_quarter = torch.nn.TransformerEncoder(enclayer_quarter, num_layers=tf_num_layers)

        self.linear_dense = torch.nn.Linear(self.n_fbank * self.n_quarter, self.n_dense)
        self.linear_out = torch.nn.Linear(self.n_dense, self.n_class)
        
    """
    spec: spectrogram [Batch, TimeFrame, FreqBin]
    """
    def forward(self, pspec, **kwargs):
        """
        pspec: block-wise amplitude spectrogram [Batch, TimeFrame, FreqBin]
        """
        [B, C, D] = pspec.shape
        
        # discrimination
        pspec = pspec[:,:,self.fbin_l:self.fbin_h]

        # block-wise explicit scale-normalization
        scales = torch.mean(pspec, dim=(1,2)).reshape(B,1,1)
        predicts = pspec / (scales + 1.0e-12)

        predicts = self.mfbank(predicts)
        predicts = self.norm(predicts)

        predicts = self.transformer_full(self.posenc(predicts))
        predicts = self.transformer_half(predicts[:,0::self.n_decimate,:])
        predicts = self.transformer_quarter(predicts[:,0::self.n_decimate,:])
        predicts = self.linear_dense(predicts.reshape(B, self.n_quarter * self.n_fbank))
        predicts = torch.sigmoid(predicts)
        predicts = self.linear_out(predicts)

        #####
        if 'probs' in kwargs:
            probs = torch.sigmoid(predicts)          
            return probs

        return predicts
