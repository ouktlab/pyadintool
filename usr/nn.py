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
    
