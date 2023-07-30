import torch
from torchinfo import summary

from models.ECAPA_CNN_TDNN import ECAPA_TDNN
from models.ECAPA_CNN_TDNN import ECAPA_CNN_TDNN

if __name__=="__main__":
    input_feats = torch.rand([5, 120, 80])
    compute_embedding = ECAPA_TDNN(80, lin_neurons=192)
    # compute_embedding = ECAPA_TDNN(80, lin_neurons=192, channels= [1024, 1024, 1024, 1024, 1536])
    # compute_embedding = ECAPA_TDNN(80, lin_neurons=192, channels= [2048, 2048, 2048, 2048, 2048, 1536], kernel_sizes= [5, 3, 3, 3, 3, 1], dilations=[1, 2, 3, 4, 5, 1], groups=[1, 1, 1, 1, 1, 1])
    outputs = compute_embedding(input_feats)
    print(outputs.shape)
    summary(compute_embedding)

    input_feats = torch.rand([5, 120, 80])
    compute_embedding = ECAPA_CNN_TDNN(int(128*80/4), lin_neurons=192) #2560=128*80/4
    outputs = compute_embedding(input_feats)
    print(outputs.shape)
    summary(compute_embedding)
