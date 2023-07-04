import torch

from models.ECAPA_CNN_TDNN import ECAPA_TDNN
from models.ECAPA_CNN_TDNN import ECAPA_CNN_TDNN

input_feats = torch.rand([5, 120, 80])
compute_embedding = ECAPA_TDNN(80, lin_neurons=192)
outputs = compute_embedding(input_feats)
print(outputs.shape)

input_feats = torch.rand([5, 120, 80])
compute_embedding = ECAPA_CNN_TDNN(int(128*80/4), lin_neurons=192) #2560=128*80/4
outputs = compute_embedding(input_feats)
print(outputs.shape)