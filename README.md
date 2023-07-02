# ECAPA CNN-TDNN

This project is a reimplementation of [ECAPA CNN-TDNN](https://arxiv.org/pdf/2104.02370.pdf)-based speaker verification.

It works based on [SpeechBrain](https://github.com/speechbrain/speechbrain), an open-source AI speech toolkit built on PyTorch.

## Introduction


## Performance

The VoxCeleb2 dataset (5,944 speakers) was used for training. (Data augmentation was skipped due to limited computational resources. I plan to update the results with data augmentation in the future.) 

When the VoxCeleb1-O dataset (40 speakers) was used for testing, the following speaker verification results were obtained:
|  Architecture  |    EER (%)  | minDCF |
|:--------------:|:-----------:| ------:|
|   ECAPA-TDNN   |             |        |
| ECAPA CNN-TDNN |             |        |

## Requirements

This project follows the requirements of [SpeechBrain](https://github.com/speechbrain/speechbrain).

For your information, I used the Docker image called [gastron/speechbrain-ci](https://hub.docker.com/r/gastron/speechbrain-ci).
```
docker pull gastron/speechbrain-ci
```

## Usage

### Training

For training, run:
```
python train_speaker_embeddings.py hparams/train_ecapa_cnn_tdnn.yaml --data_folder="my data"
```

### Test

For test, run:
```
python speaker_verification_cosine.py hparams/verification_ecapa_cnn_tdnn.yaml --data_folder="my data"
```



## Acknolwedgement

This code is heavily based on the SpeechBrain's recipe of ECAPA-TDNN.

I would like to acknowledge their contributions and the use of their code as a foundation for this project.
