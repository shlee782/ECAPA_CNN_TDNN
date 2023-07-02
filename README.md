# ECAPA CNN-TDNN

This project is a reimplementation of [ECAPA CNN-TDNN](https://arxiv.org/pdf/2104.02370.pdf)-based speaker verification.

It works based on [SpeechBrain](https://github.com/speechbrain/speechbrain), which is an open-source AI speech toolkit built on PyTorch.

## Introduction

## Performance

The VoxCeleb2 dataset (5,944 speakers) was used for training. (Data augmentation was skipped due to limited computational resources. I plan to update the results with data augmentation in the future.) Note that the VoxCeleb1-O dataset (40 speakers) was utilized for test.

| Architecture   | VoxCeleb1-O |        |
|----------------|-------------|--------|
|                | EER         | minDCF |
| ECAPA-TDNN     |             |        |
| ECAPA CNN-TDNN |             |        |

## Requirements

This project follows the requirements of [SpeechBrain](https://github.com/speechbrain/speechbrain).

For your information, I used the docker image called [gastron/speechbrain-ci](https://hub.docker.com/r/gastron/speechbrain-ci).
```
docker pull gastron/speechbrain-ci
```

## Usage






## Acknolwedgement

This code is heavily based on the SpeechBrain's recipe of ECAPA-TDNN.

I'd like to acknowledge their contributions and the use of their code as a foundation for this project.
