# 기존 코드 구성과 새 위치

## 2019년 기존 GAN 비교 프로젝트

| 기존 경로 | 역할 | 정리 후 위치 |
| --- | --- | --- |
| `extract_imbalanced_index.py` | 클래스별 subset 추출 | `experiments/2019_baselines/original/extract_imbalanced_index.py` |
| `model.py` | MNIST/CIFAR-10 Vanilla GAN 모델 | `experiments/2019_baselines/original/model.py` |
| `main.py` | Vanilla GAN 학습 | `experiments/2019_baselines/original/main.py` |
| `oversampling.py` | 체크포인트 기반 샘플 생성 | `experiments/2019_baselines/original/oversampling.py` |
| `AC_GAN/model.py` | ACGAN 모델 | `experiments/2019_baselines/original/AC_GAN/model.py` |
| `AC_GAN/main.py` | ACGAN 학습 | `experiments/2019_baselines/original/AC_GAN/main.py` |
| `Classification/cnn_model.py` | LeNet/ResNet 분류기 | `experiments/2019_baselines/original/Classification/cnn_model.py` |
| `Classification/cnn_main.py` | 합성 데이터 downstream 평가 | `experiments/2019_baselines/original/Classification/cnn_main.py` |

## 2023년 개인 후속 연구

| 원본 파일 | 실험 역할 |
| --- | --- |
| `vanilla_gan_1025.py` | Unconditional GAN 기준 실험 |
| `cgan_1024.py`, `cgan_1024_1.py` | Conditional GAN 실험 |
| `ac_gan.py`, `ac_gan_binary.py` | Auxiliary classifier 구조 실험 |
| `wgan_1025.py` | Weight clipping을 사용한 Wasserstein GAN 실험 |
| `wgan_gp.py` | Gradient penalty를 사용한 Wasserstein GAN 실험 |
| `bagan.py` | Fully connected BAGAN 계열 구조 실험 |
| `bagan_conv.py` | Convolutional BAGAN 계열 구조 실험 |
