# 원본 연구 코드 매핑

## 2019 로컬 프로젝트

원본 디렉터리 구조를 `experiments/2019_baselines/original/` 아래에 보존했습니다.

| 원본 파일 | 역할 | 현재 위치 |
| --- | --- | --- |
| `extract_imbalanced_index.py` | 클래스별 subset index 생성 | `experiments/2019_baselines/original/extract_imbalanced_index.py` |
| `model.py` | MNIST/CIFAR-10 Vanilla GAN | `experiments/2019_baselines/original/model.py` |
| `main.py` | Vanilla GAN 학습 | `experiments/2019_baselines/original/main.py` |
| `oversampling.py` | checkpoint 기반 이미지 생성 | `experiments/2019_baselines/original/oversampling.py` |
| `AC_GAN/model.py` | ACGAN 모델 | `experiments/2019_baselines/original/AC_GAN/model.py` |
| `AC_GAN/main.py` | ACGAN 학습 | `experiments/2019_baselines/original/AC_GAN/main.py` |
| `Classification/cnn_model.py` | LeNet/ResNet | `experiments/2019_baselines/original/Classification/cnn_model.py` |
| `Classification/cnn_main.py` | downstream 분류 평가 | `experiments/2019_baselines/original/Classification/cnn_main.py` |

## 2023 원본 GitHub

```text
원본 main        → experiments/2023_model_exploration/branches/main/
원본 1-feature-a → experiments/2023_model_exploration/branches/1-feature-a/
```

### `main`: 초기 구조 실험 9개

| 파일 | 역할 |
| --- | --- |
| `vanilla_gan_1025.py` | Unconditional GAN 기준 실험 |
| `cgan_1024.py`, `cgan_1024_1.py` | Conditional GAN 실험 |
| `ac_gan.py`, `ac_gan_binary.py` | Auxiliary classifier 구조 실험 |
| `wgan_1025.py` | Weight clipping WGAN |
| `wgan_gp.py` | Gradient penalty WGAN |
| `bagan.py` | Fully connected BAGAN 계열 탐색 |
| `bagan_conv.py` | Convolutional BAGAN 계열 탐색 |

### `1-feature-a`: 후속 파이프라인 29개

| 묶음 | 대표 파일 | 역할 |
| --- | --- | --- |
| 생성 | `generate_images_[wgan_gp_conv_cifar10].py` | 학습된 Generator로 이미지 저장 |
| 분류 | `train_classifier.py`, `train_classifier_imb.py` | 증강 실험과 불균형 baseline |
| 연결 | `gan_to_classifier.py` | GAN 생성 데이터를 classifier에 연결 |
| Edge | `implementation_edge.py`, `gan_to_classifier_edge.py` | boundary/edge sample 실험 |
| 공통 모델 | `models_gan.py`, `models_classifier.py` | GAN과 classifier 정의 |
| BAGAN 변형 | `bagan_conv_cifar10*.py` | CIFAR-10 구조·조건 변형 |
| WGAN-GP 변형 | `wgan_gp*_cifar10*.py`, `proposed_wgan_gp*.py` | convolutional 및 제안 구조 탐색 |

브랜치 사이에 이름이 같은 파일도 각 snapshot을 정확히 보존하기 위해 양쪽에 유지했습니다.
