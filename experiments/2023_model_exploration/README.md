# 2023년 개인 후속 연구

2019년 프로젝트에서 다룬 불균형 데이터 문제를 별도의 개인 연구 코드베이스에서 다시 탐구한
작업이다.

원본 저장소: <https://github.com/DDohyeon2941/gan_for_imbalanced_dataset>

여러 GAN objective와 conditional structure를 실험했다. `bagan_conv.py`는 fully connected
구조의 `bagan.py` 이후 convolutional encoder/decoder를 적용해 본 구조 탐색 코드로 정리한다.
완료된 비교 평가를 뒷받침할 자료가 충분하지 않으므로 검증된 최종 모델로 표현하지 않는다.

당시 저장소의 Python 파일 9개는 `original/`에 수정 없이 보존했다.
