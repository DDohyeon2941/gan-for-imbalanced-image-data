# 원본 보존과 변경 원칙

## 보존한 것

- `experiments/2019_baselines/original/`의 연구 당시 Python 코드와 디렉터리 구조
- `experiments/2023_model_exploration/branches/main/`의 원본 브랜치 snapshot
- `experiments/2023_model_exploration/branches/1-feature-a/`의 원본 브랜치 snapshot
- 2019 checkpoint, loss graph 및 로컬 생성 산출물
- 참고 구현의 README와 LICENSE

2019 원본 Python 32개와 2023 `1-feature-a` Python 29개는 복사 당시 source와 SHA-256 hash를 비교해 내용이 같음을 확인했습니다.

## 정리 과정에서 바꾼 것

- 두 연구의 시점과 성격을 구분하는 한국어 문서를 추가했습니다.
- 2023 코드를 원본 Git 브랜치별 디렉터리로 분리했습니다.
- checkpoint 추적을 위한 Git LFS 규칙을 추가했습니다.
- 대량 생성 이미지, dataset, cache와 toy output을 Git 추적에서 제외했습니다.
- 원본을 수정하지 않고 검증하는 `tools/toy_smoke_test.py`와 최소 requirements를 추가했습니다.

## 바꾸지 않은 것

- 모델 구조, loss, optimizer와 hyperparameter
- label conditioning과 sampling 방식
- dataset imbalance 구성 로직
- 상대경로와 checkpoint naming
- 과거 API 사용 방식
- 결과 수치와 연구 결론

## 향후 변경 규칙

실행 호환성이 필요하면 원본 파일을 덮어쓰지 않고 별도 wrapper 또는 정리된 재현 코드로 만듭니다. 원본과 동작이 달라지는 경우 파일별로 변경 이유와 영향 범위를 문서화합니다. 특히 loss 누적, logits 처리, BAGAN conditioning, 평가 지표 계산은 설명 없이 고치지 않습니다.
