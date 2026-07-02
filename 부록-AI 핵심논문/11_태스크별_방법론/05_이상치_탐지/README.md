# 05 · 이상치 탐지 — 논문편 (바이블 → 최신)

라벨 없이 "정상에서 벗어난 소수"를 어떻게 정량화할까요. 밀도로? 경계로? 고립의 쉬움으로? 표현 학습으로? 그 답의 계보가 이 테마입니다. 이상탐지는 "정상을 무엇으로 모형화하는가"의 렌즈 교체사이며, 여섯 편의 논문이 그 렌즈가 바뀌어 온 과정을 보여 줍니다.

## 바이블에서 최신까지, 한눈에

| 분류 | 논문 | 핵심 한 줄 |
|---|---|---|
| 바이블 | LOF (2000) | 국소 밀도비로 다밀도 데이터의 이상을 잡다 |
| 바이블 | One-Class SVM (2001) | SVM을 비지도로 — 정상영역의 support를 추정 |
| 바이블 | Isolation Forest (2008) | 랜덤 분할로 이상을 직접 고립, 선형 시간 |
| 최신 | Deep SVDD (2018) | 정상을 잠재 공간의 최소 초구로 매핑 |
| 최신 | Anomaly Transformer (2022) | association discrepancy로 시계열 이상 |
| 최신 | Deep Isolation Forest (2023) | 신경망 표현 위 축평행 분할 — iForest의 딥 확장 |

## 이 테마를 관통하는 한 문장

거리/밀도(LOF)와 경계(One-Class SVM)로 출발해, "이상을 직접 고립"이라는 발상의 전환(Isolation Forest)이 표준을 세웠고, 딥러닝이 표현(Deep SVDD)·시계열(Anomaly Transformer)·고립의 재해석(Deep iForest)으로 그 위를 다시 썼습니다. "정상을 무엇으로 모형화하는가"의 렌즈가 거듭 교체된 역사입니다.

## 읽는 순서

| 순서 | 파일 | 한 줄 소개 |
|:---:|---|---|
| 1 | [1_바이블_LOF.md](./1_바이블_LOF.md) | 절대 거리에서 상대 밀도비로 |
| 2 | [2_바이블_OneClassSVM.md](./2_바이블_OneClassSVM.md) | 정상만으로 경계를 학습하는 커널법 |
| 3 | [3_바이블_IsolationForest.md](./3_바이블_IsolationForest.md) | 정상을 모형화하지 말고 이상을 고립하라 |
| 4 | [4_최신_DeepSVDD.md](./4_최신_DeepSVDD.md) | One-Class를 딥러닝으로 끌어올리다 |
| 5 | [5_최신_AnomalyTransformer.md](./5_최신_AnomalyTransformer.md) | 시계열 이상을 어텐션의 불일치로 |
| 6 | [6_최신_DeepIsolationForest.md](./6_최신_DeepIsolationForest.md) | 표현 학습과 고립을 결합 |
| — | [정리노트.md](./정리노트.md) | 계보표 + 한 줄 요약 + 연결 |

## 기초·심화편과의 연결

- 기초편 05(이상 점수·z-score·거리·PR-AUC), 심화편 05(밀도·고립·경계·딥·시계열 갈래 비교와 평가의 난점). 이 논문들이 그 갈래의 원전입니다.

> 다음 → [1_바이블_LOF.md](./1_바이블_LOF.md)
