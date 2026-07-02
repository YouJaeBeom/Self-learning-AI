# 01 · 분류와 회귀 — 정형데이터를 푼 다섯 논문

표(table) 형태 정형데이터의 분류와 회귀를 가장 잘 푸는 방법은 무엇일까요. 한 그루의 결정트리는 약합니다. 이 테마는 그 약함을 모으고(배깅), 이어달리고(부스팅), 즉석에서 읽는(in-context) 길로 풀어 온 다섯 원전을 따라갑니다. 트리를 어떻게 모으고, 끝내 어떻게 넘어서는가의 이야기입니다.

## 바이블에서 최신까지, 한눈에

| 분류 | 논문 | 핵심 한 줄 |
|---|---|---|
| 바이블 | Random Forests (Breiman, 2001) | 배깅+무작위 특성으로 트리의 분산을 잡은 앙상블 |
| 바이블 | XGBoost (Chen & Guestrin, 2016) | 정규화·희소 인식·시스템 최적화로 부스팅 대규모화 |
| 최신 | LightGBM (Ke et al., 2017) | GOSS·EFB로 부스팅을 대폭 가속 |
| 최신 | CatBoost (Prokhorenkova et al., 2018) | ordered boosting으로 타깃 누수 제거 |
| 최신 | TabPFN v2 (Hollmann et al., 2025, Nature) | 합성 사전학습 트랜스포머의 in-context 정형 학습 |

## 이 테마를 관통하는 한 문장

한 그루 트리의 분산을 배깅으로 잡고(Random Forests, 2001), 편향을 부스팅으로 파고들어(XGBoost, 2016) 대규모화한 뒤, 그 부스팅을 더 빠르게(LightGBM)·더 안전하게(CatBoost) 다듬었고, 마침내 학습 패러다임 자체를 바꾼 정형 파운데이션 모델(TabPFN)이 등장했습니다. 분산을 모아 잡고, 편향을 파고들고, 빠르고 안전하게 다듬은 뒤, 피팅 자체를 넘어서는 흐름입니다.

## 읽는 순서

| 순서 | 파일 | 한 줄 소개 |
|:---:|---|---|
| 1 | [1_바이블_RandomForest.md](./1_바이블_RandomForest.md) | 배깅+무작위 특성 — 트리 떼의 분산 잡기 |
| 2 | [2_바이블_XGBoost.md](./2_바이블_XGBoost.md) | 정규화·희소 인식으로 부스팅을 표준으로 |
| 3 | [3_최신_LightGBM.md](./3_최신_LightGBM.md) | GOSS·EFB·히스토그램으로 속도 혁명 |
| 4 | [4_최신_CatBoost.md](./4_최신_CatBoost.md) | ordered boosting으로 미세 누수까지 차단 |
| 5 | [5_최신_TabPFN.md](./5_최신_TabPFN.md) | in-context로 피팅 없는 정형 예측 (Nature 2025) |
| — | [정리노트.md](./정리노트.md) | 계보표 + 한 줄 요약 + 연결 |

## 기초·심화편과의 연결

- 기초편 01(분류/회귀의 정의, 트리·선형, 배깅/부스팅의 직관), 심화편 01(부스팅의 원리, 3대 라이브러리 비교, 트리 대 딥러닝 논쟁).
- 보조 고전(별도 글 없이 본문에서 언급): SVM(Cortes & Vapnik, 1995), k-means(Lloyd, 1982).

> 다음 → [1_바이블_RandomForest.md](./1_바이블_RandomForest.md)
