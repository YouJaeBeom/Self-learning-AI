# 태스크별 방법론 — 핵심논문 트랙 (Part 2)

Part 1이 LLM과 딥러닝의 프런티어를 문제별로 다뤘다면, Part 2는 목적(태스크)별 방법론의 원전으로 들어갑니다. 분류, 군집화, 시계열, 이상치 탐지처럼 실무에서 늘 마주치는 8개 태스크마다, 그 분야를 연 고전 바이블과 최근 판을 바꾼 임팩트작을 나란히 읽습니다. 모든 인용은 편집자가 웹으로 검증한 [마스터 리스트](../_편집실/검증된_논문_마스터_태스크별.md)에서 가져왔습니다.

## 8개 태스크 (각 폴더 = 바이블 → 최신)

| # | 태스크 | 바이블 → 최신 (대표) |
|:--:|---|---|
| 01 | [분류·회귀](./01_분류와_회귀/) | Random Forest·XGBoost → LightGBM·CatBoost·TabPFN(Nature) |
| 02 | [군집화](./02_군집화/) | DBSCAN·Spectral → HDBSCAN·Contrastive Clustering·SCAN |
| 03 | [차원축소·표현학습](./03_차원축소와_표현학습/) | t-SNE·UMAP → SimCLR·MAE |
| 04 | [시계열 분석](./04_시계열_분석/) | Box-Jenkins·DeepAR → N-BEATS·PatchTST·Chronos·TimesFM |
| 05 | [이상치 탐지](./05_이상치_탐지/) | LOF·OC-SVM·Isolation Forest → Deep SVDD·Anomaly Transformer |
| 06 | [그래프·GNN](./06_그래프와_GNN/) | DeepWalk·node2vec·GCN·GraphSAGE → GAT·GraphGPS |
| 07 | [연합학습](./07_연합학습/) | FedAvg·FedProx·Secure Agg → Ditto·FedBN·FederatedScope-LLM |
| 08 | [컨티뉴얼 러닝](./08_컨티뉴얼_러닝/) | EWC·iCaRL·LwF → L2P·Experience Replay |

각 태스크는 기초 → 심화 → 논문(이 트랙) 순으로 읽으면 원리에서 원전까지 이어집니다.

> 핵심논문편 메인 → [README](../README.md) · 검증된 논문 마스터 → [태스크별 마스터](../_편집실/검증된_논문_마스터_태스크별.md)
