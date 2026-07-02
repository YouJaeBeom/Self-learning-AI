# 07 · 연합학습 — 논문편 (바이블 → 최신)

데이터를 한곳에 모으지 않고 어떻게 함께 학습할까요. 그리고 프라이버시를 지키고, 이질성을 견디고, 거대 LLM까지 연합하려면 무엇이 필요할까요. 휴대폰, 병원, 기업처럼 데이터가 흩어져 있고 모을 수 없는 상황에서, 모델만 주고받아 함께 학습하는 것이 연합학습입니다. 이 테마는 그 발상이 현실의 이질성·프라이버시·규모를 만나며 진화한 계보를 여섯 편의 논문으로 따라갑니다.

## 바이블에서 최신까지, 한눈에

| 분류 | 논문 | 핵심 한 줄 |
|---|---|---|
| 바이블 | FedAvg (2017) | 연합학습을 창시 — 로컬 학습 + 가중평균 |
| 바이블 | FedProx (2020) | proximal term으로 non-IID·straggler 대응 |
| 바이블 | Secure Aggregation (2017) | 서버가 개별 업데이트를 못 보는 집계 |
| 최신 | Ditto (ICML 2021) | 정규화 개인화로 공정성·견고성 |
| 최신 | FedBN (ICLR 2021) | BN을 로컬 유지해 feature non-IID 완화 |
| 최신 | FederatedScope-LLM (2024) | LLM 연합 파인튜닝 종합 패키지 |

## 이 테마를 관통하는 한 문장

FedAvg가 연합학습을 열었고(2017), FedProx가 이질성을, Secure Aggregation이 프라이버시를 보강했으며, 개인화(Ditto·FedBN)를 거쳐 파운데이션 모델 시대(FederatedScope-LLM)로 확장됩니다. "데이터를 모으지 않고 배운다"는 한 발상이, 현실의 이질성·프라이버시·규모를 만나며 진화한 이야기입니다.

## 읽는 순서

| 순서 | 파일 | 한 줄 소개 |
|:---:|---|---|
| 1 | [1_바이블_FedAvg.md](./1_바이블_FedAvg.md) | 연합학습의 창시 — 가중평균 |
| 2 | [2_바이블_FedProx.md](./2_바이블_FedProx.md) | 이질성을 견디는 proximal term |
| 3 | [3_바이블_SecureAggregation.md](./3_바이블_SecureAggregation.md) | 서버도 개별을 못 보게 |
| 4 | [4_최신_Ditto.md](./4_최신_Ditto.md) | 개인화로 공정성·견고성 |
| 5 | [5_최신_FedBN.md](./5_최신_FedBN.md) | BN을 로컬에 — feature shift 완화 |
| 6 | [6_최신_FederatedScopeLLM.md](./6_최신_FederatedScopeLLM.md) | 연합 + 거대 LLM |
| — | [정리노트.md](./정리노트.md) | 계보표 + 한 줄 요약 + 연결 |

## 기초·심화편과의 연결

- 기초편 07(연합학습·FedAvg·프라이버시 동기), 심화편 07(non-IID·통신·개인화·보안 집계와 연합 LLM). 이 논문들이 그 원전입니다.

> 다음 → [1_바이블_FedAvg.md](./1_바이블_FedAvg.md)
