# 이질성을 견디는 proximal term — FedProx

> **Federated Optimization in Heterogeneous Networks (FedProx)** — Li et al., 2020, MLSys (arXiv:1812.06127)
> 원문: https://arxiv.org/abs/1812.06127 · 분류: 바이블

FedAvg[2]는 클라이언트 데이터가 비슷하다고(IID에 가깝다고) 가정할 때 잘 작동합니다. 그러나 현실은 제각각(non-IID)이고, 기기 성능도 제각각(system heterogeneity)입니다. 한 사람의 키보드 데이터와 다른 사람의 키보드 데이터는 어휘부터 다르고, 최신 폰과 구형 폰은 처리 속도가 다릅니다. FedProx는 이 두 이질성을 정면으로 다뤄, 이질적 연합학습의 표준 베이스라인이 되었습니다. 핵심은 단 하나의 항을 더하는 것입니다.

## 배경: 두 가지 이질성

연합학습 현장에는 두 종류의 이질성이 있습니다. 첫째, 통계적 이질성(non-IID)입니다. 클라이언트마다 데이터 분포가 달라, 로컬 학습이 서로 다른 방향으로 끌고 가 글로벌 모델이 흔들립니다. 이를 client drift라고 합니다. 각 클라이언트가 자기 데이터에 최적화되며 멀어지면, 평균을 내도 좋은 모델이 안 나옵니다. 둘째, 시스템 이질성(straggler)입니다. 기기 성능 차로 어떤 클라이언트는 정해진 로컬 학습을 다 못 끝냅니다. FedAvg는 이런 부분 작업을 잘 못 다룹니다. 다 못한 클라이언트를 버리면 데이터가 편향되고, 그냥 받으면 학습이 불안정합니다.

> 핵심 질문: 클라이언트들이 제각각 흩어지지 않게 글로벌에 묶어 두면서, 부분적으로만 학습한 클라이언트도 안전하게 받아들일 수 없을까?

## 핵심 아이디어: 글로벌에 고무줄로 묶기

FedAvg의 로컬 목적함수에 proximal term(근접항)을 더합니다. "로컬에서 학습하되, 현재 글로벌 모델에서 너무 멀어지지는 말라"는 것입니다. 글로벌에서 멀어질수록 벌점을 줘, 클라이언트들이 제각각 튀는 것을 억제합니다.

> 비유 — 고무줄로 묶기. 각 클라이언트(요리사)가 자기 노하우로 레시피를 고치되, 표준 레시피에 고무줄로 묶여 있어 너무 멀리는 못 갑니다. 모두가 적당히 묶인 채 개선하니, 평균을 내도 한 방향으로 모입니다. 이 한 항 덕분에 부분적으로만 학습한 클라이언트(straggler)의 업데이트도 안전하게 수용됩니다.

## 수식으로 들어가기

수정된 로컬 목적함수는 다음과 같습니다.

```
min  F_k(w) + (μ/2)·‖w − w_global‖²
     └─보통의 로컬 손실   └─proximal term: 글로벌에서의 이탈을 벌함
```

앞항 `F_k(w)`는 클라이언트 k의 보통 로컬 손실이고, 뒷항이 핵심입니다. 현재 글로벌 모델 `w_global`에서 멀어진 거리의 제곱에 비례하는 벌점입니다. μ가 크면 강하게 묶어 안정적이지만 느리고, 작으면 자유롭게 학습합니다. 흥미롭게도 μ=0이면 FedProx는 정확히 FedAvg가 됩니다. 즉 FedProx는 FedAvg의 일반화입니다. proximal term이 학습을 안정화하므로, 각 클라이언트가 가능한 만큼만 학습해도(부분 작업) 안정성이 보장됩니다.

| 항목 | FedAvg | FedProx |
|---|---|---|
| 로컬 목적 | 로컬 손실만 | 로컬 손실 + proximal |
| non-IID | 흔들림(drift) | 묶어서 완화 |
| straggler | 다루기 어려움 | 부분 작업 수용 |

## 임팩트: 왜 바이블이 되었나

- **이질적 연합학습의 표준 베이스라인**입니다. non-IID와 straggler를 다루는 거의 모든 후속 연구의 비교 대상입니다.
- **수렴 보장**을 제공합니다. 이질적 환경에서의 이론적 수렴 분석을 제시해, 실용성과 이론을 함께 갖췄습니다.
- "글로벌에 묶는 정규화"라는 아이디어는 개인화(Ditto[4])로도 이어집니다. 같은 발상을 다른 곳에 적용하는 것이죠.

## 한계와 주의점

- **μ 튜닝**이 필요합니다. 너무 강하면 로컬 적응을 막고, 약하면 drift가 남습니다. 데이터마다 조정해야 합니다.
- **이질성을 완화할 뿐 해결은 아닙니다.** 극심한 non-IID에서는 단일 글로벌 모델 자체의 한계가 남습니다. 개인화(Ditto·FedBN)가 이를 넘으려 합니다.
- **통신 구조는 그대로**입니다. FedAvg와 같은 통신 패턴이라 통신 비용 문제는 별도 해법이 필요합니다.

## 더 읽을거리와 연결

- 이 책 내부: 앞 글 **FedAvg**[2](일반화의 원형), **Ditto**[4](정규화를 개인화로).
- 기초편 07·심화편 07(non-IID·client drift)과 함께 보면 좋습니다.

## 한 줄 요약

> FedProx는 로컬 학습에 proximal term을 더해 '글로벌에서 너무 멀어지지 말라'고 묶음으로써 non-IID와 straggler를 견디게 한, 이질적 연합학습의 표준 베이스라인(FedAvg의 일반화)이다.

## 참고문헌

[1] T. Li, A. K. Sahu, M. Zaheer, M. Sanjabi, A. Talwalkar, V. Smith, "Federated Optimization in Heterogeneous Networks (FedProx)," MLSys 2020. arXiv:1812.06127. https://arxiv.org/abs/1812.06127
[2] B. McMahan et al., "Communication-Efficient Learning of Deep Networks from Decentralized Data (FedAvg)," AISTATS 2017. arXiv:1602.05629.
[4] T. Li, S. Hu, A. Beirami, V. Smith, "Ditto: Fair and Robust Federated Learning Through Personalization," ICML 2021.

> 다음 → [3_바이블_SecureAggregation.md](./3_바이블_SecureAggregation.md)
