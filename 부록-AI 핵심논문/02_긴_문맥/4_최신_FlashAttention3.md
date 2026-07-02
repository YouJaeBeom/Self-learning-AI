# 최신 하드웨어로 한 번 더 — FlashAttention-3

> **FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision** — Shah et al. (+Dao), 2024, NeurIPS Spotlight (arXiv:2407.08608)
> 원문: https://arxiv.org/abs/2407.08608 · 분류: 최신

앞 글의 FlashAttention[2]은 메모리 병목을 풀었지만 한 가지 숙명을 안고 있습니다. 하드웨어에 특화돼 있다는 점입니다. GPU 세대가 바뀌면 그 새 칩이 가진 능력을 다시 끌어내야 합니다. FlashAttention-3는 NVIDIA의 최신 GPU 아키텍처인 Hopper(H100)에 맞춰 어텐션 커널을 한 번 더 밀어붙입니다. 알고리즘은 그대로지만, 새 칩의 비동기 실행과 저정밀 연산 기능을 최대한 활용해 같은 정확 어텐션을 1.5~2배 빠르게 만듭니다. 이 글은 "왜 어텐션 커널은 GPU 세대마다 다시 짜야 하는가"라는 질문을 통해 하드웨어와 알고리즘이 맞물리는 지점을 살펴봅니다.

## 배경: 첫 FlashAttention이 놓친 것

2022년의 FlashAttention은 당시 GPU(주로 Ampere, A100)를 기준으로 설계되었습니다. 그런데 2022~2024년 사이 GPU 아키텍처는 크게 바뀌었습니다. Hopper 세대는 두 가지 새로운 무기를 들고 왔습니다.

- **비동기 실행 유닛**: 데이터를 메모리에서 가져오는 작업(TMA, Tensor Memory Accelerator)과 행렬 곱(WGMMA, warpgroup matrix multiply)을 서로 독립적으로, 동시에 진행할 수 있게 되었습니다. 기존에는 "데이터 가져오고 → 계산하고 → 다시 가져오고"를 순차적으로 했다면, 이제 가져오면서 동시에 계산할 수 있습니다.
- **FP8 저정밀 연산**: 8비트 부동소수점으로 행렬 곱을 하면 16비트(FP16/BF16)보다 처리량이 약 2배입니다.

첫 FlashAttention은 이 새 기능들을 활용하도록 설계되지 않았기 때문에, H100의 이론적 성능을 35% 정도밖에 끌어내지 못했습니다. 비싼 최신 칩의 절반 이상이 놀고 있었던 셈입니다.

> 핵심 질문: 최신 GPU의 비동기·저정밀 기능을 살려, 같은 정확 어텐션을 더 빠르게 돌릴 수 없을까?

## 핵심 아이디어: 하드웨어의 새 능력을 최대한 겹쳐 쓴다

FlashAttention-3의 핵심은 새 하드웨어 기능을 최대한 활용해 연산 유닛이 노는 시간을 없애는 것입니다. 세 가지 기법이 핵심입니다.

- **워프 특화(warp specialization)와 비동기**: 일부 워프(GPU 스레드 묶음)는 데이터를 가져오는 일(producer)에, 다른 워프는 계산하는 일(consumer)에 전담시켜, 적재와 연산을 파이프라인으로 겹칩니다.
- **연산 겹치기(GEMM-softmax overlap)**: 어텐션은 행렬 곱(GEMM)과 softmax가 번갈아 나오는데, softmax(지수·합산)는 행렬 곱과 다른 종류의 연산 유닛을 씁니다. 한 블록의 softmax를 계산하는 동안 다음 블록의 행렬 곱을 동시에 돌려, 두 유닛을 모두 바쁘게 유지합니다.
- **FP8과 비간섭 처리(incoherent processing)**: FP8로 처리량을 높이되, 정밀도 손실을 줄이기 위해 값을 무작위 직교 변환으로 분산시켜 이상치(outlier)의 영향을 완화합니다.

> 비유 — 주방의 동선 최적화. 같은 요리(어텐션)라도, 굽는 동안 동시에 다음 재료를 손질하고(비동기 겹치기), 무거운 도구 대신 가벼운 도구로 빠르게 다루면(FP8) 같은 시간에 더 많은 접시가 나옵니다. 레시피(알고리즘)는 그대로지만 주방 운영(구현)을 바꾼 것입니다.

## 어떻게 작동하나

```
기존 FlashAttention: [적재] → [GEMM] → [softmax] → [적재] → …  (순차, 유닛이 번갈아 놀음)
FlashAttention-3:    [적재 ┃ GEMM ┃ softmax]  모두 겹쳐서 동시 진행
```

이 겹치기 덕분에 H100에서 FP16 기준 약 1.5~2배, FP8 기준으로는 더 높은 처리량(보고된 바로 약 1.2 PFLOPS 근처)을 달성하면서, 여전히 **정확 어텐션**을 유지합니다. 평문으로 요약하면 "어텐션의 알고리즘은 같지만, 최신 칩의 비동기·저정밀 기능에 맞춰 구현을 다시 짜 더 빠르게 만들었다"가 됩니다.

## 임팩트: 무엇을 바꿨나

- **최신 하드웨어에서 긴 문맥의 기준 커널**이 되었습니다. H100에서 LLM을 학습·서빙하는 곳이라면 사실상 표준으로 쓰입니다.
- "알고리즘은 그대로, 하드웨어에 맞춰 다시 짠다"는 FlashAttention의 철학이 한 번의 작업이 아니라 지속적 갱신임을 보여 줍니다. 새 GPU 세대(Blackwell 등)가 나올 때마다 이 작업이 반복될 것입니다.
- 긴 문맥과 대규모 서빙(T03)의 실효 비용을 낮춰, 실전 배포에 직접 기여합니다. 같은 전기료로 더 많은 토큰을 처리할 수 있게 만드는 일입니다.

## 한계와 주의점

- **특정 하드웨어(Hopper)에 특화**되어 있습니다. 다른 칩에서는 이점이 그대로 옮겨지지 않으며, 구형 GPU 사용자에게는 해당되지 않습니다.
- FP8 같은 저정밀은 정밀도 관리가 필요해, 모든 상황에서 무조건 켜는 것이 아닙니다. 학습 안정성이나 정확도가 중요한 경우 신중히 적용해야 합니다.
- 어텐션의 근본 복잡도 `O(n²)`는 여전히 남습니다. 단일 칩의 효율을 짜내는 것이라, 초장문맥은 결국 분산(다음 글 Ring)이 필요합니다.

## 더 읽을거리와 연결

- 이 책 내부: 앞 글 **FlashAttention**[2](원리), 다음 글 **Ring Attention**(분산으로 더 길게), **T03 효율·서빙**.
- 심화편 S3·S7(효율·서빙)와 함께 보면 좋습니다.

## 한 줄 요약

> FlashAttention-3는 최신 GPU(Hopper)의 비동기·FP8 기능에 맞춰 정확 어텐션 커널을 다시 짜 1.5~2배 빠르게 만든다. "알고리즘은 그대로, 하드웨어에 맞춰 갱신한다"는 철학의 최신판이다.

## 참고문헌

[1] J. Shah et al., "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision," NeurIPS 2024 Spotlight. arXiv:2407.08608. https://arxiv.org/abs/2407.08608
[2] T. Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness," NeurIPS 2022. arXiv:2205.14135.
[3] R. Y. Liu, M. Zaharia, P. Abbeel, "Ring Attention with Blockwise Transformers for Near-Infinite Context," 2023. arXiv:2310.01889.

> 다음 → [5_최신_RingAttention.md](./5_최신_RingAttention.md)
