# 메모리 왕복을 줄이다 — FlashAttention

> **FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness** — Dao et al., 2022, NeurIPS (arXiv:2205.14135)
> 원문: https://arxiv.org/abs/2205.14135 · 분류: 최신

위치 문제(RoPE[5]·ALiBi[6])를 풀었으니, 이제 긴 문맥의 두 번째 벽인 `O(n²)` 비용을 마주할 차례입니다. 그런데 FlashAttention의 통찰은 의외의 방향에서 옵니다. 어텐션이 느린 진짜 이유는 곱셈·덧셈의 양이 많아서가 아니라, **거대한 점수표를 느린 메모리에 썼다가 다시 읽는** 데 있다는 것입니다. 그 메모리 병목만 없애면, 결과를 한 글자도 바꾸지 않고도 어텐션을 몇 배 빠르게, 그리고 메모리를 선형으로 만들 수 있습니다. 이 한 편이 긴 문맥 시대를 실용 가능하게 만든 핵심 엔진이며, 오늘날 PyTorch와 Hugging Face의 기본 어텐션 경로로 들어가 있습니다.

## 배경: GPU의 메모리 계층을 알아야 한다

이 논문을 이해하려면 GPU 안에 속도가 다른 두 종류의 메모리가 있다는 사실을 알아야 합니다.

- **HBM(High Bandwidth Memory)**: GPU의 주 메모리. 수십 GB로 크지만 상대적으로 느립니다(대역폭 약 1.5~3 TB/s). 우리가 "GPU 메모리 40GB"라고 말할 때의 그 메모리입니다.
- **SRAM(on-chip memory)**: 연산 유닛 바로 옆의 초고속 메모리. 수십 MB로 매우 작지만 HBM보다 한 자릿수 이상 빠릅니다(약 19 TB/s).

표준 어텐션은 이렇게 동작합니다. 먼저 `Q·Kᵀ`로 `n×n` 점수 행렬 `S`를 만들어 HBM에 씁니다. 그다음 `S`를 HBM에서 다시 읽어 softmax를 적용하고 결과 `P`를 HBM에 씁니다. 마지막으로 `P`를 다시 읽어 `V`와 곱합니다. 길이 `n`이 8K라면 `S`는 6,400만 개 원소짜리 행렬이고, 이것을 HBM에 두 번 쓰고 두 번 읽습니다. 연산 유닛은 이 거대한 데이터가 오가기를 기다리며 놀게 됩니다. 즉 어텐션은 연산에 묶인(compute-bound) 작업이 아니라 메모리에 묶인(memory-bound) 작업이었습니다.

> 핵심 질문: 결과를 전혀 바꾸지 않고, 거대한 점수표를 HBM에 쓰는 메모리 병목만 없앨 수 없을까?

## 핵심 아이디어: 연산이 아니라 메모리 왕복을 줄인다

FlashAttention은 "연산량을 줄이자"가 아니라 "느린 메모리 왕복을 줄이자"라는 IO 인식(IO-aware) 관점에 집중합니다.

> 비유 — 거대한 장부를 책상 위에서 처리하기. 100만 행짜리 장부를 창고(HBM)에 펼쳐 두고 한 줄씩 오가며 합산하는 대신, 한 묶음씩 책상(SRAM)으로 가져와 처리하고 중간 합계만 들고 다음 묶음으로 넘어갑니다. 최종 합계는 완전히 같지만, 창고를 오가는 동선을 없애 시간을 크게 아낍니다.

## 어떻게 작동하나: 타일링과 온라인 softmax

이 아이디어를 실현하는 두 기법이 타일링과 온라인 softmax입니다.

**타일링(tiling)**: `n×n` 점수표를 작은 블록으로 쪼개, SRAM에 올라갈 만큼씩만 Q, K, V 블록을 가져와 처리합니다. 거대한 점수표 `S` 전체를 HBM에 저장하지 않는 것이 핵심입니다.

그런데 여기에 문제가 있습니다. softmax는 한 행 전체의 최댓값과 합을 알아야 정규화할 수 있는데, 블록 단위로 처리하면 그 행을 한 번에 다 볼 수 없습니다. 이를 푸는 것이 **온라인 softmax**입니다. 블록을 하나씩 보면서 지금까지의 최댓값 `m`과 정규화 합 `ℓ`을 누적·보정해 나갑니다.

```
각 블록을 볼 때마다:
  m_new = max(m_old, 현재 블록의 최댓값)      ← 최댓값 갱신
  ℓ_new = ℓ_old · exp(m_old − m_new) + (현재 블록 합)   ← 보정 후 누적
  출력 누적값도 같은 비율로 보정
```

새 블록에서 더 큰 값을 만나면, 이전까지 누적한 결과를 새 최댓값 기준으로 다시 스케일해 보정합니다. 이 점진적 보정 덕분에 전체 행을 한 번에 보지 않고도 수학적으로 정확히 같은 softmax를 완성할 수 있습니다.

```
계산하는 함수:  softmax(Q·Kᵀ/√d_k)·V   ← 표준 어텐션과 수학적으로 동일
HBM 메모리:    O(n²) → O(n)            ← n×n 표를 저장하지 않음
```

여기서 강조할 점은 이것이 **근사가 아니라 정확(exact) 어텐션**이라는 것입니다. 희소 어텐션이나 선형 어텐션은 일부 연결을 버리거나 근사해 품질을 깎지만, FlashAttention은 같은 답을 더 영리한 동선으로 계산할 뿐입니다.

| 항목 | 표준 어텐션 | FlashAttention |
|---|---|---|
| 계산 결과 | 정확 | 정확(동일) |
| HBM 메모리 | `O(n²)` | `O(n)` |
| HBM 읽기/쓰기 | 많음(점수표 왕복) | 적음(블록만) |
| 실측 속도 | 기준 | 수 배 빠름 |

## 임팩트: 무엇을 바꿨나

- **산업 표준이 되었습니다.** PyTorch의 기본 어텐션 경로(scaled dot-product attention)와 Hugging Face의 `flash_attention_2` 등으로 널리 채택되어, 사실상 모든 LLM 학습·추론 파이프라인이 이 커널 위에서 돕니다.
- 긴 문맥 학습과 추론을 현실적인 비용으로 만든 핵심 엔진입니다. 같은 GPU로 더 긴 문맥을, 더 빠르게 다룰 수 있게 되었습니다.
- "더 빠르게 하려면 연산을 줄여야 한다"는 통념을 깨고, 하드웨어 메모리 계층을 의식한 구현이 같은 수식을 몇 배 빠르게 만든다는 것을 보였습니다. 이 철학은 Mamba의 하드웨어 인식 스캔[7]과도 통하고, 다음 글 FlashAttention-3로 이어집니다.

## 한계와 주의점

- 어텐션의 **연산량 자체(`O(n²)`)는 그대로**입니다. 줄인 것은 메모리 왕복이라, 무한히 긴 문맥이 공짜가 되지는 않습니다. 충분히 긴 문맥에서는 결국 연산량이 병목이 됩니다.
- 하드웨어에 특화된 커널이라, GPU 세대가 바뀌면 새 칩의 기능에 맞춰 다시 최적화해야 합니다. 바로 이 점 때문에 FlashAttention-3가 등장합니다.

## 더 읽을거리와 연결

- 이 책 내부: 다음 글 **FlashAttention-3**(최신 하드웨어), 그다음 **Ring Attention**(분산으로 더 길게), **T03**(PagedAttention 등 서빙 단계 메모리 관리).
- 심화편 S3(FlashAttention 직관)·S7(서빙)와 함께 보면 좋습니다.

## 한 줄 요약

> FlashAttention은 `n×n` 점수표를 HBM에 저장하지 않고 타일 단위로 처리(타일링 + 온라인 softmax)해, 같은 결과를 몇 배 빠르게 메모리 선형으로 낸다. 줄인 것은 연산이 아니라 메모리 왕복이며, 이 정확 어텐션 커널은 산업 표준이 되었다.

## 참고문헌

[1] T. Dao et al., "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness," NeurIPS 2022. arXiv:2205.14135. https://arxiv.org/abs/2205.14135
[2] J. Shah et al., "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision," NeurIPS 2024. arXiv:2407.08608.
[5] J. Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding," Neurocomputing 2024. arXiv:2104.09864.
[6] O. Press, N. Smith, M. Lewis, "Train Short, Test Long: ALiBi," ICLR 2022. arXiv:2108.12409.
[7] A. Gu, T. Dao, "Mamba: Linear-Time Sequence Modeling with Selective State Spaces," COLM 2024. arXiv:2312.00752.

> 다음 → [4_최신_FlashAttention3.md](./4_최신_FlashAttention3.md)
