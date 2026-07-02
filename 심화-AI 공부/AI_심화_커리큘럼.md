# AI 심화 — 응용·최신 방법론 커리큘럼

기초편 「AI 원리 학습 노트」의 속편입니다. 기초편이 원리를 이해하는 책이라면, 심화편은 "그 원리가 실제로 어떻게 만들어지고, 2025–2026 최전선은 어디까지 갔는가"를 다룹니다.

| 항목 | 내용 |
|---|---|
| 전제 | 기초편 0~6단계 수료 (핵심 수식 5개·역전파·어텐션·다음토큰예측 이해) |
| 목표 | 응용 방법론과 최신 연구 흐름의 핵심 아이디어를 잡고, 대표 논문으로 다리를 놓는다 |
| 깊이 정책 | 직관 유지 + 논문 다리 놓기 — 기초편 톤 유지, 수식은 의미 중심, 각 주제에 대표 논문 1~3편 |
| 언어 | 한국어 우선, 핵심 논문은 영어 원문 병행 |
| 작성 시점 | 2026년 6월 (최신 기법은 빠르게 변하므로 시점 명시) |

## 심화편의 학습 방식 — 렌즈 하나를 더한다

기초편의 네 렌즈(목적 → 개념 → 방법론 → 수식)에 대표 논문/자료를 더합니다. 대표 논문은 이 기법이 처음 제안된, 또는 표준이 된 논문으로, '원전으로 가는 다리'입니다. 논문은 읽으라고 있는 게 아니라 길을 가리키라고 있습니다. 각 절은 논문의 핵심 아이디어 한 줄과 "왜 중요한가"를 잡아 주고, 더 깊이 갈 사람을 위해 원문을 가리킵니다. 모든 수식을 따라갈 필요는 없습니다. 기초편 원칙 그대로입니다.

## 전체 로드맵 (11 모듈)

| # | 모듈 | 한 줄 질문 | 기초편 연결 |
|:---:|---|---|:---:|
| S0 | 심화 오리엔테이션 | 최신 흐름을 어떻게 따라잡고 논문을 어떻게 읽는가? | 0단계 |
| S1 | 심화 이론·최적화 | Adam은 왜 그렇게 작동하고, 일반화는 왜 되는가? | 1·2단계 |
| S2 | 생성모델 계보 | 이미지·영상은 어떻게 생성되는가? (Diffusion) | 3단계 |
| S3 | 트랜스포머 심화 | 어텐션을 어떻게 더 빠르고 길게 만드는가? | 4단계 |
| S4 | LLM 사전학습 심화 | 수천 GPU로 모델 하나를 어떻게 학습시키는가? | 5단계 |
| S5 | 정렬·후학습 심화 | RLHF·DPO 파이프라인은 실제로 어떻게 도는가? | 6단계 |
| S6 | 추론·테스트타임 | o1·R1은 어떻게 '더 오래 생각'하는가? | 6단계 |
| S7 | 효율·서빙 | 거대 모델을 어떻게 싸게 굴리는가? (MoE·양자화) | 6단계 |
| S8 | 검색·지식 (RAG) | 환각을 줄이는 검색 시스템은 어떻게 설계되는가? | 6단계 |
| S9 | 에이전트·시스템 | 모델이 도구를 쓰고 계획하게 하려면? | 6단계 |
| S10 | 멀티모달·프런티어 | 텍스트 너머 + 2026의 열린 문제 | 6단계 |

## 모듈별 상세

### S0 · 심화 오리엔테이션
빠르게 바뀌는 분야에서 길 잃지 않기. 신호와 노이즈 구분. 논문 구조(초록·기여·실험) 빠르게 읽기, arXiv/벤치마크/리더보드 읽는 법, 재현성·과대광고 구분. 대표 자료는 Sebastian Raschka의 'Ahead of AI' 큐레이션, Papers with Code, 주요 연구소 블로그.

### S1 · 심화 이론·최적화
경사하강을 넘어, 현대 옵티마이저·정규화가 왜 작동하는지. 모멘텀·RMSProp·Adam 분해, 2차 정보(헤시안)·자연경사 직관, 학습률 스케줄·워밍업, 일반화(평탄한 최소값), 배치/레이어 정규화. 대표 논문은 Adam(Kingma & Ba, 2014), Batch Norm(Ioffe & Szegedy, 2015).

### S2 · 생성모델 계보
"분류"를 넘어 "생성". 데이터 분포 자체를 배우는 모델 계보. 오토인코더 → VAE(잠재변수·ELBO) → GAN(생성자 대 판별자) → Diffusion(노이즈 제거) → Flow Matching(직선 경로). 대표 논문은 VAE(Kingma & Welling, 2013), GAN(Goodfellow et al., 2014), DDPM(Ho et al., 2020), Score-based SDE(Song et al., 2021), Flow Matching(Lipman et al., 2022).

### S3 · 트랜스포머 심화
기초편 어텐션의 실전 한계(O(n²)·문맥 길이)를 뚫기. FlashAttention(IO 인식), 희소·선형 어텐션, RoPE·ALiBi, KV 캐시, 긴 문맥. 대표 논문은 FlashAttention(Dao et al., 2022), RoFormer/RoPE(Su et al., 2021), ALiBi(Press et al., 2021).

### S4 · LLM 사전학습 심화
"다음 토큰 예측"을 수천 GPU·수조 토큰 규모로 실현하는 공학. 데이터 큐레이션·중복 제거, 토크나이저 심화, 분산학습(데이터·텐서·파이프라인 병렬, ZeRO), Chinchilla 최적, 혼합정밀. 대표 논문은 Scaling Laws(Kaplan et al., 2020), Chinchilla(Hoffmann et al., 2022), Megatron-LM/ZeRO(Rajbhandari et al., 2019).

### S5 · 정렬·후학습 심화
사전학습 모델을 쓸모 있고 안전하게 만드는 후학습 전 과정. SFT → 보상 모델링 → RLHF(PPO), DPO, GRPO·KTO·IPO, RLAIF·Constitutional AI, 합성 데이터. 대표 논문은 InstructGPT/RLHF(Ouyang et al., 2022), Constitutional AI(Bai et al., 2022), DPO(Rafailov et al., 2023), GRPO(Shao et al., 2024).

### S6 · 추론·테스트타임 컴퓨트
추론 시점에 더 많이 계산해 어려운 문제를 푸는 원리. CoT·self-consistency, 테스트타임 스케일링, 과정 보상 모델(PRM)·검증자, self-taught reasoning, o-시리즈·R1의 학습 방식. 대표 논문은 CoT(Wei et al., 2022), Self-Consistency(Wang et al., 2022), STaR(Zelikman et al., 2022), Let's Verify Step by Step/PRM(Lightman et al., 2023), DeepSeek-R1(2025).

### S7 · 효율·서빙
거대 모델을 현실적 비용·속도로 학습·배포. MoE(게이팅·희소 활성화·로드 밸런싱), 양자화(GPTQ·AWQ·QLoRA), speculative decoding, KV 캐시·PagedAttention(vLLM), 증류. 대표 논문은 Switch Transformer(Fedus et al., 2021), Mixtral(2024), GPTQ(Frantar et al., 2022), QLoRA(Dettmers et al., 2023), Speculative Decoding(Leviathan et al., 2022), PagedAttention/vLLM(Kwon et al., 2023).

### S8 · 검색·지식 (RAG 심화)
모델의 지식 한계·환각을 외부 검색으로 보완하는 시스템 설계. 임베딩·벡터DB, 하이브리드 검색(BM25+벡터), 리랭킹, 청킹, GraphRAG, 평가(faithfulness). 대표 논문은 RAG(Lewis et al., 2020), DPR(Karpukhin et al., 2020), ColBERT(Khattab & Zaharia, 2020), GraphRAG(Microsoft, 2024).

### S9 · 에이전트·시스템
모델이 추론하고 도구를 쓰고 행동하게 만드는 구조. ReAct, tool/function calling, Reflexion, 계획·메모리, 멀티에이전트, LLM-as-judge·벤치 설계. 대표 논문은 ReAct(Yao et al., 2022), Toolformer(Schick et al., 2023), Reflexion(Shinn et al., 2023), MT-Bench/LLM-as-judge(Zheng et al., 2023).

### S10 · 멀티모달·프런티어
텍스트 너머(이미지·음성·영상)로의 확장과 2026의 열린 문제. CLIP, VLM(LLaVA·Flamingo), 통합 멀티모달, world model, 장기 과제(환각·추론 신뢰성·정렬·평가). 대표 논문은 CLIP(Radford et al., 2021), Flamingo(Alayrac et al., 2022), LLaVA(Liu et al., 2023).

## 집필 원칙

깊이 정책을 준수합니다. 직관 먼저, 논문은 다리입니다. 유도와 증명은 넣지 않고 필요하면 원문으로 보냅니다. 논문은 실재하는 대표작만 인용하며, 연도·저자가 불확실하면 그렇게 표기합니다. 기초편과 동일한 폴더·파일·템플릿 구조(README + 번호 본문 + 정리노트)를 따르고, 최신 기법은 각 절에 작성 시점(2026-06)을 명시합니다.
