

논문 출처:
[SmolLM2: When Smol Goes Big -- Data-Centric Training of a Small Language Model](https://huggingface.co/papers/2502.02737)



# Abstrct
- Large Language Model을 활용하는 데에 많은 계산 비용이 소요되고, 제한된 환경에서 배포가 어려운 문제가 있음
- 이 문제를 해결하기 위해 소형(17억 개의 매개변수) 언어 모델인 SmmolLM2의 개발 과정을 문서화하였음 
- 강력한 성능을 달성하기 위해 SmolLM2를 약 11조 개의 데이터 토큰으로 multi-stage process를 통해 훈련하였고, 이 과정에서 웹 텍스트, 특화된 수학과 코드, instruction-following data를 혼합사용하였음
- 또한, 이 과정에서 기존 데이터셋의 품질이 낮다고 판단하여 새로운 전문 데이터셋(Fine-Math, Stack-Edu, SmolTalk)을 추가적으로 소개함
- 궁극적으로, 저자들은 SmolLM2가 다른 최근 소형 언어모델(Llama3.2-1B, Qwen2.5-1.5B)과 벼교했을 때 우수한 성능을 보여준다는 것을 입증하고, 이 모델의 개발 과정에서 준비한 데이터셋을 본 논문에서 공개함


# Introduction
- 언어 모델의 크기에 관계 없이, 훈련하는데에 사용되는 데이터 관리가 성능을 좌우하므로 매우 중요하지만, 작은 모델의 경우에는 특히 데이터 품질이 성능에 더 큰 영향을 미친다.
>  While important for an LM of any size, data curation has an especially outsized influence for smaller models, as their limited capacity must be carefully optimized for learning core knowledge and fundamental
capabilities rather than memorizing incidental fact



