import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_default_device("cuda")

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype="auto", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)

inputs = tokenizer("- 내 피치 평균: 176Hz, 표준편차: 34Hz, jitter 2.1% - 원곡 피치 평균: 193Hz, 표준편차: 12Hz, jitter 0.8% - 차이: 3~4초 구간 +0.3 semitone 이런 데이터에서 음정 불안정, 주요 차이점, 코칭 포인트를 피드백 해줘.", return_tensors="pt", return_attention_mask=False)

outputs = model.generate(**inputs, max_length=300)
text = tokenizer.batch_decode(outputs)[0]
print(text)