import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://router.huggingface.co/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {os.environ['HF_TOKEN']}",
}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

response = query({
    "messages": [
        {
            "role": "user",
            "content": "나의 pitch 평균: 176Hz, 표준편차: 34Hz, jitter: 2.1%, unvoiced 비율: 2.5% 원곡 평균: 193Hz, 표준편차: 12Hz, jitter: 0.8%, unvoiced: 1.3% 차이점, 개선점, 예상 voice type/문제점, 연습 Point를 전문가로서 알려줘"
        }
    ],
    "model": "deepseek-ai/DeepSeek-V3:novita"
})

print(response["choices"][0]["message"])