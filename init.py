from flask import Flask, request, send_file, jsonify
import tempfile
import librosa
import whisper
from spleeter.separator import Separator
import os
import zipfile
from flask_cors import CORS, cross_origin
import tensorflow as tf
import numpy as np
from dotenv import load_dotenv

import requests

load_dotenv()

API_URL = "https://router.huggingface.co/v1/chat/completions"
HF_TOKEN = os.environ.get('HF_TOKEN')
headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
}

def query_llm(user_feature, singer_feature):
    prompt = (
        f"나의 pitch 평균: {user_feature.get('pitch_mean', 0):.1f}Hz, "
        f"표준편차: {user_feature.get('pitch_std', 0):.1f}Hz, "
        f"jitter: {user_feature.get('jitter_percent', 0):.2f}%, "
        f"voiced 비율: {user_feature.get('voiced_ratio', 0):.1f}% "
        f"원곡 평균: {singer_feature.get('pitch_mean', 0):.1f}Hz, "
        f"표준편차: {singer_feature.get('pitch_std', 0):.1f}Hz, "
        f"jitter: {singer_feature.get('jitter_percent', 0):.2f}%, "
        f"voiced: {singer_feature.get('voiced_ratio', 0):.1f}% "
        "차이점, 개선점, 예상 voice type/문제점, 연습 Point를 전문가로서 알려줘"
    )
    payload = {
        "messages": [
            {
                "role": "user",
                "content": prompt
            }
        ],
        "model": "deepseek-ai/DeepSeek-V3:novita"
    }
    response = requests.post(API_URL, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()

app=Flask(__name__)
CORS(app, resources={r"/api/*": {"origins" : "*"}})
app.debug=True

print(tf.config.list_physical_devices('GPU'))

def analyze_audio(audio_path):
    y, sr = librosa.load(audio_path, sr=22050)
    
    #에너지(threshold)로 노이즈 게이트 (선택적 전처리)
    y[np.abs(y) < 0.01] = 0

    print("[분석 중] 음정 추출")
    f0, voiced_flag, _ = librosa.pyin(
        y,
        fmin=librosa.note_to_hz('C2'),
        fmax=librosa.note_to_hz('C7'),
        frame_length=2048,
        hop_length=256
    )

    # RMS 기반 추가 필터링
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=256)[0]
    rms_thr = 0.02  # 실험적 임계값
    pitch_hz = []
    for i, (f, v) in enumerate(zip(f0, voiced_flag)):
        if v and (f is not None) and (not np.isnan(f)) and (rms[i] > rms_thr):
            pitch_hz.append(float(f))
        else:
            pitch_hz.append(None)


    print("[분석 중] 박자 추출")
    onset_frames = librosa.onset.onset_detect(
        y=y, sr=sr, backtrack=True,
    )
    onset_times = librosa.frames_to_time(onset_frames, sr=sr).tolist()

    print("[분석 중] 가사 추출 (Whisper)")
    model = whisper.load_model("base")   # "tiny", "base" 등 선택 가능
    result = model.transcribe(audio_path, language='ko')
    segments = result["segments"]
    lyrics_segments = []
    for seg in segments:
        start = float(seg['start'])
        end = float(seg['end'])
        duration = end - start
        lyrics_segments.append({
            "start": round(start, 2),
            "duration": round(duration, 2),
            "text": seg['text']
        })

    
    return {
        "sample_rate": int(sr),
        "pitch_hz": pitch_hz,
        "onset_times": onset_times,
        "lyrics": lyrics_segments
    }

separator = Separator('spleeter:2stems')


@app.route('/separate', methods=['POST'])
def separate_audio():
    # 1. 파일 업로드 받기
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file'}), 400
    file = request.files['audio']

    # 2. 임시 경로에 저장
    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, file.filename)
        file.save(input_path)

        # 3. Spleeter로 source separation (GPU 사용)
        out_dir = os.path.join(tmpdir, 'output')
        os.makedirs(out_dir, exist_ok=True)
        separator.separate_to_file(
            input_path,
            out_dir,
            codec='wav', # mp3도 가능
            synchronous=True
        )

        # 4. 분리된 파일 위치: out_dir/{원본파일이름}/vocals.wav, accompaniment.wav
        session_dir = os.path.join(
            out_dir,
            os.path.splitext(os.path.basename(input_path))[0]
        )

        stems = ['vocals.wav', 'accompaniment.wav']
        files = {}
        for stem in stems:
            stem_path = os.path.join(session_dir, stem)
            if os.path.exists(stem_path):
                files[stem.replace('.wav', '')] = stem_path

        # 5. 요청에 따라 각 세션(wav 파일)을 보내줌 (zip 포함)
        # 5-a. zip으로 묶어서 반환하는 예시:
        from zipfile import ZipFile
        zip_path = os.path.join(tmpdir, 'separated.zip')
        with ZipFile(zip_path, 'w') as zipf:
            for name, fpath in files.items():
                zipf.write(fpath, arcname=os.path.basename(fpath))

        return send_file(zip_path, as_attachment=True, download_name='separated.zip')

@app.route('/analyze', methods=['POST'])
def analyze():
    # 1. 파일 업로드 체크
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file uploaded'}), 400

    file = request.files['audio']

    # 2. 임시 폴더에 저장
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[-1]) as tmp:
        file.save(tmp)
        tmp.flush()
        tmp_path = tmp.name

    try:
        # 3. 분석 실행
        result = analyze_audio(tmp_path)
        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 500

    finally:
        # 4. 임시 파일 정리
        os.unlink(tmp_path)

@app.route('/llm_pitch_feedback', methods=['POST'])
def llm_pitch_feedback():
    data = request.get_json()
    user_feat = data.get('user_feature', {})
    singer_feat = data.get('singer_feature', {})

    # LLM API 호출
    try:
        llm_response = query_llm(user_feat, singer_feat)
        feedback = llm_response['choices'][0]['message']['content']
    except Exception as e:
        return jsonify({"error": str(e)}), 500

    return jsonify({
        "feedback": feedback
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80, debug=True)