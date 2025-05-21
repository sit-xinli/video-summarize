import subprocess
from faster_whisper import WhisperModel 
from transformers import pipeline
from pydub import AudioSegment
import math
import os

# 1. 動画から音声を抽出
def extract_audio(video_path, audio_path):
    command = ["ffmpeg", "-i", video_path, "-q:a", "0", "-map", "a", audio_path]
    subprocess.run(command, check=True)

# 2. 音声を一定時間ごとに分割（例：5分ごと）
def split_audio(file_path, chunk_length_ms=5 * 60 * 1000):  # 5分
    audio = AudioSegment.from_wav(file_path)
    total_length = len(audio)
    num_chunks = math.ceil(total_length / chunk_length_ms)

    os.makedirs("chunks", exist_ok=True)
    chunk_paths = []

    for i in range(num_chunks):
        start = i * chunk_length_ms
        end = min((i + 1) * chunk_length_ms, total_length)
        chunk = audio[start:end]
        chunk_path = f"chunks/chunk_{i+1}.wav"
        chunk.export(chunk_path, format="wav")
        chunk_paths.append(chunk_path)

    return chunk_paths

# 3. Whisperで音声を文字起こし
def transcribe_audio(audio_path, model_size="medium", language="ja"):
    model = WhisperModel(model_size,compute_type="int8") # TBD:use GPU
    segments, _ = model.transcribe(audio_path, language=language)
    text = " ".join([segment.text for segment in segments])
    return text

# 4. Hugging FaceのBARTで要約
def summarize_text(text, model_name="facebook/bart-large-cnn"):
    summarizer = pipeline("summarization", model=model_name)
    summary = summarizer(text, max_length=60, min_length=20, do_sample=False)
    return summary[0]["summary_text"]

# 実行部分
if __name__ == "__main__":
    video_file = "input-video.mp4"
    audio_file = "full-audio.wav"

    print("🔊 音声を抽出中...")
    extract_audio(video_file, audio_file)

    print("🔄 音声を分割中...")
    chunk_paths = split_audio(audio_file)

    for chunk_path in chunk_paths:
        print(f"📝 {chunk_path} を文字起こし中...")
        transcription = transcribe_audio(chunk_path)
        
        print(f"📄 {chunk_path} を要約中...")
        summary = summarize_text(transcription)

        print(f"\n--- {chunk_path} の文字起こし結果 ---\n", transcription)
        print(f"\n--- {chunk_path} の要約結果 ---\n", summary)

