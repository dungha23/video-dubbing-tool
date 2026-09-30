import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import edge_tts
import whisper
from pydub import AudioSegment


def run(cmd):
    subprocess.run(cmd, check=True)


def extract_audio(video: Path, audio: Path):
    run([
        "ffmpeg", "-y", "-i", str(video),
        "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", str(audio)
    ])


def transcribe(audio: Path, language: str, model_name: str):
    model = whisper.load_model(model_name)
    result = model.transcribe(
        str(audio),
        language=language,
        fp16=False,
        verbose=False,
        temperature=0,
    )
    segments = []
    for seg in result.get("segments", []):
        text = (seg.get("text") or "").strip()
        if text and seg["end"] > seg["start"]:
            segments.append({
                "start": float(seg["start"]),
                "end": float(seg["end"]),
                "text": text,
            })
    return segments


def translate_with_argos(items, source_lang, target_lang):
    try:
        from argostranslate import translate
    except Exception as exc:
        raise RuntimeError(
            "Thiếu Argos Translate. Cài: pip install argostranslate. "
            "Sau đó cài gói ngôn ngữ tương ứng từ argos-translate-cli."
        ) from exc

    for item in items:
        translated = translate.translate(item["text"], source_lang, target_lang)
        item["translation"] = translated.strip() if translated else item["text"]


def tts_make(text: str, out_path: Path, voice: str):
    async def _go():
        await edge_tts.Communicate(text, voice).save(str(out_path))
    import asyncio
    asyncio.run(_go())


def atempo_chain(factor: float):
    # ffmpeg atempo limits are 0.5-2.0 per filter; chain if needed
    pieces = []
    while factor > 2.0:
        pieces.append("atempo=2.0")
        factor /= 2.0
    while factor < 0.5:
        pieces.append("atempo=0.5")
        factor /= 0.5
    pieces.append(f"atempo={factor:.6f}")
    return ",".join(pieces)


def fit_length(input_mp3: Path, target_seconds: float, out_wav: Path):
    audio = AudioSegment.from_file(input_mp3)
    original_seconds = len(audio) / 1000.0
    target_seconds = max(target_seconds, 0.08)
    if original_seconds <= 0.0001:
        raise ValueError("Audio huỷ bỏ")
    factor = original_seconds / target_seconds
    factor = max(0.75, min(1.5, factor))
    run([
        "ffmpeg", "-y", "-i", str(input_mp3),
        "-filter:a", atempo_chain(factor),
        "-ar", "48000", "-ac", "2", str(out_wav)
    ])


def build_timeline(items, work_dir: Path, voice: str, subtitle_path: Path):
    total = AudioSegment.empty()
    cursor_ms = 0
    for idx, item in enumerate(items):
        raw = work_dir / f"tts_{idx:03d}.mp3"
        fitted = work_dir / f"fit_{idx:03d}.wav"
        tts_make(item["translation"], raw, voice)
        fit_length(raw, item["end"] - item["start"], fitted)
        clip = AudioSegment.from_file(fitted)
        start_ms = int(item["start"] * 1000)
        if start_ms > cursor_ms:
            total += AudioSegment.silent(duration=start_ms - cursor_ms)
        total += clip[: max(1, int((item["end"] - item["start"]) * 1000))]
        cursor_ms = start_ms + int((item["end"] - item["start"]) * 1000)

    output_audio = work_dir / "final_audio.wav"
    total.export(output_audio, format="wav")
    return output_audio


def time_fmt(sec: float) -> str:
    millis = max(0, round(sec * 1000))
    h, rem = divmod(millis, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(items, srt_path: Path):
    with srt_path.open("w", encoding="utf-8") as f:
        for idx, item in enumerate(items, 1):
            f.write(f"{idx}\n")
            f.write(f"{time_fmt(item['start'])} --> {time_fmt(item['end'])}\n")
            f.write(f"{item['translation']}\n\n")


def mux_video(video: Path, audio: Path, srt: Path, output: Path):
    run([
        "ffmpeg", "-y", "-i", str(video), "-i", str(audio), "-i", str(srt),
        "-map", "0:v:0", "-map", "1:a:0", "-map", "2:0",
        "-c:v", "copy", "-c:a", "aac", "-c:s", "mov_text",
        "-metadata:s:s:0", "language=vie",
        "-metadata:s:s:0", "title=Vietnamese",
        "-shortest", str(output)
    ])


def main():
    parser = argparse.ArgumentParser(description="Offline Vietnamese dubbing with subtitles")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--language", default="en")
    parser.add_argument("--model", default="base")
    parser.add_argument("--voice", default="vi-VN-NhanNeural")
    parser.add_argument("--keep-temp", action="store_true")
    args = parser.parse_args()

    video = Path(args.input)
    if not video.exists():
        raise FileNotFoundError(video)

    work = Path("temp_video_dub")
    work.mkdir(exist_ok=True)
    audio = work / "input.wav"
    srt_path = Path(args.output).with_suffix(".vi.srt")

    try:
        print("1/4 Tách âm thanh...")
        extract_audio(video, audio)

        print("2/4 Nhận diện thoại theo đoạn...")
        items = transcribe(audio, args.language, args.model)
        if not items:
            raise RuntimeError("Không phát hiện lời thoại trong video.")

        print("3/4 Dịch sang tiếng Việt (offline)...")
        translate_with_argos(items, args.language, "vi")

        print("4/4 Tạo giọng đọc và ghép subtitle + video...")
        build_timeline(items, work, args.voice, srt_path)
        write_srt(items, srt_path)
        mux_video(video, work / "final_audio.wav", srt_path, Path(args.output))

        print(f"Video đã xuất: {args.output}")
        print(f"Subtitle: {srt_path}")
    finally:
        if not args.keep_temp:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        sys.exit(1)
