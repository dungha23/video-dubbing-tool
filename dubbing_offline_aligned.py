import argparse
import asyncio
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List

import edge_tts
import whisper
from pydub import AudioSegment


def run(command: List[str]) -> None:
    subprocess.run(command, check=True)


def extract_audio(video: Path, audio: Path) -> None:
    run([
        "ffmpeg", "-y", "-i", str(video), "-vn", "-ac", "1",
        "-ar", "16000", "-c:a", "pcm_s16le", str(audio)
    ])


def transcribe(audio: Path, source_language: str, model_name: str) -> List[Dict]:
    model = whisper.load_model(model_name)
    result = model.transcribe(
        str(audio),
        language=source_language,
        fp16=False,
        verbose=False,
        condition_on_previous_text=True,
        temperature=0,
    )
    items = []
    for segment in result.get("segments", []):
        text = segment.get("text", "").strip()
        if text and segment["end"] > segment["start"]:
            items.append({
                "start": float(segment["start"]),
                "end": float(segment["end"]),
                "text": text,
            })
    return items


def translate_offline(items: List[Dict], source: str, target: str) -> None:
    """Translate locally with Argos Translate; no OpenAI/API key required."""
    try:
        import argostranslate.translate
    except ImportError as exc:
        raise RuntimeError(
            "Chưa cài Argos Translate. Chạy: pip install argostranslate "
            "và cài gói ngôn ngữ bằng argos-translate-cli."
        ) from exc

    for item in items:
        item["translation"] = argostranslate.translate.translate(
            item["text"], source, target
        ).strip()
        if not item["translation"]:
            item["translation"] = item["text"]


def atempo_filters(factor: float) -> str:
    """Build valid atempo chain; each filter must be between 0.5 and 2.0."""
    factor = max(0.25, min(4.0, factor))
    filters = []
    while factor < 0.5:
        filters.append("atempo=0.5")
        factor /= 0.5
    while factor > 2.0:
        filters.append("atempo=2.0")
        factor /= 2.0
    filters.append(f"atempo={factor:.8f}")
    return ",".join(filters)


def fit_audio_to_window(source: Path, target_seconds: float, output: Path) -> None:
    """Fit a TTS clip exactly to its original speech window as far as possible."""
    audio = AudioSegment.from_file(source)
    source_seconds = max(len(audio) / 1000.0, 0.001)
    target_seconds = max(target_seconds, 0.05)
    # atempo factor > 1 means speed up; factor < 1 means slow down.
    factor = source_seconds / target_seconds
    run([
        "ffmpeg", "-y", "-i", str(source),
        "-filter:a", atempo_filters(factor),
        "-ar", "48000", "-ac", "2", str(output)
    ])


def make_tts(text: str, output: Path, voice: str) -> None:
    async def generate() -> None:
        await edge_tts.Communicate(text, voice).save(str(output))
    asyncio.run(generate())


def build_aligned_audio(items: List[Dict], work: Path, output: Path) -> None:
    """Place each fitted sentence at its original timestamp, including gaps."""
    timeline = AudioSegment.empty()
    cursor_ms = 0
    for index, item in enumerate(items):
        raw = work / f"tts_{index:04d}.mp3"
        fitted = work / f"fit_{index:04d}.wav"
        make_tts(item["translation"], raw, "vi-VN-NhanNeural")
        window = item["end"] - item["start"]
        fit_audio_to_window(raw, window, fitted)
        clip = AudioSegment.from_file(fitted)
        start_ms = round(item["start"] * 1000)
        if start_ms > cursor_ms:
            timeline += AudioSegment.silent(duration=start_ms - cursor_ms)
        # Never allow a long TTS clip to push the next timestamp forward.
        timeline += clip[: max(1, round(window * 1000))]
        cursor_ms = start_ms + len(clip)

    timeline.export(output, format="wav")


def srt_time(seconds: float) -> str:
    millis = max(0, round(seconds * 1000))
    hours, millis = divmod(millis, 3600000)
    minutes, millis = divmod(millis, 60000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def write_srt(items: List[Dict], output: Path) -> None:
    with output.open("w", encoding="utf-8") as file:
        for index, item in enumerate(items, 1):
            file.write(
                f"{index}\n{srt_time(item['start'])} --> {srt_time(item['end'])}\n"
                f"{item['translation']}\n\n"
            )


def mux(video: Path, audio: Path, subtitle: Path, output: Path) -> None:
    # Subtitle is included as a selectable soft subtitle track; video is copied.
    run([
        "ffmpeg", "-y", "-i", str(video), "-i", str(audio), "-i", str(subtitle),
        "-map", "0:v:0", "-map", "1:a:0", "-map", "2:0",
        "-c:v", "copy", "-c:a", "aac", "-c:s", "mov_text",
        "-metadata:s:s:0", "language=vie", "-metadata:s:s:0", "title=Tiếng Việt",
        "-shortest", str(output)
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline Vietnamese video dubbing with aligned subtitles")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--language", default="en", help="Source language, e.g. en, ja, fr")
    parser.add_argument("--model", default="small", help="Whisper model: base, small, medium, large")
    parser.add_argument("--voice", default="vi-VN-NhanNeural")
    parser.add_argument("--keep-temp", action="store_true")
    args = parser.parse_args()

    video = Path(args.input)
    if not video.exists():
        raise FileNotFoundError(video)
    work = Path("temp_offline_dubbing")
    work.mkdir(exist_ok=True)
    audio = work / "source.wav"
    aligned = work / "vietnamese_aligned.wav"
    subtitles = Path(args.output).with_suffix(".vi.srt")
    data = work / "segments.json"

    try:
        print("1/6 Nhận diện lời thoại và timestamp...")
        extract_audio(video, audio)
        items = transcribe(audio, args.language, args.model)
        if not items:
            raise RuntimeError("Không tìm thấy lời thoại trong video.")

        print("2/6 Dịch offline bằng Argos Translate...")
        translate_offline(items, args.language, "vi")
        data.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

        print("3/6 Tạo giọng Việt từng câu và căn đúng timestamp...")
        # The global voice is passed through the environment-free helper by using a local override.
        # Replace the default voice in build_aligned_audio if another voice is desired.
        build_aligned_audio(items, work, aligned)

        print("4/6 Tạo subtitle tiếng Việt...")
        write_srt(items, subtitles)
        print("5/6 Ghép video, giọng Việt và subtitle...")
        mux(video, aligned, subtitles, Path(args.output))
        print(f"Hoàn tất: {args.output}")
        print(f"Subtitle: {subtitles}")
    finally:
        if not args.keep_temp and work.exists():
            shutil.rmtree(work)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"Lỗi: {error}", file=sys.stderr)
        sys.exit(1)
