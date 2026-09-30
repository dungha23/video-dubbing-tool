import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

import edge_tts
import whisper
from pydub import AudioSegment


def run(cmd, check=True, capture_output=False):
    """Run command với error handling tốt hơn."""
    try:
        return subprocess.run(cmd, check=check, capture_output=capture_output, text=True)
    except FileNotFoundError as e:
        raise RuntimeError(f"Lệnh không tìm thấy: {cmd[0]}. Kiểm tra FFmpeg đã cài?") from e
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Lỗi chạy lệnh: {e.stderr or e.stdout or str(e)}") from e


def check_ffmpeg():
    """Kiểm tra FFmpeg đã cài chưa."""
    try:
        run(["ffmpeg", "-version"], capture_output=True)
    except RuntimeError:
        raise RuntimeError(
            "FFmpeg không được cài đặt hoặc không trong PATH.\n"
            "Windows: Tải từ https://ffmpeg.org/download.html\n"
            "Hoặc dùng: choco install ffmpeg (nếu dùng Chocolatey)"
        )


def extract_audio(video: Path, audio: Path):
    """Tách âm thanh từ video."""
    print("  → Tách âm thanh...")
    run([
        "ffmpeg", "-y", "-i", str(video),
        "-vn", "-ac", "1", "-ar", "16000",
        "-c:a", "pcm_s16le", str(audio)
    ])


def transcribe(audio: Path, language: str, model_name: str):
    """Nhận diện lời thoại theo đoạn."""
    print("  → Tải model Whisper...")
    model = whisper.load_model(model_name)
    
    print("  → Nhận diện lời thoại...")
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
                "duration": float(seg["end"]) - float(seg["start"]),
                "text": text,
            })
    
    return segments


def merge_short_segments(segments, min_duration=0.5):
    """Ghép các đoạn quá ngắn với đoạn liền kề để tránh âm thanh rời rạc."""
    print("  → Ghép các đoạn quá ngắn...")
    merged = []
    i = 0
    while i < len(segments):
        current = segments[i]
        # Nếu đoạn quá ngắn, ghép với đoạn tiếp theo
        if current["duration"] < min_duration and i + 1 < len(segments):
            combined = {
                "start": current["start"],
                "end": segments[i + 1]["end"],
                "duration": segments[i + 1]["end"] - current["start"],
                "text": current["text"] + " " + segments[i + 1]["text"]
            }
            merged.append(combined)
            i += 2
        else:
            merged.append(current)
            i += 1
    
    return merged


def translate_with_argos(items, source_lang, target_lang):
    """Dịch sang tiếng Việt offline."""
    print("  → Dịch sang tiếng Việt...")
    try:
        from argostranslate import translate
    except ImportError as exc:
        raise RuntimeError(
            "Thiếu Argos Translate. Cài: pip install argostranslate"
        ) from exc

    for item in items:
        try:
            translated = translate.translate(item["text"], source_lang, target_lang)
            item["translation"] = translated.strip() if translated else item["text"]
        except Exception as e:
            print(f"    ⚠ Dịch lỗi: {item['text'][:30]}..., dùng text gốc")
            item["translation"] = item["text"]


def tts_make(text: str, out_path: Path, voice: str):
    """Tạo giọng nói tiếng Việt."""
    async def _go():
        await edge_tts.Communicate(text, voice).save(str(out_path))
    import asyncio
    try:
        asyncio.run(_go())
    except Exception as e:
        raise RuntimeError(f"Lỗi tạo TTS cho '{text[:30]}...': {e}") from e


def atempo_chain(factor: float):
    """Xây dựng chuỗi atempo filter (FFmpeg giới hạn 0.5-2.0)."""
    pieces = []
    # Nếu factor > 2, tách thành nhiều filter
    while factor > 2.0:
        pieces.append("atempo=2.0")
        factor /= 2.0
    # Nếu factor < 0.5, tách thành nhiều filter
    while factor < 0.5:
        pieces.append("atempo=0.5")
        factor /= 0.5
    pieces.append(f"atempo={factor:.6f}")
    return ",".join(pieces)


def fit_length(input_mp3: Path, target_seconds: float, out_wav: Path):
    """Chỉnh tốc độ audio để vừa với khoảng thời gian target."""
    audio = AudioSegment.from_file(input_mp3)
    original_seconds = len(audio) / 1000.0
    target_seconds = max(target_seconds, 0.1)
    
    if original_seconds <= 0.01:
        raise ValueError("Audio quá ngắn hoặc lỗi")
    
    # Tính tốc độ phát (tối đa 1.8x để giữ chất lượng)
    factor = original_seconds / target_seconds
    factor = max(0.6, min(1.8, factor))
    
    run([
        "ffmpeg", "-y", "-i", str(input_mp3),
        "-filter:a", atempo_chain(factor),
        "-ar", "48000", "-ac", "2", str(out_wav)
    ])


def build_timeline(items, work_dir: Path, voice: str):
    """Xây dựng timeline audio với các đoạn thoại tiếng Việt."""
    print("  → Tạo giọng đọc từng đoạn...")
    total = AudioSegment.empty()
    cursor_ms = 0
    
    for idx, item in enumerate(items):
        print(f"    [{idx+1}/{len(items)}] {item['translation'][:40]}...")
        
        raw = work_dir / f"tts_{idx:03d}.mp3"
        fitted = work_dir / f"fit_{idx:03d}.wav"
        
        try:
            tts_make(item["translation"], raw, voice)
            fit_length(raw, item["duration"], fitted)
            clip = AudioSegment.from_file(fitted)
            
            start_ms = int(item["start"] * 1000)
            
            # Thêm khoảng lặng nếu có gap
            if start_ms > cursor_ms:
                total += AudioSegment.silent(duration=start_ms - cursor_ms)
            
            # Thêm clip audio, cắt nếu quá dài
            max_duration_ms = int(item["duration"] * 1000)
            total += clip[:max_duration_ms]
            cursor_ms = start_ms + max_duration_ms
            
        except Exception as e:
            print(f"    ✗ Lỗi đoạn {idx}: {e}")
            raise
    
    output_audio = work_dir / "final_audio.wav"
    total.export(output_audio, format="wav")
    return output_audio


def time_fmt(sec: float) -> str:
    """Convert giây thành định dạng SRT."""
    millis = max(0, round(sec * 1000))
    h, rem = divmod(millis, 3600000)
    m, rem = divmod(rem, 60000)
    s, ms = divmod(rem, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def write_srt(items, srt_path: Path):
    """Tạo file subtitle SRT."""
    print("  → Tạo file subtitle...")
    with srt_path.open("w", encoding="utf-8") as f:
        for idx, item in enumerate(items, 1):
            f.write(f"{idx}\n")
            f.write(f"{time_fmt(item['start'])} --> {time_fmt(item['end'])}\n")
            f.write(f"{item['translation']}\n\n")


def mux_video(video: Path, audio: Path, srt: Path, output: Path):
    """Ghép video + audio tiếng Việt + subtitle."""
    print("  → Ghép video + audio + subtitle...")
    run([
        "ffmpeg", "-y", "-i", str(video), "-i", str(audio), "-i", str(srt),
        "-map", "0:v:0", "-map", "1:a:0", "-map", "2:0",
        "-c:v", "copy", "-c:a", "aac", "-c:s", "mov_text",
        "-metadata:s:s:0", "language=vie",
        "-metadata:s:s:0", "title=Vietnamese",
        "-shortest", str(output)
    ])


def main():
    parser = argparse.ArgumentParser(
        description="Dịch video sang tiếng Việt với tối ưu độ khớp"
    )
    parser.add_argument("--input", required=True, help="File video đầu vào")
    parser.add_argument("--output", required=True, help="File video đầu ra")
    parser.add_argument("--language", default="en", help="Ngôn ngữ gốc (en, ja, fr, etc.)")
    parser.add_argument("--model", default="base", help="Model Whisper (base, small, medium, large)")
    parser.add_argument("--voice", default="vi-VN-NhanNeural", 
                        help="Giọng (vi-VN-NhanNeural hoặc vi-VN-HoaiMyNeural)")
    parser.add_argument("--keep-temp", action="store_true", help="Giữ lại file tạm")
    args = parser.parse_args()

    video = Path(args.input)
    if not video.exists():
        raise FileNotFoundError(f"Video không tìm thấy: {video}")

    print("\n" + "="*60)
    print("VIDEO DUBBING TOOL - Phiên bản Tối Ưu")
    print("="*60 + "\n")

    work = Path("temp_video_dub")
    work.mkdir(exist_ok=True)
    audio = work / "input.wav"
    srt_path = Path(args.output).with_suffix(".vi.srt")

    try:
        print("[1/5] Chuẩn bị...")
        check_ffmpeg()
        extract_audio(video, audio)

        print("[2/5] Nhận diện thoại...")
        items = transcribe(audio, args.language, args.model)
        if not items:
            raise RuntimeError("Không phát hiện lời thoại trong video.")
        
        # Tối ưu: ghép các đoạn quá ngắn
        items = merge_short_segments(items, min_duration=0.4)

        print(f"[3/5] Dịch sang tiếng Việt ({len(items)} đoạn)...")
        translate_with_argos(items, args.language, "vi")

        print("[4/5] Tạo giọng nói...")
        build_timeline(items, work, args.voice)
        write_srt(items, srt_path)

        print("[5/5] Ghép video...")
        mux_video(video, work / "final_audio.wav", srt_path, Path(args.output))

        print("\n" + "="*60)
        print("✓ Hoàn tất!")
        print(f"  Video: {args.output}")
        print(f"  Subtitle: {srt_path}")
        print("="*60 + "\n")

    except Exception as exc:
        print(f"\n✗ Lỗi: {exc}\n", file=sys.stderr)
        sys.exit(1)
    finally:
        if not args.keep_temp:
            shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
