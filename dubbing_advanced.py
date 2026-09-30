import argparse
import asyncio
import os
import subprocess
import sys
import json
from pathlib import Path
from typing import List, Dict
import time

import edge_tts
import whisper
from openai import OpenAI
from pydub import AudioSegment


def extract_audio(input_video: str, output_audio: str) -> str:
    """Extract audio from video."""
    print("=" * 60)
    print("[Step 1/8] Extracting audio from video...")
    print("=" * 60)
    subprocess.run([
        "ffmpeg", "-y", "-i", input_video,
        "-vn", "-acodec", "libmp3lame", "-q:a", "5",
        output_audio
    ], check=True, capture_output=True)
    print(f"✓ Audio extracted: {output_audio}\n")
    return output_audio


def transcribe_audio_with_timestamps(audio_path: str, language: str = "en") -> List[Dict]:
    """Transcribe audio and extract sentence-level timestamps."""
    print("=" * 60)
    print("[Step 2/8] Transcribing audio with timestamps...")
    print("=" * 60)
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, language=language, fp16=False)
    
    segments = result.get("segments", [])
    sentences = []
    
    for segment in segments:
        start = segment["start"]
        end = segment["end"]
        text = segment["text"].strip()
        
        if text and len(text) > 1:  # Filter very short segments
            sentences.append({
                "id": len(sentences),
                "start": start,
                "end": end,
                "duration": end - start,
                "original": text,
                "translated": "",
                "tts_audio_path": "",
                "adjusted_audio_path": "",
                "tts_duration": 0,
                "speed_factor": 1.0
            })
    
    print(f"✓ Found {len(sentences)} sentences\n")
    print("Transcribed segments:")
    print("-" * 60)
    for i, sent in enumerate(sentences):
        print(f"[{i+1:2d}] {sent['start']:6.2f}s - {sent['end']:6.2f}s | "
              f"Duration: {sent['duration']:.2f}s | {sent['original'][:50]}")
    print("-" * 60 + "\n")
    
    return sentences


def translate_sentences_batch(sentences: List[Dict], api_key: str) -> List[Dict]:
    """Translate all sentences to Vietnamese using GPT."""
    print("=" * 60)
    print("[Step 3/8] Translating sentences to Vietnamese...")
    print("=" * 60)
    client = OpenAI(api_key=api_key)
    
    # Combine all sentences for context-aware translation
    original_texts = "\n".join([f"[{i}] {s['original']}" for i, s in enumerate(sentences)])
    
    prompt = (
        "Bạn là biên dịch viên chuyên nghiệp. Dịch từng câu dưới đây sang tiếng Việt tự nhiên. "
        "Giữ nguyên ý nghĩa, giọng điệu. Không giải thích thêm.\n"
        "Trả về kết quả theo định dạng JSON:\n"
        '{"translations": ["dịch câu 1", "dịch câu 2", ...]}\n\n'
        f"{original_texts}"
    )
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Bạn là biên dịch viên chuyên nghiệp. Dịch gọn gàng, tự nhiên."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    
    try:
        result_text = response.choices[0].message.content
        # Try to parse JSON
        if "```json" in result_text:
            result_text = result_text.split("```json")[1].split("```")[0]
        elif "```" in result_text:
            result_text = result_text.split("```")[1].split("```")[0]
        
        result_json = json.loads(result_text)
        translations = result_json.get("translations", [])
    except (json.JSONDecodeError, IndexError):
        print("⚠ Could not parse JSON, using original text as fallback\n")
        translations = [s['original'] for s in sentences]
    
    print("Translations:")
    print("-" * 60)
    for i, sent in enumerate(sentences):
        if i < len(translations):
            sent['translated'] = translations[i].strip()
        else:
            sent['translated'] = sent['original']
        print(f"[{i+1:2d}] {sent['translated'][:60]}")
    print("-" * 60 + "\n")
    
    return sentences


async def generate_tts_for_sentence(sentence: Dict, voice: str, output_dir: str) -> Dict:
    """Generate TTS audio for a single sentence."""
    text = sentence['translated']
    audio_file = Path(output_dir) / f"sentence_{sentence['id']:03d}.mp3"
    
    try:
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(str(audio_file))
        
        # Get actual TTS duration
        audio = AudioSegment.from_file(str(audio_file))
        tts_duration = len(audio) / 1000.0  # Convert to seconds
        
        sentence['tts_audio_path'] = str(audio_file)
        sentence['tts_duration'] = tts_duration
        
        return sentence
    except Exception as e:
        print(f"✗ Error generating TTS for sentence {sentence['id']}: {e}")
        sentence['tts_audio_path'] = None
        return sentence


async def generate_tts_for_all_sentences(sentences: List[Dict], output_dir: str, voice: str) -> List[Dict]:
    """Generate TTS audio for all sentences sequentially."""
    print("=" * 60)
    print("[Step 4/8] Generating TTS audio for each sentence...")
    print("=" * 60)
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print("Generating audio files:")
    print("-" * 60)
    for i, sent in enumerate(sentences):
        await generate_tts_for_sentence(sent, voice, str(output_path))
        if sent['tts_audio_path']:
            print(f"[{i+1:2d}] ✓ {sent['translated'][:40]}... "
                  f"(Duration: {sent['tts_duration']:.2f}s)")
        else:
            print(f"[{i+1:2d}] ✗ Failed to generate TTS")
    print("-" * 60 + "\n")
    
    return sentences


def adjust_audio_speed(audio_path: str, target_duration: float) -> tuple:
    """
    Adjust audio speed to fit target duration.
    Returns: (adjusted_audio_path, speed_factor)
    """
    audio = AudioSegment.from_file(audio_path)
    current_duration_ms = len(audio)
    target_duration_ms = target_duration * 1000
    
    if current_duration_ms == 0 or target_duration <= 0:
        return audio_path, 1.0
    
    speed_factor = current_duration_ms / target_duration_ms
    
    # Limit speed changes to reasonable range (0.7x to 1.5x)
    original_speed = speed_factor
    speed_factor = max(0.7, min(1.5, speed_factor))
    
    adjusted_path = audio_path.replace(".mp3", "_adjusted.mp3")
    
    try:
        subprocess.run([
            "ffmpeg", "-y", "-i", audio_path,
            "-filter:a", f"atempo={speed_factor}",
            adjusted_path
        ], check=True, capture_output=True)
        return adjusted_path, speed_factor
    except Exception as e:
        print(f"⚠ Failed to adjust speed: {e}, using original")
        return audio_path, 1.0


def adjust_all_audio_speeds(sentences: List[Dict]) -> List[Dict]:
    """Adjust audio speed for all sentences to match original duration."""
    print("=" * 60)
    print("[Step 5/8] Adjusting audio speed to match original timing...")
    print("=" * 60)
    
    print("Speed adjustment:")
    print("-" * 60)
    for sent in sentences:
        if not sent['tts_audio_path']:
            continue
        
        original_duration = sent['duration']
        tts_duration = sent['tts_duration']
        
        adjusted_path, speed_factor = adjust_audio_speed(
            sent['tts_audio_path'],
            original_duration
        )
        
        sent['adjusted_audio_path'] = adjusted_path
        sent['speed_factor'] = speed_factor
        
        print(f"[{sent['id']+1:2d}] Original: {original_duration:.2f}s | "
              f"TTS: {tts_duration:.2f}s | Speed: {speed_factor:.2f}x | "
              f"{sent['translated'][:30]}")
    print("-" * 60 + "\n")
    
    return sentences


def add_silence(duration_ms: float) -> AudioSegment:
    """Create a silent audio segment."""
    return AudioSegment.silent(duration=int(duration_ms))


def merge_audio_segments_with_silence(sentences: List[Dict], output_audio: str) -> str:
    """
    Merge individual sentence audio files with silence gaps in between.
    This maintains timing accuracy.
    """
    print("=" * 60)
    print("[Step 6/8] Merging audio segments with timing alignment...")
    print("=" * 60)
    
    combined = AudioSegment.empty()
    current_time = 0  # Track current position in audio
    
    print("Merging segments:")
    print("-" * 60)
    for i, sent in enumerate(sentences):
        if not sent['adjusted_audio_path'] or not Path(sent['adjusted_audio_path']).exists():
            print(f"[{i+1:2d}] ✗ Skipping: audio file not found")
            continue
        
        audio_segment = AudioSegment.from_file(sent['adjusted_audio_path'])
        segment_duration = len(audio_segment)  # in milliseconds
        
        # Add silence if there's a gap
        target_start = sent['start'] * 1000  # Convert to milliseconds
        if current_time < target_start:
            silence_duration = target_start - current_time
            combined += add_silence(silence_duration)
            print(f"[{i+1:2d}] + Added silence: {silence_duration/1000:.2f}s")
        
        # Add the audio segment
        combined += audio_segment
        current_time = target_start + segment_duration
        
        print(f"[{i+1:2d}] ✓ Added: {sent['translated'][:35]}... "
              f"(at {sent['start']:.2f}s, duration {segment_duration/1000:.2f}s)")
    
    # Export combined audio
    combined.export(output_audio, format="mp3")
    print("-" * 60)
    print(f"✓ Total audio duration: {len(combined)/1000:.2f}s")
    print(f"✓ Audio merged: {output_audio}\n")
    
    return output_audio


def replace_audio_in_video(video_path: str, new_audio_path: str, output_video: str) -> str:
    """Replace original audio with new Vietnamese audio."""
    print("=" * 60)
    print("[Step 7/8] Replacing audio in video...")
    print("=" * 60)
    subprocess.run([
        "ffmpeg", "-y", "-i", video_path,
        "-i", new_audio_path,
        "-c:v", "copy",
        "-c:a", "aac",
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-shortest",
        output_video,
    ], check=True, capture_output=True)
    print(f"✓ Video output: {output_video}\n")
    return output_video


def save_translation_log(sentences: List[Dict], output_file: str) -> None:
    """Save translation and timing information."""
    print("=" * 60)
    print("[Step 8/8] Saving translation log...")
    print("=" * 60)
    
    log_data = []
    total_time_shift = 0
    
    for sent in sentences:
        time_shift = abs(sent['tts_duration'] - sent['duration'])
        total_time_shift += time_shift
        
        log_data.append({
            "id": sent['id'],
            "start": sent['start'],
            "end": sent['end'],
            "original_duration": sent['duration'],
            "tts_duration": sent['tts_duration'],
            "adjusted_duration": sent['duration'],  # After speed adjustment
            "speed_factor": sent['speed_factor'],
            "time_shift": time_shift,
            "original": sent['original'],
            "translated": sent['translated']
        })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
    
    avg_time_shift = total_time_shift / len(sentences) if sentences else 0
    print(f"✓ Translation log saved: {output_file}")
    print(f"✓ Average time shift per sentence: {avg_time_shift:.3f}s")
    print(f"✓ Total segments: {len(sentences)}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Advanced video dubbing with sentence-by-sentence processing and timing alignment."
    )
    parser.add_argument("--input", required=True, help="Path to input video file")
    parser.add_argument("--output", required=True, help="Path to output video file")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"),
                        help="OpenAI API key")
    parser.add_argument("--voice", default="vi-VN-NhanNeural",
                        help="Vietnamese Edge TTS voice")
    parser.add_argument("--language", default="en",
                        help="Source language for Whisper transcription (en, fr, es, etc.)")
    parser.add_argument("--keep-temp", action="store_true",
                        help="Keep temporary files for debugging")
    return parser.parse_args()


async def main() -> None:
    args = parse_args()
    
    if not args.api_key:
        print("✗ Missing OpenAI API key. Set OPENAI_API_KEY or pass --api-key")
        sys.exit(1)
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"✗ Input file not found: {input_path}")
        sys.exit(1)
    
    # Create temp directory
    temp_dir = Path("temp")
    temp_dir.mkdir(exist_ok=True)
    
    # Define file paths
    audio_path = temp_dir / "extracted_audio.mp3"
    merged_audio_path = temp_dir / "merged_audio.mp3"
    output_path = Path(args.output)
    log_path = Path("translation_log.json")
    
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  VIDEO DUBBING TOOL - Advanced Mode  ".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    
    try:
        start_time = time.time()
        
        # Step 1: Extract audio
        extract_audio(str(input_path), str(audio_path))
        
        # Step 2: Transcribe with timestamps
        sentences = transcribe_audio_with_timestamps(str(audio_path), language=args.language)
        
        # Step 3: Translate sentences
        sentences = translate_sentences_batch(sentences, args.api_key)
        
        # Step 4: Generate TTS for each sentence
        sentences = await generate_tts_for_all_sentences(sentences, str(temp_dir), args.voice)
        
        # Step 5: Adjust audio speed
        sentences = adjust_all_audio_speeds(sentences)
        
        # Step 6: Merge audio segments
        merge_audio_segments_with_silence(sentences, str(merged_audio_path))
        
        # Step 7: Replace audio in video
        replace_audio_in_video(str(input_path), str(merged_audio_path), str(output_path))
        
        # Step 8: Save translation log
        save_translation_log(sentences, str(log_path))
        
        elapsed_time = time.time() - start_time
        print("=" * 60)
        print("✓ DUBBING COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print(f"Output video: {output_path}")
        print(f"Translation log: {log_path}")
        print(f"Processing time: {elapsed_time:.1f}s")
        print()
        
        # Cleanup temp files if not keeping them
        if not args.keep_temp:
            import shutil
            shutil.rmtree(temp_dir)
            print("Temporary files cleaned up.")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
