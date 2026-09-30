import argparse
import asyncio
import os
import subprocess
import sys
import json
from pathlib import Path
from typing import List, Dict

import edge_tts
import whisper
from openai import OpenAI
from pydub import AudioSegment


def extract_audio(input_video: str, output_audio: str) -> str:
    """Extract audio from video."""
    print("[Step 1/7] Extracting audio from video...")
    subprocess.run([
        "ffmpeg", "-y", "-i", input_video,
        "-vn", "-acodec", "libmp3lame", "-q:a", "5",
        output_audio
    ], check=True, capture_output=True)
    print(f"✓ Audio extracted: {output_audio}")
    return output_audio


def transcribe_audio_with_timestamps(audio_path: str, language: str = "en") -> List[Dict]:
    """Transcribe audio and extract sentence-level timestamps."""
    print("[Step 2/7] Transcribing audio with timestamps...")
    model = whisper.load_model("base")
    result = model.transcribe(audio_path, language=language, fp16=False)
    
    segments = result.get("segments", [])
    sentences = []
    
    for segment in segments:
        start = segment["start"]
        end = segment["end"]
        text = segment["text"].strip()
        
        if text:
            sentences.append({
                "start": start,
                "end": end,
                "duration": end - start,
                "original": text,
                "translated": "",
                "audio_path": ""
            })
    
    print(f"✓ Found {len(sentences)} sentences")
    for i, sent in enumerate(sentences):
        print(f"  [{i+1}] ({sent['start']:.2f}s - {sent['end']:.2f}s): {sent['original']}")
    
    return sentences


def translate_sentences_batch(sentences: List[Dict], api_key: str) -> List[Dict]:
    """Translate all sentences to Vietnamese using GPT."""
    print("\n[Step 3/7] Translating sentences to Vietnamese...")
    client = OpenAI(api_key=api_key)
    
    # Combine all sentences for context-aware translation
    original_texts = "\n".join([f"[{i}] {s['original']}" for i, s in enumerate(sentences)])
    
    prompt = (
        "Bạn là biên dịch viên chuyên nghiệp. Dịch từng câu dưới đây sang tiếng Việt tự nhiên. "
        "Giữ nguyên ý nghĩa, giọng điệu và không giải thích thêm.\n"
        "Trả về kết quả theo định dạng JSON:\n"
        '{"translations": ["dịch câu 1", "dịch câu 2", ...]}\n\n'
        f"{original_texts}"
    )
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "Bạn là biên dịch viên chuyên nghiệp."},
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
        print("⚠ Could not parse JSON, using fallback translation")
        translations = [s['original'] for s in sentences]
    
    for i, sent in enumerate(sentences):
        if i < len(translations):
            sent['translated'] = translations[i].strip()
        else:
            sent['translated'] = sent['original']
        print(f"  [{i+1}] {sent['translated']}")
    
    return sentences


async def generate_tts_for_sentences(sentences: List[Dict], output_dir: str, voice: str) -> List[Dict]:
    """Generate TTS audio for each sentence."""
    print(f"\n[Step 4/7] Generating TTS audio for {len(sentences)} sentences...")
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    for i, sent in enumerate(sentences):
        text = sent['translated']
        audio_file = output_path / f"sentence_{i:03d}.mp3"
        
        try:
            communicate = edge_tts.Communicate(text, voice)
            await communicate.save(str(audio_file))
            sent['audio_path'] = str(audio_file)
            print(f"  [{i+1}] Generated: {audio_file.name}")
        except Exception as e:
            print(f"  ✗ Error generating TTS for sentence {i+1}: {e}")
            sent['audio_path'] = None
    
    return sentences


def adjust_audio_speed(audio_path: str, target_duration: float, output_path: str) -> float:
    """Adjust audio speed to fit target duration."""
    audio = AudioSegment.from_file(audio_path)
    current_duration_ms = len(audio)
    target_duration_ms = target_duration * 1000
    
    if current_duration_ms == 0:
        return 1.0
    
    speed_factor = current_duration_ms / target_duration_ms
    
    # Limit speed changes to reasonable range (0.5x to 2.0x)
    speed_factor = max(0.5, min(2.0, speed_factor))
    
    subprocess.run([
        "ffmpeg", "-y", "-i", audio_path,
        "-filter:a", f"atempo={speed_factor}",
        output_path
    ], check=True, capture_output=True)
    
    return speed_factor


def merge_audio_segments(sentences: List[Dict], output_audio: str) -> str:
    """Merge individual sentence audio files with gaps."""
    print("\n[Step 5/7] Merging audio segments...")
    
    combined = AudioSegment.empty()
    
    for i, sent in enumerate(sentences):
        if not sent['audio_path'] or not Path(sent['audio_path']).exists():
            print(f"  ✗ Skipping sentence {i+1}: audio file not found")
            continue
        
        # Adjust audio speed to fit original duration
        adjusted_audio_path = f"temp/adjusted_{i:03d}.mp3"
        speed_factor = adjust_audio_speed(
            sent['audio_path'],
            sent['duration'],
            adjusted_audio_path
        )
        
        audio_segment = AudioSegment.from_file(adjusted_audio_path)
        combined += audio_segment
        print(f"  [{i+1}] Added: {sent['translated'][:50]}... (speed: {speed_factor:.2f}x)")
    
    combined.export(output_audio, format="mp3")
    print(f"✓ Audio merged: {output_audio}")
    return output_audio


def replace_audio_in_video(video_path: str, new_audio_path: str, output_video: str) -> str:
    """Replace original audio with new Vietnamese audio."""
    print("\n[Step 6/7] Replacing audio in video...")
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
    print(f"✓ Video output: {output_video}")
    return output_video


def save_translation_log(sentences: List[Dict], output_file: str) -> None:
    """Save translation and timing information."""
    print(f"\n[Step 7/7] Saving translation log...")
    log_data = []
    for sent in sentences:
        log_data.append({
            "start": sent['start'],
            "end": sent['end'],
            "duration": sent['duration'],
            "original": sent['original'],
            "translated": sent['translated']
        })
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(log_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ Translation log saved: {output_file}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Replace video audio with Vietnamese voiceover (sentence by sentence)"
    )
    parser.add_argument("--input", required=True, help="Path to input video file")
    parser.add_argument("--output", required=True, help="Path to output video file")
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY"),
                        help="OpenAI API key")
    parser.add_argument("--voice", default="vi-VN-NhanNeural",
                        help="Vietnamese Edge TTS voice (vi-VN-NhanNeural for male, vi-VN-HoaiMyNeural for female)")
    parser.add_argument("--language", default="en",
                        help="Source language for Whisper transcription")
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
    
    try:
        # Step 1: Extract audio
        extract_audio(str(input_path), str(audio_path))
        
        # Step 2: Transcribe with timestamps
        sentences = transcribe_audio_with_timestamps(str(audio_path), language=args.language)
        
        # Step 3: Translate sentences
        sentences = translate_sentences_batch(sentences, args.api_key)
        
        # Step 4: Generate TTS for each sentence
        sentences = await generate_tts_for_sentences(sentences, str(temp_dir), args.voice)
        
        # Step 5: Merge audio segments
        merge_audio_segments(sentences, str(merged_audio_path))
        
        # Step 6: Replace audio in video
        replace_audio_in_video(str(input_path), str(merged_audio_path), str(output_path))
        
        # Step 7: Save translation log
        save_translation_log(sentences, str(log_path))
        
        print(f"\n✓ Done! Output saved to: {output_path}")
        print(f"✓ Translation log saved to: {log_path}")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
