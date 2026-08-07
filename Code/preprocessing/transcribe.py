"""Interview transcription with OpenAI's Whisper, run locally.

Matches the paper: Whisper "base" model, no diarization needed since
each speaker was recorded on a separate channel.
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Code/
from config import AUDIO_CUT_DIR, TRANSCRIPTS_DIR

warnings.filterwarnings("ignore", message="FP16 is not supported on CPU; using FP32 instead")

import whisper  # pip install git+https://github.com/openai/whisper.git

WHISPER_MODEL_NAME = "base"


def transcribe_recordings(audio_dir: Path = AUDIO_CUT_DIR, output_dir: Path = TRANSCRIPTS_DIR) -> None:
    # whisper.load_model() downloads once and caches under ~/.cache/whisper
    model = whisper.load_model(WHISPER_MODEL_NAME)
    output_dir.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(audio_dir.glob("*.wav"))
    for i, wav_path in enumerate(wav_files):
        print(f"{i} transcribing {wav_path.name}")
        result = model.transcribe(
            str(wav_path),
            language="en",
            condition_on_previous_text=False,
            hallucination_silence_threshold=2,
        )
        output_path = output_dir / f"{wav_path.stem}.txt"
        output_path.write_text(result["text"], encoding="utf-8")

    print(f"Done. Transcribed {len(wav_files)} recordings -> {output_dir}")


if __name__ == "__main__":
    transcribe_recordings()
