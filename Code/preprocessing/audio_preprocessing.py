"""Audio preprocessing: .m4a -> .wav conversion and long-silence removal.

Interviews were recorded in Zoom with each speaker on a separate audio
channel, so no speaker diarization is needed here.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # Code/
from config import AUDIO_RAW_DIR, AUDIO_CUT_DIR

from pydub import AudioSegment
from pydub.silence import split_on_silence

SILENCE_THRESHOLD_DB = -50
MIN_SILENCE_LEN_MS = 5000


def convert_m4a_to_wav(m4a_path: Path, wav_path: Path) -> None:
    audio = AudioSegment.from_file(m4a_path, format="m4a")
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    audio.export(wav_path, format="wav")


def remove_long_silences(
    audio_path: Path,
    output_path: Path,
    silence_threshold: int = SILENCE_THRESHOLD_DB,
    silence_length: int = MIN_SILENCE_LEN_MS,
) -> None:
    audio = AudioSegment.from_file(audio_path, format="wav")
    chunks = split_on_silence(
        audio,
        min_silence_len=silence_length,
        silence_thresh=silence_threshold,
        keep_silence=0,
    )
    processed_audio = sum(chunks)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed_audio.export(output_path, format="wav")


def process_raw_recordings(raw_dir: Path = AUDIO_RAW_DIR, cut_dir: Path = AUDIO_CUT_DIR) -> None:
    """Convert every .m4a under raw_dir to a silence-trimmed .wav in cut_dir.

    raw_dir is searched recursively, so organizing recordings into
    cohort subfolders (e.g. CLBP/MDD/HC) is optional.
    """
    m4a_files = sorted(raw_dir.rglob("*.m4a"))
    for i, m4a_path in enumerate(m4a_files):
        print(f"{i} processing {m4a_path.name}")
        tmp_wav = cut_dir / f"_tmp_{m4a_path.stem}.wav"
        convert_m4a_to_wav(m4a_path, tmp_wav)
        output_path = cut_dir / f"{m4a_path.stem}.wav"
        remove_long_silences(tmp_wav, output_path)
        tmp_wav.unlink()
    print(f"Done. Processed {len(m4a_files)} recordings -> {cut_dir}")


if __name__ == "__main__":
    process_raw_recordings()
