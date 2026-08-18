"""One-time conversion: split the combined "Human Transcripts LLM-Pain.docx"
(15 manually-transcribed reference interviews, 5 per cohort) into
individual per-subject .txt files matching the automatic transcripts'
naming convention (Data/transcripts/human/<Study ID>_<Dx>.txt).

Not part of the regular reproducible pipeline: the source .docx lives
outside this repo and isn't redistributed (only its output is, since
that's covered by the same IRB approval as the automatic transcripts).
Rerun this only if the source document changes, with SOURCE_DOCX
pointed at your own copy.

Subject headers ("Subject 1221") are their own bold+underlined
paragraph in the source document — detected structurally here rather
than by counting digits in flattened text, which misreads a 5-digit
study ID (13002) as a 4-digit one followed by stray content.

Strips "[uncomprehensive word]" / "[uncomprehensive phrase]" markers
(where the human transcriber could not make out the audio) from the
output text. This is a deliberate simplification for WER scoring (see
wer_transcription_accuracy.py): removing the marker with nothing
leaves no reference token at that position, so if the automatic
transcript did produce a word there, it's counted as an insertion
error even in the (observed at least once) case where the automatic
transcript was actually correct and the human transcriber was not.
Given only 17 such markers across all 15 transcripts, this is treated
as a documented limitation rather than built out with segment-level
alignment/exclusion logic.
"""
import html
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import TRANSCRIPTS_DIR

SOURCE_DOCX = Path("/Users/rnorel/Documents/Neuro/Pain/Data/Human Transcripts LLM-Pain.docx")
OUTPUT_DIR = TRANSCRIPTS_DIR / "human"

PARAGRAPH_PATTERN = re.compile(r"<w:p [^>]*>.*?</w:p>|<w:p>.*?</w:p>", re.DOTALL)
RUN_TEXT_PATTERN = re.compile(r"<w:t[^>]*>(.*?)</w:t>", re.DOTALL)
DX_LABELS = {"CLBP", "MDD", "HC"}
UNCOMPREHENSIVE_PATTERN = re.compile(r"\[uncomprehensive[^\]]*\]")


def _paragraph_text(paragraph_xml: str) -> str:
    return html.unescape("".join(RUN_TEXT_PATTERN.findall(paragraph_xml))).strip()


def split_transcripts(docx_path: Path) -> dict:
    """Return {(study_id, dx): transcript_text}, reading paragraph by
    paragraph. A paragraph that's exactly a cohort label ("CLBP") sets
    the current cohort for subjects until the next label; a paragraph
    that's exactly "Subject <ID>" starts a new subject's transcript,
    which accumulates every following paragraph until the next header.
    """
    with zipfile.ZipFile(docx_path) as z:
        xml = z.read("word/document.xml").decode("utf-8")

    transcripts = {}
    current_dx = None
    current_key = None
    body_parts = []

    def flush():
        if current_key is not None:
            body = UNCOMPREHENSIVE_PATTERN.sub("", " ".join(body_parts))
            transcripts[current_key] = re.sub(r"\s+", " ", body).strip()

    for p_xml in PARAGRAPH_PATTERN.findall(xml):
        text = _paragraph_text(p_xml)
        if not text:
            continue

        if text in DX_LABELS:
            current_dx = text
            continue

        m = re.fullmatch(r"Subject (\d+)", text)
        if m:
            flush()
            current_key = (m.group(1), current_dx)
            body_parts = []
            continue

        body_parts.append(text)

    flush()
    return transcripts


def main():
    transcripts = split_transcripts(SOURCE_DOCX)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for (study_id, dx), body in sorted(transcripts.items()):
        out_path = OUTPUT_DIR / f"{study_id}_{dx}.txt"
        out_path.write_text(body, encoding="utf-8")
        print(f"wrote {out_path.name} ({len(body.split())} words)")

    print(f"\n{len(transcripts)} transcripts -> {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
