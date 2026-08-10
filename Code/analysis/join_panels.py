"""Combine panels A/B and C into one vertically-stacked figure with
panel letter labels, matching the paper's Fig. 4 / eFigure 2 layout.
"""
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

FONT_PATH = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
LABEL_FONTSIZE = 120
GAP = 20
LABEL_Y_OFFSET = 20


def join_ab_c(img_ab_path, img_c_path, output_path, label_fontsize: int = LABEL_FONTSIZE, gap: int = GAP):
    """Stack the A/B panel image above the C panel image, centering C and
    adding "A"/"B"/"C" labels. C is scaled to fit the A/B panel's width."""
    img_ab = Image.open(img_ab_path)
    img_c = Image.open(img_c_path)
    font = ImageFont.truetype(FONT_PATH, label_fontsize)

    target_width = img_ab.width
    c_scale = min(target_width * 0.95 / img_c.width, 3)
    new_c_width = int(img_c.width * c_scale)
    new_c_height = int(img_c.height * c_scale)
    img_c_resized = img_c.resize((new_c_width, new_c_height), Image.LANCZOS)

    label_height = label_fontsize + 15
    total_height = label_height + img_ab.height + gap + label_height + new_c_height
    canvas = Image.new("RGB", (target_width, total_height), "white")
    draw = ImageDraw.Draw(canvas)

    draw.text((15, LABEL_Y_OFFSET), "A", fill="black", font=font)
    b_x = target_width // 2 + 30
    draw.text((b_x, LABEL_Y_OFFSET), "B", fill="black", font=font)
    canvas.paste(img_ab, (0, label_height))

    c_label_y = label_height + img_ab.height + gap
    draw.text((15, c_label_y + LABEL_Y_OFFSET), "C", fill="black", font=font)
    c_x = (target_width - new_c_width) // 2
    canvas.paste(img_c_resized, (c_x, c_label_y + label_height))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output_path, dpi=(300, 300))
    return canvas
