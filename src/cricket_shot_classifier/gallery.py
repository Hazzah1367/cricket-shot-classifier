"""Generate numbered contact sheets for label and visual-quality review."""

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from . import CLASSES
from .prepare import read_rows


def contact_sheets(rows, root, output):
    output.mkdir(parents=True, exist_ok=True)
    index = {}
    for label in CLASSES:
        selected = [r for r in rows if r["label"] == label]
        for start in range(0, len(selected), 20):
            sheet = Image.new("RGB", (1500, 1100), "white")
            draw = ImageDraw.Draw(sheet)
            for j, row in enumerate(selected[start : start + 20]):
                with Image.open(root / row["path"]) as image:
                    thumb = ImageOps.contain(
                        ImageOps.exif_transpose(image).convert("RGB"), (295, 235)
                    )
                x, y = (j % 5) * 300, (j // 5) * 275
                sheet.paste(thumb, (x, y))
                image_id = Path(row["path"]).stem
                draw.text((x + 3, y + 240), image_id, fill="black")
                index[image_id] = row["path"]
            sheet.save(output / f"{label}_{start // 20:02d}.jpg", quality=92)
    (output / "index.json").write_text(json.dumps(index, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/candidates.csv"))
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, default=Path("reports/contact_sheets"))
    args = parser.parse_args()
    contact_sheets(read_rows(args.manifest), args.root, args.output)


if __name__ == "__main__":
    main()
