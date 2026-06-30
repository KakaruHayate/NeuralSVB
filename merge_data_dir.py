"""Merge data/raw/PopBuTFy/data/* directories up one level into data/raw/PopBuTFy/.

PopBuTFy.zip has its content under a `data/` subdirectory so unzipping gives:
  data/raw/PopBuTFy/data/Female1#.../*.mp3

While text_labels.zip has no prefix, giving:
  data/raw/PopBuTFy/Female1#.../*.txt

We need everything at the same level: data/raw/PopBuTFy/Female1#.../*.{mp3,txt}
"""
import shutil
from pathlib import Path

raw_dir = Path('data/raw/PopBuTFy')
src_dir = raw_dir / 'data'

if not src_dir.is_dir():
    print(f"Source directory {src_dir} not found. Nothing to merge.")
    exit(0)

items = list(src_dir.iterdir())
print(f"Merging {len(items)} items from {src_dir}/ to {raw_dir}/ ...")

for item in items:
    dest = raw_dir / item.name
    if item.is_dir():
        if dest.exists():
            # Directory exists (from text_labels), merge files
            for f in item.iterdir():
                shutil.move(str(f), str(dest / f.name))
            item.rmdir()  # Remove the now-empty source dir
        else:
            # Doesn't exist at dest level - just move the whole thing
            shutil.move(str(item), str(dest))
    else:
        # File (like TERMS_OF_ACCESS) - just move
        if dest.exists():
            dest.unlink()
        shutil.move(str(item), str(dest))

# Remove the now-empty data/ directory
if src_dir.is_dir():
    remaining = list(src_dir.iterdir())
    if not remaining:
        src_dir.rmdir()
        print(f"Removed empty {src_dir}/")

print("Merge complete.")

# Verify
total_dirs = sum(1 for d in raw_dir.iterdir() if d.is_dir())
print(f"Total directories under {raw_dir}/: {total_dirs}")
