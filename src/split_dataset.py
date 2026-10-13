
from pathlib import Path
import random
import shutil

root = Path(__file__).resolve().parents[1] / "dataset"
images = root / "images"
labels = root / "labels"

files = sorted(
    p for p in (images / "train").iterdir()
    if p.suffix.lower() in [".jpg", ".jpeg", ".png"]
)

# Check all image-label pairs before moving anything
for image in files:
    assert (labels / "train" / (image.stem + ".txt")).exists(), (
        f"Missing label: {image.name}"
    )

random.seed(42)
random.shuffle(files)

n = len(files)
n_val = round(n * 0.10)
n_test = round(n * 0.10)

splits = {
    "val": files[:n_val],
    "test": files[n_val:n_val + n_test]
}

for split, selected in splits.items():
    for image in selected:
        label = labels / "train" / (image.stem + ".txt")

        shutil.move(str(image), str(images / split / image.name))
        shutil.move(str(label), str(labels / split / label.name))

print("Dataset split complete!")
for split in ["train", "val", "test"]:
    count = len(list((images / split).iterdir()))
    print(f"{split}: {count} images")
    
from pathlib import Path

dataset = Path("dataset")
errors = []
total_boxes = 0

for split in ["train", "val", "test"]:
    for file in (dataset / "labels" / split).glob("*.txt"):
        for line_no, line in enumerate(
            file.read_text().splitlines(), 1
        ):
            if not line.strip():
                continue

            try:
                values = list(map(float, line.split()))

                if len(values) != 5:
                    raise ValueError("Expected 5 values")

                cls, x, y, w, h = values

                if cls not in [0, 1, 2, 3]:
                    raise ValueError("Invalid class ID")

                if not (
                    0 <= x <= 1 and 0 <= y <= 1
                    and 0 < w <= 1 and 0 < h <= 1
                    and 0 <= x - w / 2
                    and x + w / 2 <= 1
                    and 0 <= y - h / 2
                    and y + h / 2 <= 1
                ):
                    raise ValueError("Invalid bounding box")

                total_boxes += 1

            except ValueError as e:
                errors.append(f"{file}:{line_no} - {e}")

print(f"Valid bounding boxes: {total_boxes}")
print(f"Annotation errors: {len(errors)}")

for error in errors[:10]:
    print(error)

