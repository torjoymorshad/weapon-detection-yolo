
from pathlib import Path

dataset = Path("dataset")
errors = []
total_boxes = 0

for split in ["train", "val", "test"]:
    for file in (dataset / "labels" / split).glob("*.txt"):
        for line_no, line in enumerate(file.read_text().splitlines(), 1):
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
