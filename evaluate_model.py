import argparse
from pathlib import Path

import cv2
import numpy as np
from keras.models import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate keras_model.h5 on a labeled image dataset and report performance metrics."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        required=True,
        help="Root folder containing one subfolder per class (e.g. Cardboard/, Glass/, ...).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("keras_model.h5"),
        help="Path to .h5 model file (default: keras_model.h5).",
    )
    parser.add_argument(
        "--labels-path",
        type=Path,
        default=Path("labels.txt"),
        help="Path to labels file (default: labels.txt).",
    )
    return parser.parse_args()


def load_labels(labels_path: Path) -> list[str]:
    labels = []
    with labels_path.open("r", encoding="utf-8") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            parts = text.split(maxsplit=1)
            labels.append(parts[1] if len(parts) == 2 else parts[0])
    return labels


def preprocess_image(image_path: Path) -> np.ndarray | None:
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_AREA)
    image = np.asarray(image, dtype=np.float32).reshape(1, 224, 224, 3)
    image = (image / 127.5) - 1
    return image


def find_dataset_images(dataset_dir: Path, class_names: list[str]) -> tuple[list[Path], list[int]]:
    image_paths: list[Path] = []
    true_indices: list[int] = []
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for idx, class_name in enumerate(class_names):
        class_dir = dataset_dir / class_name
        if not class_dir.exists():
            print(f"[WARN] Missing class folder: {class_dir}")
            continue

        files = sorted(path for path in class_dir.iterdir() if path.suffix.lower() in extensions)
        image_paths.extend(files)
        true_indices.extend([idx] * len(files))

    return image_paths, true_indices


def print_metrics(cm: np.ndarray, class_names: list[str]) -> None:
    total = cm.sum()
    correct = np.trace(cm)
    accuracy = (correct / total) if total else 0.0

    print("\n=== Overall Performance ===")
    print(f"Samples evaluated : {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy           : {accuracy * 100:.2f}%")

    print("\n=== Per-Class Performance ===")
    print(f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1-Score':>10} {'Support':>10}")

    for i, class_name in enumerate(class_names):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        support = cm[i, :].sum()

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

        print(f"{class_name:<12} {precision:>10.4f} {recall:>10.4f} {f1:>10.4f} {support:>10}")

    print("\n=== Confusion Matrix (rows=true, cols=pred) ===")
    header = " " * 12 + " ".join([f"{name[:10]:>10}" for name in class_names])
    print(header)
    for i, class_name in enumerate(class_names):
        row = " ".join([f"{val:>10}" for val in cm[i]])
        print(f"{class_name[:12]:<12} {row}")


def main() -> None:
    args = parse_args()

    if not args.dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {args.dataset_dir}")
    if not args.model_path.exists():
        raise FileNotFoundError(f"Model file not found: {args.model_path}")
    if not args.labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {args.labels_path}")

    class_names = load_labels(args.labels_path)
    model = load_model(args.model_path, compile=False)

    image_paths, true_indices = find_dataset_images(args.dataset_dir, class_names)
    if not image_paths:
        raise RuntimeError("No images found. Ensure dataset-dir has class subfolders with images.")

    num_classes = len(class_names)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    skipped = 0

    for image_path, true_idx in zip(image_paths, true_indices):
        image = preprocess_image(image_path)
        if image is None:
            skipped += 1
            print(f"[WARN] Skipping unreadable image: {image_path}")
            continue

        prediction = model.predict(image, verbose=0)
        pred_idx = int(np.argmax(prediction))
        cm[true_idx, pred_idx] += 1

    print_metrics(cm, class_names)
    if skipped:
        print(f"\nSkipped images: {skipped}")


if __name__ == "__main__":
    main()
