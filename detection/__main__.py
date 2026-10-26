"""Package entrypoint for detection experiments.

Run with:
    python -m detection
"""

from detection.ModelDetection import (
    BERTContrastiveDetectionModel,
    CRFBERTDetectionModel,
    VanillaBERTDetectionModel,
)


def main() -> None:
    models = [
        VanillaBERTDetectionModel.__name__,
        CRFBERTDetectionModel.__name__,
        BERTContrastiveDetectionModel.__name__,
    ]
    print("detection package loaded correctly.")
    print("Available model classes:")
    for model_name in models:
        print(f"- {model_name}")
    print("Use a dedicated train/infer runner to instantiate one of these classes.")


if __name__ == "__main__":
    main()
