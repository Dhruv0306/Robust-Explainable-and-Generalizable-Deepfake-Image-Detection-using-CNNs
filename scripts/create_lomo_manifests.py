from pathlib import Path

import pandas as pd


SOURCE = Path("data/manifests/manifest.csv")
OUTPUT_DIR = Path("data/manifests/lomo")

ORIGINAL = "Original"
MANIPULATIONS = [
    "Deepfakes",
    "Face2Face",
    "FaceSwap",
    "NeuralTextures",
]
EXPECTED_CATEGORIES = {ORIGINAL, *MANIPULATIONS}
EXPECTED_SPLIT_COUNTS = {
    "train": 100,
    "val": 18,
    "test": 12,
}


def unique_video_counts(df):
    """Count distinct videos for each category and split."""
    return (
        df.drop_duplicates(["video_id", "category", "split"])
        .groupby(["category", "split"])["video_id"]
        .nunique()
        .unstack(fill_value=0)
    )


def validate_source(df):
    required = {
        "frame_path", "video_id", "category", "label", "split"
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

    if df[list(required)].isna().any().any():
        raise ValueError("Required manifest columns contain missing values.")

    categories = set(df["category"].unique())
    if categories != EXPECTED_CATEGORIES:
        raise ValueError(
            f"Unexpected categories. Found {sorted(categories)}, "
            f"expected {sorted(EXPECTED_CATEGORIES)}"
        )

    splits = set(df["split"].unique())
    if splits != set(EXPECTED_SPLIT_COUNTS):
        raise ValueError(f"Unexpected split names: {sorted(splits)}")

    expected_labels = {ORIGINAL: "real"}
    expected_labels.update({category: "fake" for category in MANIPULATIONS})

    for category, expected_label in expected_labels.items():
        labels = set(df.loc[df["category"] == category, "label"].astype(str).str.lower())
        if labels != {expected_label}:
            raise ValueError(
                f"{category} has labels {sorted(labels)}, "
                f"expected only {expected_label!r}"
            )

    video_splits = (
        df.drop_duplicates(["video_id", "category", "split"])
        .groupby(["video_id", "category"])["split"]
        .nunique()
    )
    if (video_splits > 1).any():
        raise ValueError("At least one video appears in multiple splits within a category.")

    if df["frame_path"].duplicated().any():
        raise ValueError("Duplicate frame_path values found in the source manifest.")


def validate_lomo_manifest(df, held_out):
    train_val = df[df["split"].isin(["train", "val"])]
    test = df[df["split"] == "test"]

    if held_out in set(train_val["category"]):
        raise ValueError(f"Held-out category {held_out} is present in train/val.")

    expected_train_val_categories = EXPECTED_CATEGORIES - {held_out}
    if set(train_val["category"].unique()) != expected_train_val_categories:
        raise ValueError(f"Unexpected train/val categories for held-out {held_out}.")

    expected_test_categories = {ORIGINAL, held_out}
    if set(test["category"].unique()) != expected_test_categories:
        raise ValueError(f"Unexpected test categories for held-out {held_out}.")

    counts = unique_video_counts(df)
    for category in expected_train_val_categories:
        for split in ("train", "val"):
            actual = int(counts.loc[category, split])
            expected = EXPECTED_SPLIT_COUNTS[split]
            if actual != expected:
                raise ValueError(
                    f"{held_out}: {category}/{split} has {actual} videos, "
                    f"expected {expected}."
                )

    for category in expected_test_categories:
        actual = int(counts.loc[category, "test"])
        expected = EXPECTED_SPLIT_COUNTS["test"]
        if actual != expected:
            raise ValueError(
                f"{held_out}: {category}/test has {actual} videos, "
                f"expected {expected}."
            )

    if set(test["label"].astype(str).str.lower()) != {"real", "fake"}:
        raise ValueError(f"{held_out}: test set does not contain both labels.")


def main():
    if not SOURCE.exists():
        raise FileNotFoundError(f"Source manifest not found: {SOURCE}")

    df = pd.read_csv(SOURCE)
    validate_source(df)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Source manifest validated.")
    print("\nCreating LOMO manifests...")

    for held_out in MANIPULATIONS:
        keep_train_val = (
            df["split"].isin(["train", "val"])
            & (df["category"] != held_out)
        )
        keep_test = (
            (df["split"] == "test")
            & df["category"].isin([ORIGINAL, held_out])
        )

        lomo_df = df.loc[keep_train_val | keep_test].copy()
        validate_lomo_manifest(lomo_df, held_out)

        output_path = OUTPUT_DIR / f"manifest_holdout_{held_out}.csv"
        lomo_df.to_csv(output_path, index=False)

        print(f"\nHeld out: {held_out}")
        print(unique_video_counts(lomo_df).to_string())
        print(f"Rows: {len(lomo_df):,}")
        print(f"Saved: {output_path}")

    print("\nAll four LOMO manifests created and validated.")


if __name__ == "__main__":
    main()