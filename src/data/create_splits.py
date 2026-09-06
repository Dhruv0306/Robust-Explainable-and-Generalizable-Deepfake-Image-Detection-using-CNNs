"""
Create leakage-safe train/val/test splits from connected components.
"""
from pathlib import Path
import logging
import json
import random
from typing import Dict, List, Set
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import *


def assign_splits(components: List[Set[str]], seed: int = 42) -> Dict[str, Set[str]]:
    """
    Assign connected components to train/val/test splits.
    Returns dict: {'train': set of video IDs, 'val': ..., 'test': ...}
    """
    logging.info("Assigning splits...")

    # Verify component count
    n_components = len(components)
    expected_total = TRAIN_GROUPS + VAL_GROUPS + TEST_GROUPS
    logging.info(f"Components: {n_components}, expected: {expected_total}")

    if n_components != expected_total:
        logging.warning(f"Component count mismatch! Using actual count {n_components}")
        # Adjust split sizes proportionally
        total = n_components
        train_n = int(total * 0.77)  # ~77%
        val_n = int(total * 0.15)    # ~15%
        test_n = total - train_n - val_n  # ~8%
    else:
        train_n, val_n, test_n = TRAIN_GROUPS, VAL_GROUPS, TEST_GROUPS

    # Shuffle components deterministically
    rng = random.Random(seed)
    shuffled = components.copy()
    rng.shuffle(shuffled)

    # Assign to splits
    train_components = shuffled[:train_n]
    val_components = shuffled[train_n:train_n + val_n]
    test_components = shuffled[train_n + val_n:train_n + val_n + test_n]

    splits = {
        "train": set().union(*train_components),
        "val": set().union(*val_components),
        "test": set().union(*test_components),
    }

    logging.info(f"Split video IDs: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

    # Verify no overlap
    assert len(splits["train"] & splits["val"]) == 0
    assert len(splits["train"] & splits["test"]) == 0
    assert len(splits["val"] & splits["test"]) == 0

    logging.info("Splits created")
    return splits


def save_splits(splits: Dict[str, Set[str]], output_dir: Path):
    """Save splits as JSON"""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "splits.json"

    # Convert sets to sorted lists
    splits_serializable = {
        k: sorted(list(v)) for k, v in splits.items()
    }

    with open(output_file, "w") as f:
        json.dump(splits_serializable, f, indent=2)

    logging.info(f"Splits saved to {output_file}")


if __name__ == "__main__":
    from utils import setup_logging, set_seed
    from inspect_dataset import inspect_dataset
    from build_relationship_graph import build_relationship_graph, find_connected_components

    setup_logging()
    set_seed(42)

    video_paths = inspect_dataset()
    graph = build_relationship_graph(video_paths)
    components = find_connected_components(graph)
    splits = assign_splits(components, seed=42)
    save_splits(splits, SPLITS_ROOT)
