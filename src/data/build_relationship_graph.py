"""
Build relationship graph from manipulated video filenames.
Parse target_source relationships, canonicalize reverse pairs.
"""
from pathlib import Path
import logging
from typing import Dict, Set, Tuple, List
import sys
sys.path.append(str(Path(__file__).parent.parent))
from config import *


def parse_video_id(filename: str) -> Tuple[str, str]:
    """
    Parse manipulated video filename: targetID_sourceID.mp4 -> (target, source)
    Original videos: just ID.mp4 -> (ID, None)
    """
    stem = Path(filename).stem
    if "_" in stem:
        target, source = stem.split("_", 1)
        return target, source
    else:
        return stem, None


def canonicalize_pair(target: str, source: str) -> Tuple[str, str]:
    """Canonicalize (target, source) as sorted tuple"""
    return tuple(sorted([target, source]))


def build_relationship_graph(video_paths: Dict[str, List[Path]]) -> Dict[str, Set[Tuple[str, str]]]:
    """
    Build relationship graph from video filenames.
    Returns: {
        'nodes': set of video IDs,
        'edges': set of canonicalized (id1, id2) pairs
    }
    """
    logging.info("Building relationship graph...")

    nodes = set()
    edges = set()

    # Original videos: just nodes
    for video_path in video_paths["Original"]:
        video_id, _ = parse_video_id(video_path.name)
        nodes.add(video_id)

    # Manipulated videos: nodes + edges
    for category in ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]:
        for video_path in video_paths[category]:
            target_id, source_id = parse_video_id(video_path.name)
            nodes.add(target_id)
            nodes.add(source_id)
            # Add canonicalized edge
            edges.add(canonicalize_pair(target_id, source_id))

    logging.info(f"Graph: {len(nodes)} nodes, {len(edges)} edges")

    return {"nodes": nodes, "edges": edges}


def find_connected_components(graph: Dict) -> List[Set[str]]:
    """Find connected components using DFS"""
    logging.info("Finding connected components...")

    nodes = graph["nodes"]
    edges = graph["edges"]

    # Build adjacency list
    adj = {node: set() for node in nodes}
    for id1, id2 in edges:
        adj[id1].add(id2)
        adj[id2].add(id1)

    visited = set()
    components = []

    def dfs(node, component):
        visited.add(node)
        component.add(node)
        for neighbor in adj[node]:
            if neighbor not in visited:
                dfs(neighbor, component)

    for node in nodes:
        if node not in visited:
            component = set()
            dfs(node, component)
            components.append(component)

    logging.info(f"Found {len(components)} connected components")

    # Log component size distribution
    sizes = sorted([len(c) for c in components], reverse=True)
    logging.info(f"Component sizes: min={min(sizes)}, max={max(sizes)}, mean={sum(sizes)/len(sizes):.1f}")

    return components


if __name__ == "__main__":
    from utils import setup_logging
    from inspect_dataset import inspect_dataset

    setup_logging()
    video_paths = inspect_dataset()
    graph = build_relationship_graph(video_paths)
    components = find_connected_components(graph)
