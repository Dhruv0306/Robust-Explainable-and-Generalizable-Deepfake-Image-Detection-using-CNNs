"""
Configuration for Approach 1: CNN Baseline Deepfake Detection
All hyperparameters, paths, and constants in one place.
"""
from pathlib import Path
import socket
import warnings

# Suppress torch's noisy CUDA warning; get_device() reports the usable fallback clearly.
warnings.filterwarnings('ignore', category=UserWarning, module=r'torch\.cuda\..*')

# === Paths ===
PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "data"
DATASET_ROOT = DATA_ROOT / "datasets" / "FaceForensics++"
OUTPUT_ROOT = DATA_ROOT / "output"
CHECKPOINT_ROOT = DATA_ROOT / "checkpoints"

# Raw video paths
ORIGINAL_VIDEOS = DATASET_ROOT / "original_sequences" / "youtube" / "c23" / "videos"
DEEPFAKES_VIDEOS = DATASET_ROOT / "manipulated_sequences" / "Deepfakes" / "c23" / "videos"
FACE2FACE_VIDEOS = DATASET_ROOT / "manipulated_sequences" / "Face2Face" / "c23" / "videos"
FACESWAP_VIDEOS = DATASET_ROOT / "manipulated_sequences" / "FaceSwap" / "c23" / "videos"
NEURALTEXTURES_VIDEOS = DATASET_ROOT / "manipulated_sequences" / "NeuralTextures" / "c23" / "videos"

# Processed output paths
FRAMES_ROOT = DATA_ROOT / "frames"
MANIFESTS_ROOT = DATA_ROOT / "manifests"
SPLITS_ROOT = DATA_ROOT / "splits"

# === Dataset ===
CATEGORIES = ["Original", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]
VIDEOS_PER_CATEGORY = 130
TOTAL_VIDEOS = 650

# Split allocation (subject to graph verification)
TRAIN_GROUPS = 100
VAL_GROUPS = 20
TEST_GROUPS = 10

# === Frame extraction ===
FPS_ASSUMPTION = 30
FRAME_SAMPLING_INTERVAL = 4  # every 4th frame
EFFECTIVE_SAMPLING_FPS = 7.5

# === Face processing ===
MIN_USABLE_FRAMES_PER_VIDEO = 20
BBOX_EXPANSION_MARGIN = 0.3  # 30% expansion around detected face bbox
MIN_FACE_SIZE = 50  # minimum face dimension in pixels
IOU_THRESHOLD = 0.5  # IoU threshold for face association across frames

# === Models ===
MODELS = {
    "xception": {
        "timm_name": "legacy_xception",
        "input_size": 299,
        "pretrained": True,
    },
    "efficientnet_b0": {
        "timm_name": "efficientnet_b0",
        "input_size": 224,
        "pretrained": True,
    },
    "resnet50": {
        "timm_name": "resnet50",
        "input_size": 224,
        "pretrained": True,
    },
}

# === Training ===
SEEDS = [42, 123, 2024]
MAX_EPOCHS = 30
EARLY_STOPPING_PATIENCE = 5
SCHEDULER_PATIENCE = 2

# Optimizer
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

# Batch size (auto-determined by available memory, fallback values)
BATCH_SIZE_GPU = 32
BATCH_SIZE_CPU = 8
NUM_WORKERS = 4

# === Augmentation ===
HFLIP_PROB = 0.5
GAUSSIAN_BLUR_PROB = 0.1
GAUSSIAN_BLUR_KERNEL_SIZES = [3, 5]
GAUSSIAN_BLUR_SIGMA_RANGE = (0.1, 2.0)

# === Evaluation ===
VIDEO_AGGREGATION_METHODS = ["mean", "median", "mode"]
PRIMARY_AGGREGATION = "mean"

# === Runtime ===
def get_run_name(model_name: str) -> str:
    """Generate run directory name: <model>_<pc>_<timestamp>"""
    from datetime import datetime
    pc_name = socket.gethostname().replace(" ", "-")
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{model_name}_{pc_name}_{timestamp}"
