import os
import glob
import numpy as np
import tiktoken
from tqdm import tqdm

# Keep the generated binaries beside this script, matching train.py's data path.
# Set FINEWEB_EDU_DATA_DIR to override the output/cache location.
DATA_DIR = os.environ.get(
    "FINEWEB_EDU_DATA_DIR",
    os.path.dirname(os.path.abspath(__file__)),
)
os.makedirs(DATA_DIR, exist_ok=True)

# Keep dataset caches under the selected data directory as well.
os.environ["HF_HOME"] = DATA_DIR
os.environ["HF_DATASETS_CACHE"] = DATA_DIR
os.environ.setdefault("MODELSCOPE_CACHE", os.path.join(DATA_DIR, ".modelscope"))

from datasets import load_dataset

try:
    from modelscope.hub.snapshot_download import dataset_snapshot_download
except ImportError as exc:
    raise SystemExit(
        "This script requires ModelScope. Install it with `pip install modelscope`."
    ) from exc

NUM_PROC = 8  
SHARD_SIZE = int(1e8)

print("Loading FineWeb-Edu dataset...")
# ModelScope mirror: https://www.modelscope.cn/datasets/AI-ModelScope/fineweb-edu
MODELSCOPE_DATASET = "AI-ModelScope/fineweb-edu"
MODELSCOPE_REVISION = os.environ.get("FINEWEB_EDU_MODELSCOPE_REVISION", "master")
MODELSCOPE_SUBSET_PATH = "sample/10BT"
snapshot_dir = dataset_snapshot_download(
    MODELSCOPE_DATASET,
    revision=MODELSCOPE_REVISION,
    cache_dir=os.path.join(DATA_DIR, ".modelscope"),
    allow_patterns=f"{MODELSCOPE_SUBSET_PATH}/*.parquet",
    max_workers=int(os.environ.get("MODELSCOPE_MAX_WORKERS", NUM_PROC)),
)
parquet_files = sorted(glob.glob(
    os.path.join(snapshot_dir, MODELSCOPE_SUBSET_PATH, "*.parquet")
))
if not parquet_files:
    raise FileNotFoundError(
        f"No parquet files found for {MODELSCOPE_DATASET} subset {MODELSCOPE_SUBSET_PATH!r} "
        f"in {snapshot_dir!r}"
    )
print(f"Loading {len(parquet_files)} parquet shards from ModelScope...")
dataset = load_dataset(
    "parquet",
    data_files={"train": parquet_files},
    split="train",
    cache_dir=DATA_DIR,
    num_proc=NUM_PROC,
)

print("Splitting dataset...")
split_dataset = dataset.train_test_split(test_size=0.005, seed=42, shuffle=True)
split_dataset['val'] = split_dataset.pop('test')

enc = tiktoken.get_encoding("gpt2")
eot = enc.eot_token

def process(example):
    ids = enc.encode_ordinary(example['text'])
    ids.append(eot)
    return {'ids': ids, 'len': len(ids)}

print("Tokenizing...")
remove_columns = [column for column in dataset.column_names if column != "text"]
tokenized = split_dataset.map(
    process,
    remove_columns=remove_columns,
    desc="Tokenizing",
    num_proc=NUM_PROC,
)

def save_to_bin(dset, filename, batch_size=100000):
    dtype = np.uint16
    
    dset_np = dset.with_format('numpy')
    token_count = 0
    
    with open(filename, 'wb') as f:
        for i in tqdm(range(0, len(dset), batch_size), desc=f'writing {filename}'):
            arr = np.concatenate(dset_np[i:i+batch_size]['ids']).astype(dtype)
            f.write(arr.tobytes())
            token_count += len(arr)
    
    print(f"Saved {filename}: {token_count:,} tokens")
    
data_dir = DATA_DIR

for split, dset in tokenized.items():
    filename = os.path.join(data_dir, f'{split}.bin')
    save_to_bin(dset, filename)

print("Done!")
print(f"Files saved in: {data_dir}")
print("  - train.bin")
print("  - val.bin")
