import os
import numpy as np
import tiktoken
from tqdm import tqdm

DATA_DIR = "/root/autodl-tmp/mhc-lite/data/fineweb_edu"

# Keep the Hugging Face dataset cache under the project data directory.
os.environ["HF_HOME"] = DATA_DIR
os.environ["HF_DATASETS_CACHE"] = DATA_DIR

from datasets import load_dataset

NUM_PROC = 8  
SHARD_SIZE = int(1e8)

print("Loading FineWeb-Edu dataset...")
dataset = load_dataset(
    "HuggingFaceFW/fineweb-edu",
    name="sample-10BT",
    split="train",
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
tokenized = split_dataset.map(
    process,
    remove_columns=['text', 'id', 'dump', 'url', 'file_path', 'language', 
                    'language_score', 'token_count', 'score', 'int_score'],
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