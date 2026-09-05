import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from model import GPT, GPTConfig


def load_model(ckpt_path, device):
    checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
    model_args = checkpoint["model_args"]
    model = GPT(GPTConfig(**model_args))

    state_dict = checkpoint["model"]
    unwanted_prefix = "_orig_mod."
    for key in list(state_dict.keys()):
        if key.startswith(unwanted_prefix):
            state_dict[key[len(unwanted_prefix):]] = state_dict.pop(key)

    model.load_state_dict(state_dict)
    model.eval()
    model.to(device)
    return model, checkpoint


def make_input(data_path, seq_len, batch_size, device):
    data = np.memmap(data_path, dtype=np.uint16, mode="r")
    needed = batch_size * seq_len
    if len(data) < needed:
        raise ValueError(f"{data_path} has only {len(data)} tokens, need {needed}")

    tokens = np.asarray(data[:needed], dtype=np.int64).reshape(batch_size, seq_len)
    return torch.from_numpy(tokens).to(device)


def layer_matrix(x, num_streams):
    if num_streams > 1:
        if x.shape[0] % num_streams != 0:
            raise ValueError(f"batch-stream axis {x.shape[0]} is not divisible by {num_streams}")
        batch = x.shape[0] // num_streams
        seq_len = x.shape[1]
        hidden = x.shape[-1]
        x = x.reshape(batch, num_streams, seq_len, hidden).permute(0, 2, 1, 3)
        return x.reshape(batch * seq_len, num_streams * hidden)

    return x.reshape(-1, x.shape[-1])


def main():
    parser = argparse.ArgumentParser(description="Dump singular values of per-layer X_L activations.")
    parser.add_argument("--ckpt", default="idt_4_medium/ckpt.pt")
    parser.add_argument("--data", default="data/openwebtext/val.bin")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=None, help="Optional JSON path for full singular values.")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, checkpoint = load_model(args.ckpt, device)
    cfg = model.config
    seq_len = args.seq_len or cfg.block_size
    if seq_len > cfg.block_size:
        raise ValueError(f"seq-len {seq_len} exceeds model block_size {cfg.block_size}")

    x_in = make_input(args.data, seq_len, args.batch_size, device)
    num_streams = getattr(cfg, "hyper_conn_n", 1)
    results = []

    def make_hook(layer_idx):
        def hook(_module, _inputs, output):
            mat = layer_matrix(output.detach().float(), num_streams)
            s = torch.linalg.svdvals(mat).cpu().tolist()
            results.append({
                "layer": layer_idx,
                "x_shape": list(output.shape),
                "matrix_shape": list(mat.shape),
                "singular_values": s,
            })
        return hook

    handles = [block.register_forward_hook(make_hook(i)) for i, block in enumerate(model.transformer.h)]
    try:
        with torch.no_grad():
            model(x_in)
    finally:
        for handle in handles:
            handle.remove()

    results.sort(key=lambda item: item["layer"])
    metadata = {
        "ckpt": args.ckpt,
        "data": args.data,
        "batch_size": args.batch_size,
        "seq_len": seq_len,
        "num_streams": num_streams,
        "model_args": checkpoint["model_args"],
        "layers": results,
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"wrote {out_path}")

    print(f"ckpt={args.ckpt}")
    print(f"input={args.data}, batch_size={args.batch_size}, seq_len={seq_len}, num_streams={num_streams}")
    for item in results:
        values = ", ".join(f"{v:.9g}" for v in item["singular_values"])
        print(f"layer {item['layer']:02d} X_L shape={item['x_shape']} matrix={item['matrix_shape']}")
        print(f"singular_values=[{values}]")


if __name__ == "__main__":
    main()
