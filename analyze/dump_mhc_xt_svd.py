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
    model = GPT(GPTConfig(**checkpoint["model_args"]))

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


def to_xt_tensor(x, num_streams):
    if x.shape[0] % num_streams != 0:
        raise ValueError(f"batch-stream axis {x.shape[0]} is not divisible by {num_streams}")
    batch_size = x.shape[0] // num_streams
    seq_len = x.shape[1]
    hidden_dim = x.shape[-1]
    return x.detach().float().reshape(batch_size, num_streams, seq_len, hidden_dim).permute(0, 2, 1, 3)


def normalize_position(position, seq_len):
    if position < 0:
        position = seq_len + position
    if not 0 <= position < seq_len:
        raise ValueError(f"position must be in [0, {seq_len - 1}] or negative-index equivalent")
    return position


def main():
    parser = argparse.ArgumentParser(description="Dump SVD of the MHC formula state X_t in R^{k x d}.")
    parser.add_argument("--ckpt", default="idt_4_medium/ckpt.pt")
    parser.add_argument("--data", default="data/openwebtext/val.bin")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=None)
    parser.add_argument("--batch-index", type=int, default=0)
    parser.add_argument("--position", type=int, default=-1, help="Token position whose k x d X_t is printed.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=None, help="Optional JSON path for selected-position singular values.")
    parser.add_argument("--all-positions-out", default=None, help="Optional JSON path for every token position.")
    args = parser.parse_args()

    device = torch.device(args.device)
    model, checkpoint = load_model(args.ckpt, device)
    cfg = model.config
    seq_len = args.seq_len or cfg.block_size
    if seq_len > cfg.block_size:
        raise ValueError(f"seq-len {seq_len} exceeds model block_size {cfg.block_size}")
    if not 0 <= args.batch_index < args.batch_size:
        raise ValueError(f"batch-index must be in [0, {args.batch_size - 1}]")

    position = normalize_position(args.position, seq_len)
    x_in = make_input(args.data, seq_len, args.batch_size, device)
    num_streams = getattr(cfg, "hyper_conn_n", 1)

    selected_results = []
    all_position_results = []

    def make_hook(layer_idx, step_name):
        def hook(_module, _inputs, output):
            xt = to_xt_tensor(output, num_streams)
            mat = xt[args.batch_index, position]
            selected_s = torch.linalg.svdvals(mat).cpu().tolist()
            selected_results.append({
                "layer": layer_idx,
                "step": step_name,
                "batch_index": args.batch_index,
                "position": position,
                "raw_output_shape": list(output.shape),
                "xt_shape": list(mat.shape),
                "singular_values": selected_s,
            })

            if args.all_positions_out:
                all_s = torch.linalg.svdvals(xt.reshape(-1, num_streams, xt.shape[-1]))
                all_s = all_s.reshape(xt.shape[0], xt.shape[1], num_streams).cpu().tolist()
                all_position_results.append({
                    "layer": layer_idx,
                    "step": step_name,
                    "xt_batch_seq_shape": list(xt.shape),
                    "singular_values_by_batch_position": all_s,
                })
        return hook

    handles = []
    for layer_idx, block in enumerate(model.transformer.h):
        handles.append(block.hc_attn.register_forward_hook(make_hook(layer_idx, "attn")))
        handles.append(block.hc_mlp.register_forward_hook(make_hook(layer_idx, "mlp")))

    try:
        with torch.no_grad():
            model(x_in)
    finally:
        for handle in handles:
            handle.remove()

    selected_results.sort(key=lambda item: (item["layer"], item["step"] != "attn"))
    all_position_results.sort(key=lambda item: (item["layer"], item["step"] != "attn"))

    metadata = {
        "ckpt": args.ckpt,
        "data": args.data,
        "batch_size": args.batch_size,
        "seq_len": seq_len,
        "batch_index": args.batch_index,
        "position": position,
        "num_streams": num_streams,
        "hidden_dim": cfg.n_embd,
        "formula": "X_t in R^{k x d}; k=num_streams, d=hidden_dim",
        "model_args": checkpoint["model_args"],
        "results": selected_results,
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        print(f"wrote {out_path}")

    if args.all_positions_out:
        all_out_path = Path(args.all_positions_out)
        all_out_path.parent.mkdir(parents=True, exist_ok=True)
        all_metadata = dict(metadata)
        all_metadata["results"] = all_position_results
        all_out_path.write_text(json.dumps(all_metadata, indent=2), encoding="utf-8")
        print(f"wrote {all_out_path}")

    print(f"ckpt={args.ckpt}")
    print(f"input={args.data}, batch_size={args.batch_size}, seq_len={seq_len}")
    print(f"selected X_t: batch={args.batch_index}, position={position}, shape=[{num_streams}, {cfg.n_embd}]")
    for item in selected_results:
        values = ", ".join(f"{v:.9g}" for v in item["singular_values"])
        print(f"layer {item['layer']:02d} {item['step']} X_t singular_values=[{values}]")


if __name__ == "__main__":
    main()
