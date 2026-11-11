# Matched iHC vs ITO-HC stream differentiation

This diagnostic compares two available checkpoints with the same small GPT
configuration (6 layers, width 512), 4 streams, 10,000 training iterations,
seed 1337, repeated expansion, and `4mean` output reduction:

- iHC: `out-owt-small-mhc-identity-static-pre-post-4streams-reduce-4mean-10000iter/ckpt.pt`
- ITO-HC: `out-owt-small-mhc-identity-tanh-offdiag-static-static-4streams-reduce-4mean-10000iter/ckpt.pt`

Both checkpoints disable dynamic pre/post routers. ITO-HC also disables the
dynamic residual generator and uses fixed `gamma=0.05`. This is therefore a
matched comparison of static-router iHC and static-router ITO-HC, not the
paper's main dynamic-router setting.

The evaluation used the same seeded validation batches for each model: 40
batches of 2 sequences, with 128 sampled token positions per batch (5,120
positions per layer). At each layer, the final LayerNorm and language-model
head were applied to each stream separately (logit lens). Across the six
layers, mean pairwise stream-logit RMSE was 0.5718 for iHC and 0.6914 for
ITO-HC; mean pairwise cosine was 0.9062 vs. 0.8624; centered cosine was
0.8375 vs. 0.7936; and top-1 prediction agreement was 0.3778 vs. 0.3657.
At the final layer, RMSE was 0.6049 vs. 0.7580, cosine 0.8922 vs. 0.8267,
centered cosine 0.8772 vs. 0.8036, and top-1 agreement 0.3833 vs. 0.2746.
These metrics indicate that the ITO-HC streams are more functionally distinct
at later layers under several output-space measures. The effect is not uniform:
at layer 0 the ITO-HC streams are more similar by RMSE and top-1 agreement.

For the ITO-HC checkpoint, reconstructing the 12 residual matrices gives a
product `P = H_11 ... H_0` whose 12 off-diagonal entries contain 10 negatives.
The product singular values are approximately
`[1.2220, 1.1608, 0.9955, 0.6977]` (condition number 1.75). This confirms
signed residual transport in this checkpoint, but a negative coefficient does
not by itself prove semantic subtraction or useful information specialization.

The checkpoints' recorded best validation losses are 3.4059 (iHC) and 3.4002
(ITO-HC), but these are single runs. Stream-logit diversity is evidence of
different per-stream predictions, not evidence that the streams encode more
complementary information or cause a reliable language-model improvement.

Detailed per-layer outputs are in the adjacent `*_layer_summary.csv`,
`*_pair_metrics.csv`, and `*_stream_losses.csv` files. Results were generated
with `analyze/eval_layer_stream_logits_similarity.py`.
