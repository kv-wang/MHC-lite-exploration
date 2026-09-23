# idt_4_small per-stream outputs before reduce_stream merge

- checkpoint: `idt_4_small/ckpt.pt`
- seed: `1337`, batch_size: `1`, block_size: `1024`
- positions: `[1019, 1020, 1021, 1022, 1023]`
- printed output: top-10 lm_head logits per stream; hidden vectors include first 16 dims in JSON

## position 1019 input=19253 ` locks` target=287 ` in`
- max pairwise logit RMSE at this position: `2.12943`

- stream 0: hidden_l2=23.0586; top5 logits: 1:986 `...` 3.46, 2:12359 `.....` 3.318, 3:16317 `......` 3.134, 4:1106 `....` 3.063, 5:93 `~` 3.053
- stream 1: hidden_l2=23.2536; top5 logits: 1:11 `,` 4.311, 2:13 `.` 4.029, 3:290 ` and` 3.889, 4:319 ` on` 3.858, 5:287 ` in` 3.843
- stream 2: hidden_l2=22.9705; top5 logits: 1:0 `!` 3.182, 2:13 `.` 2.68, 3:329 ` for` 2.585, 4:11 `,` 2.535, 5:290 ` and` 2.514
- stream 3: hidden_l2=23.2209; top5 logits: 1:11 `,` 3.069, 2:13 `.` 2.841, 3:379 ` at` 2.802, 4:357 ` (` 2.751, 5:986 `...` 2.717

## position 1020 input=287 ` in` target=790 ` every`
- max pairwise logit RMSE at this position: `1.80759`

- stream 0: hidden_l2=23.1158; top5 logits: 1:46149 ` crochet` 3.289, 2:25749 ` wool` 3.146, 3:24759 ` Wool` 2.859, 4:21181 ` yarn` 2.821, 5:607 ` her` 2.739
- stream 1: hidden_l2=23.4783; top5 logits: 1:262 ` the` 4.156, 2:1339 ` case` 3.699, 3:257 ` a` 3.672, 4:2166 ` front` 3.376, 5:428 ` this` 3.299
- stream 2: hidden_l2=23.1739; top5 logits: 1:1021 ` hand` 2.442, 2:262 ` the` 2.336, 3:257 ` a` 2.249, 4:530 ` one` 2.089, 5:340 ` it` 2.061
- stream 3: hidden_l2=23.1357; top5 logits: 1:262 ` the` 2.397, 2:257 ` a` 2.291, 3:477 ` all` 2.245, 4:530 ` one` 2.033, 5:428 ` this` 1.861

## position 1021 input=790 ` every` target=3124 ` color`
- max pairwise logit RMSE at this position: `1.54049`

- stream 0: hidden_l2=23.1712; top5 logits: 1:18600 ` booth` 3.644, 2:6131 ` bag` 3.529, 3:11668 ` bags` 3.218, 4:32263 ` sock` 3.117, 5:3650 ` store` 3.099
- stream 1: hidden_l2=23.3394; top5 logits: 1:2060 ` single` 4.077, 2:5228 ` corner` 3.758, 3:1110 ` day` 3.562, 4:1744 ` possible` 3.531, 5:530 ` one` 3.483
- stream 2: hidden_l2=23.1931; top5 logits: 1:3650 ` store` 2.403, 2:33451 ` drawer` 2.354, 3:6131 ` bag` 2.334, 4:18600 ` booth` 2.324, 5:2665 ` section` 2.287
- stream 3: hidden_l2=23.0928; top5 logits: 1:530 ` one` 2.07, 2:835 ` way` 1.683, 3:1263 ` big` 1.672, 4:286 ` of` 1.663, 5:2060 ` single` 1.572

## position 1022 input=3124 ` color` target=739 ` under`
- max pairwise logit RMSE at this position: `2.40295`

- stream 0: hidden_l2=23.0473; top5 logits: 1:27043 ` palette` 4.143, 2:33988 ` coloring` 3.593, 3:3124 ` color` 3.274, 4:30121 ` dye` 3.198, 5:7577 ` colors` 3.057
- stream 1: hidden_l2=23.3365; top5 logits: 1:290 ` and` 3.97, 2:286 ` of` 3.874, 3:11 `,` 3.667, 4:356 ` we` 3.57, 5:13 `.` 3.512
- stream 2: hidden_l2=23.1101; top5 logits: 1:40758 ` imaginable` 3.368, 2:0 `!` 3.06, 3:13 `.` 2.747, 4:27043 ` palette` 2.643, 5:1695 ` available` 2.531
- stream 3: hidden_l2=23.2635; top5 logits: 1:13 `.` 2.843, 2:290 ` and` 2.787, 3:11 `,` 2.745, 4:532 ` -` 2.734, 5:986 `...` 2.604

## position 1023 input=739 ` under` target=262 ` the`
- max pairwise logit RMSE at this position: `1.44725`

- stream 0: hidden_l2=22.9629; top5 logits: 1:17979 ` shade` 2.67, 2:30720 ` coats` 2.524, 3:41822 ` linen` 2.493, 4:3124 ` color` 2.488, 5:34794 ` blankets` 2.426
- stream 1: hidden_l2=23.4109; top5 logits: 1:262 ` the` 4.205, 2:616 ` my` 3.672, 3:674 ` our` 3.615, 4:27521 ` wraps` 3.379, 5:465 ` his` 3.321
- stream 2: hidden_l2=23.0725; top5 logits: 1:607 ` her` 2.724, 2:262 ` the` 2.666, 3:616 ` my` 2.656, 4:1110 ` day` 2.555, 5:720 ` $` 2.376
- stream 3: hidden_l2=23.163; top5 logits: 1:262 ` the` 2.444, 2:257 ` a` 2.237, 3:530 ` one` 2.017, 4:607 ` her` 2.007, 5:477 ` all` 1.967
