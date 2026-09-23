# Effective H_res matrices: identity_tanh_offidiag/ckpt.pt

```text
checkpoint_path: /root/autodl-tmp/MHC-lite-exploration/identity_tanh_offidiag/ckpt.pt
iter_num: 10000
best_val_loss: 3.3676745891571045
hyper_conn_type: mhc
mhc_h_res_mode: identity_tanh_offdiag
hyper_conn_n: 4
hyper_conn_reduce_stream_mode: 4mean
mhc_h_res_offdiag_init_scale: 0.05
out_prefix_method: mhc-identity-tanh-offdiag-4streams-reduce-4mean-gamma0p05-10000iter
num_static_alpha: 12

restore_formula: H_res = I + gamma * ((1 1^T - I) o tanh(W))
where W = static_alpha[:, num_input_views:], num_input_views = static_alpha.shape[1] - static_alpha.shape[0]
gamma = exp(<module>.h_res_offdiag_log_scale); if missing, fallback to config mhc_h_res_offdiag_init_scale

transformer.h.0.hc_attn.static_alpha
static_alpha_shape: (4, 5), W_shape: (4, 4), h_res_shape: (4, 4)
num_input_views: 1, gamma: 0.0842361674, log_scale: -2.4741308689, scale_source: transformer.h.0.hc_attn.h_res_offdiag_log_scale
H_res =
[ 1.00000000,  0.03126202,  0.00199235,  0.00743684 ]
[ -0.02955870,  1.00000000,  0.00199235,  0.00743684 ]
[ -0.02955870,  0.03126202,  1.00000000,  0.00743684 ]
[ -0.02955870,  0.03126202,  0.00199235,  1.00000000 ]
row_sums = [ 1.04069126,  0.97987044,  1.00914013,  1.00369561 ]
col_sums = [ 0.91132385,  1.09378612,  1.00597703,  1.02231050 ]
offdiag_abs_max: 0.0312620215, offdiag_abs_mean: 0.0175624769

transformer.h.0.hc_mlp.static_alpha
static_alpha_shape: (4, 5), W_shape: (4, 4), h_res_shape: (4, 4)
num_input_views: 1, gamma: 0.4461074769, log_scale: -0.8071953654, scale_source: transformer.h.0.hc_mlp.h_res_offdiag_log_scale
H_res =
[ 1.00000000,  -0.27733961,  -0.37075034,  -0.08658758 ]
[ -0.43497807,  1.00000000,  -0.43203020,  -0.14135689 ]
[ -0.40698338,  -0.29890034,  1.00000000,  -0.13722903 ]
[ 0.08536069,  -0.04524889,  0.04620652,  1.00000000 ]
row_sums = [ 0.26532251,  -0.00836515,  0.15688723,  1.08631837 ]
col_sums = [ 0.24339925,  0.37851119,  0.24342601,  0.63482654 ]
offdiag_abs_max: 0.4349780679, offdiag_abs_mean: 0.2302476317

transformer.h.1.hc_attn.static_alpha
static_alpha_shape: (4, 5), W_shape: (4, 4), h_res_shape: (4, 4)
num_input_views: 1, gamma: 0.0975626558, log_scale: -2.3272604942, scale_source: transformer.h.1.hc_attn.h_res_offdiag_log_scale
H_res =
[ 1.00000000,  -0.01769595,  -0.05592721,  -0.04074458 ]
[ -0.05215343,  1.00000000,  -0.02364867,  0.01133790 ]
[ -0.05189173,  -0.00453576,  1.00000000,  -0.00711321 ]
[ -0.02839975,  0.00813788,  0.00071317,  1.00000000 ]
row_sums = [ 0.88563222,  0.93553579,  0.93645930,  0.98045129 ]
col_sums = [ 0.86755508,  0.98590618,  0.92113727,  0.96348011 ]
offdiag_abs_max: 0.0559272096, offdiag_abs_mean: 0.0251916032

transformer.h.1.hc_mlp.static_alpha
static_alpha_shape: (4, 5), W_shape: (4, 4), h_res_shape: (4, 4)
num_input_views: 1, gamma: 0.1018540636, log_scale: -2.2842142582, scale_source: transformer.h.1.hc_mlp.h_res_offdiag_log_scale
H_res =
[ 1.00000000,  -0.05523200,  -0.04321521,  -0.01973300 ]
[ -0.05359991,  1.00000000,  -0.03307929,  0.01808051 ]
[ -0.05280624,  -0.01603533,  1.00000000,  -0.00690603 ]
[ -0.03881958,  -0.02543952,  -0.01802117,  1.00000000 ]
row_sums = [ 0.88181978,  0.93140137,  0.92425239,  0.91771972 ]
col_sums = [ 0.85477424,  0.90329319,  0.90568435,  0.99144149 ]
offdiag_abs_max: 0.0552320033, offdiag_abs_mean: 0.0317473151

transformer.h.2.hc_attn.static_alpha
static_alpha_shape: (4, 5), W_shape: (4, 4), h_res_shape: (4, 4)
num_input_views: 1, gamma: 0.1256906986, log_scale: -2.0739312172, scale_source: transformer.h.2.hc_attn.h_res_offdiag_log_scale
H_res =
[ 1.00000000,  -0.07608568,  -0.10361512,  -0.08050511 ]
[ -0.04442742,  1.00000000,  -0.01770047,  0.07839055 ]
[ -0.04267226,  -0.03335043,  1.00000000,  -0.00296053 ]
[ -0.06595662,  -0.03034991,  -0.03132829,  1.00000000 ]
row_sums = [ 0.73979408,  1.01626265,  0.92101681,  0.87236518 ]
col_sums = [ 0.84694374,  0.86021399,  0.84735614,  0.99492490 ]
offdiag_abs_max: 0.1036151201, offdiag_abs_mean: 0.0506118685

transformer.h.2.hc_mlp.static_alpha
static_alpha_shape: (4, 5), W_shape: (4, 4), h_res_shape: (4, 4)
num_input_views: 1, gamma: 0.1090710834, log_scale: -2.2157554626, scale_source: transformer.h.2.hc_mlp.h_res_offdiag_log_scale
H_res =
[ 1.00000000,  -0.03349564,  -0.05033276,  -0.03813053 ]
[ -0.04244567,  1.00000000,  -0.00871558,  0.07256928 ]
[ -0.04436517,  -0.03471073,  1.00000000,  0.00324374 ]
[ -0.04499238,  -0.02186021,  -0.01292451,  1.00000000 ]
row_sums = [ 0.87804103,  1.02140808,  0.92416787,  0.92022288 ]
col_sums = [ 0.86819679,  0.90993339,  0.92802715,  1.03768253 ]
offdiag_abs_max: 0.0725692809, offdiag_abs_mean: 0.0339821838

transformer.h.3.hc_attn.static_alpha
static_alpha_shape: (4, 5), W_shape: (4, 4), h_res_shape: (4, 4)
num_input_views: 1, gamma: 0.1001332700, log_scale: -2.3012533188, scale_source: transformer.h.3.hc_attn.h_res_offdiag_log_scale
H_res =
[ 1.00000000,  -0.04265926,  -0.05098805,  -0.04080176 ]
[ -0.02853590,  1.00000000,  -0.01091530,  0.03583485 ]
[ -0.03420972,  -0.02626441,  1.00000000,  0.01121212 ]
[ -0.05293741,  -0.02445391,  -0.01528144,  1.00000000 ]
row_sums = [ 0.86555088,  0.99638367,  0.95073795,  0.90732723 ]
col_sums = [ 0.88431698,  0.90662235,  0.92281520,  1.00624526 ]
offdiag_abs_max: 0.0529374145, offdiag_abs_mean: 0.0311745107

transformer.h.3.hc_mlp.static_alpha
static_alpha_shape: (4, 5), W_shape: (4, 4), h_res_shape: (4, 4)
num_input_views: 1, gamma: 0.0990424007, log_scale: -2.3122072220, scale_source: transformer.h.3.hc_mlp.h_res_offdiag_log_scale
H_res =
[ 1.00000000,  -0.03501375,  -0.05955811,  -0.03412143 ]
[ -0.02313368,  1.00000000,  -0.04531566,  0.01113270 ]
[ -0.03177782,  -0.02533001,  1.00000000,  0.01100555 ]
[ -0.04911143,  -0.02198700,  -0.01382749,  1.00000000 ]
row_sums = [ 0.87130672,  0.94268334,  0.95389771,  0.91507411 ]
col_sums = [ 0.89597708,  0.91766924,  0.88129872,  0.98801684 ]
offdiag_abs_max: 0.0595581084, offdiag_abs_mean: 0.0301095489

transformer.h.4.hc_attn.static_alpha
static_alpha_shape: (4, 5), W_shape: (4, 4), h_res_shape: (4, 4)
num_input_views: 1, gamma: 0.0940511152, log_scale: -2.3639168739, scale_source: transformer.h.4.hc_attn.h_res_offdiag_log_scale
H_res =
[ 1.00000000,  -0.03509057,  -0.05555235,  -0.03998959 ]
[ -0.01506433,  1.00000000,  -0.03379816,  -0.00203350 ]
[ -0.01724853,  -0.01316854,  1.00000000,  -0.01220977 ]
[ -0.04922399,  -0.02070710,  -0.01007721,  1.00000000 ]
row_sums = [ 0.86936748,  0.94910395,  0.95737314,  0.91999173 ]
col_sums = [ 0.91846311,  0.93103385,  0.90057230,  0.94576716 ]
offdiag_abs_max: 0.0555523485, offdiag_abs_mean: 0.0253469720

transformer.h.4.hc_mlp.static_alpha
static_alpha_shape: (4, 5), W_shape: (4, 4), h_res_shape: (4, 4)
num_input_views: 1, gamma: 0.0846862569, log_scale: -2.4688019753, scale_source: transformer.h.4.hc_mlp.h_res_offdiag_log_scale
H_res =
[ 1.00000000,  -0.02218764,  -0.01836056,  -0.03085230 ]
[ -0.01082560,  1.00000000,  -0.02589499,  -0.00585353 ]
[ -0.01314608,  -0.01359771,  1.00000000,  -0.01610544 ]
[ -0.04461229,  -0.01767250,  -0.00315805,  1.00000000 ]
row_sums = [ 0.92859948,  0.95742589,  0.95715082,  0.93455720 ]
col_sums = [ 0.93141603,  0.94654214,  0.95258641,  0.94718874 ]
offdiag_abs_max: 0.0446122885, offdiag_abs_mean: 0.0185222253

transformer.h.5.hc_attn.static_alpha
static_alpha_shape: (4, 5), W_shape: (4, 4), h_res_shape: (4, 4)
num_input_views: 1, gamma: 0.0738418624, log_scale: -2.6058294773, scale_source: transformer.h.5.hc_attn.h_res_offdiag_log_scale
H_res =
[ 1.00000000,  -0.00399232,  -0.00876339,  0.01606280 ]
[ -0.00697183,  1.00000000,  -0.02080263,  -0.00050449 ]
[ -0.00798240,  0.00267067,  1.00000000,  0.00520668 ]
[ -0.03209295,  -0.01562037,  0.00160333,  1.00000000 ]
row_sums = [ 1.00330710,  0.97172105,  0.99989498,  0.95389003 ]
col_sums = [ 0.95295286,  0.98305798,  0.97203732,  1.02076495 ]
offdiag_abs_max: 0.0320929550, offdiag_abs_mean: 0.0101894904

transformer.h.5.hc_mlp.static_alpha
static_alpha_shape: (4, 5), W_shape: (4, 4), h_res_shape: (4, 4)
num_input_views: 1, gamma: 0.1162172034, log_scale: -2.1522943974, scale_source: transformer.h.5.hc_mlp.h_res_offdiag_log_scale
H_res =
[ 1.00000000,  -0.04447244,  -0.05382726,  0.04345136 ]
[ -0.01038319,  1.00000000,  -0.02604738,  -0.00358807 ]
[ -0.01193302,  0.00634502,  1.00000000,  0.00576267 ]
[ -0.03811081,  -0.07562481,  -0.08633793,  1.00000000 ]
row_sums = [ 0.94515163,  0.95998138,  1.00017464,  0.79992646 ]
col_sums = [ 0.93957299,  0.88624775,  0.83378744,  1.04562593 ]
offdiag_abs_max: 0.0863379315, offdiag_abs_mean: 0.0338236652

```
