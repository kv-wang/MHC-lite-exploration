COMMON_LOG_ARGS="--wandb_log_layer_stats=False --wandb_log_layer_cosine=False --wandb_log_layer_grad_norm=False --wandb_log_layer_activation_norm=False --wandb_log_layer_activation_grad_norm=False --mhc_log_constraint_errors=False"

# fineweb_edu
torchrun --standalone --nproc_per_node=8 train.py config/train_fineweb_edu.py config/small_model.py config/with_hc.py $COMMON_LOG_ARGS 
torchrun --standalone --nproc_per_node=8 train.py config/train_fineweb_edu.py config/small_model.py config/with_mhc_lite.py $COMMON_LOG_ARGS 
torchrun --standalone --nproc_per_node=8 train.py config/train_fineweb_edu.py config/small_model.py config/with_mhc.py $COMMON_LOG_ARGS 
torchrun --standalone --nproc_per_node=8 train.py config/train_fineweb_edu.py config/small_model.py $COMMON_LOG_ARGS  

torchrun --standalone --nproc_per_node=8 train.py config/train_fineweb_edu.py config/medium_model.py config/with_hc.py $COMMON_LOG_ARGS 
torchrun --standalone --nproc_per_node=8 train.py config/train_fineweb_edu.py config/medium_model.py config/with_mhc_lite.py $COMMON_LOG_ARGS 
torchrun --standalone --nproc_per_node=8 train.py config/train_fineweb_edu.py config/medium_model.py config/with_mhc.py $COMMON_LOG_ARGS 
torchrun --standalone --nproc_per_node=8 train.py config/train_fineweb_edu.py config/medium_model.py $COMMON_LOG_ARGS  

torchrun --standalone --nproc_per_node=8 train.py config/train_fineweb_edu.py config/large_model.py config/with_hc.py $COMMON_LOG_ARGS 
torchrun --standalone --nproc_per_node=8 train.py config/train_fineweb_edu.py config/large_model.py config/with_mhc_lite.py $COMMON_LOG_ARGS 
torchrun --standalone --nproc_per_node=8 train.py config/train_fineweb_edu.py config/large_model.py config/with_mhc.py $COMMON_LOG_ARGS 
torchrun --standalone --nproc_per_node=8 train.py config/train_fineweb_edu.py config/large_model.py $COMMON_LOG_ARGS  


# owt
torchrun --standalone --nproc_per_node=8 train.py config/train_owt.py config/small_model.py config/with_hc.py $COMMON_LOG_ARGS 
torchrun --standalone --nproc_per_node=8 train.py config/train_owt.py config/small_model.py config/with_mhc_light.py $COMMON_LOG_ARGS 
torchrun --standalone --nproc_per_node=8 train.py config/train_owt.py config/small_model.py config/with_mhc.py $COMMON_LOG_ARGS 
torchrun --standalone --nproc_per_node=8 train.py config/train_owt.py config/small_model.py $COMMON_LOG_ARGS  

torchrun --standalone --nproc_per_node=8 train.py config/train_owt.py config/medium_model.py config/with_hc.py $COMMON_LOG_ARGS 
torchrun --standalone --nproc_per_node=8 train.py config/train_owt.py config/medium_model.py config/with_mhc_lite.py $COMMON_LOG_ARGS 
torchrun --standalone --nproc_per_node=8 train.py config/train_owt.py config/medium_model.py config/with_mhc.py $COMMON_LOG_ARGS 
torchrun --standalone --nproc_per_node=8 train.py config/train_owt.py config/medium_model.py $COMMON_LOG_ARGS  

torchrun --standalone --nproc_per_node=8 train.py config/train_owt.py config/large_model.py config/with_hc.py $COMMON_LOG_ARGS 
torchrun --standalone --nproc_per_node=8 train.py config/train_owt.py config/large_model.py config/with_mhc_lite.py $COMMON_LOG_ARGS 
torchrun --standalone --nproc_per_node=8 train.py config/train_owt.py config/large_model.py config/with_mhc.py $COMMON_LOG_ARGS 
torchrun --standalone --nproc_per_node=8 train.py config/train_owt.py config/large_model.py $COMMON_LOG_ARGS  




