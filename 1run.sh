torchrun --standalone --nproc_per_node=1 train_gpt2.py \
    --input_bin "data/fineweb10B/fineweb_train_*.bin" \
    --input_val_bin "data/fineweb10B/fineweb_val_*.bin" \
    --model d12 \
    --batch_size 32 \
    --accumulation 1 \
    --sequence_length 1024 \
    --val_loss_every 125 \
    --val_max_steps 5 \
    --num_iterations 4000 \
    --weight_decay 0.0 \
    --learning_rate 0.0024 \
    --warmup_iters 500 \
    --warmdown_iters 1000