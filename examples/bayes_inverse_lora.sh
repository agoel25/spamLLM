uv run -m examples.bayes_inverse_lora \
    --batch_size 4 \
    --max_seq_len 512 \
    --num_epochs 30 \
    --learning_rate 5e-4 \
    --lora_r 4 \
    --lora_alpha 8