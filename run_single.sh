poetry run python app/federated_learning.py \
    --clip 0.1 \
    --noise-multiplier 12.74 \
    --num-rounds 100 \
    --embed-grads \
    --num-clients-public 20 \
    --embedding-num-bases 20