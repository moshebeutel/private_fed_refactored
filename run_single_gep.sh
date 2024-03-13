poetry run python app/federated_learning.py \
    --classes_per_user 2 \
    --clip 0.001 \
    --noise-multiplier 12.79182 \
    --num-rounds 10 \
    --embed-grads \
    --num-clients-public 100 \
    --embedding-num-bases 90