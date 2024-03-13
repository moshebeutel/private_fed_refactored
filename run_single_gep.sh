poetry run python app/federated_learning.py \
    --classes_per_user 10 \
    --clip 0.1 \
    --noise-multiplier 0.0 \
    --num-rounds 100 \
    --embed-grads \
    --num-clients-public 50 \
    --embedding-num-bases 10