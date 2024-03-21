python3 train.py \
	--do_train \
	--epoch 5 \
	--train_batch_size 32 \
	--eval_batch_size 64 \
	--learning_rate 2e-5 \
	--max_grad_norm 1.0 \
	--evaluate_during_training \
	--seed 123456 \
	--alpha 1. \
	--layers even_idx \
	--model_name microsoft/unixcoder-base \
# layers: even_idx, odd_idx, all, none