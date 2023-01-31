#seq_len = [1, 5, 10, 15, 25, 30, 60, 90, 180]
ctx_len=90
model="transformer"
#model="lstm"
#model="mlp"
layers=12
data_path=./data/
ckpt_path=./ckpt/$model"_LL"$layers"_ctx_"$ctx_len"_bs32"
rm -rf $ckpt_path
mkdir -p $ckpt_path

CUDA_VISIBLE_DEVICES=6 \
python3 train.py  --model=$model \
                  --dropout 0.1 \
                  --embedding_dim 14 \
                  --hidden_dim 512 \
                  --layers $layers \
                  --num_class 4 \
                  --ctx_len $ctx_len \
                  --train_data $data_path/train_seq_$ctx_len.pk \
                  --dev_data $data_path/dev_seq_$ctx_len.pk \
                  --test_data $data_path/test_seq_$ctx_len.pk \
                  --batch_size 32 \
                  --lr 1e-5 \
                  --epoch 150 \
                  --gpuid 0 \
                  --print_every 100 \
                  --save_every 10000 \
                  --save_dir $ckpt_path \
