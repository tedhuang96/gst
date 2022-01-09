temporal=lstm
decode_style=recursive
ghost=--ghost
spatial_num_heads_edges=1
only_observe_full_period=

num_epochs=200
spatial=gumbel_social_transformer
detach_sample=--detach_sample
rotation_pattern=random
lr=1e-3
spatial_num_heads=8
spatial_num_layers=3
embedding_size=32
deterministic=--deterministic
lstm_hidden_size=32
batch_size=16
init_temp=0.5


for random_seed in 1000; do
    for dataset in eth hotel univ zara1 zara2; do
        python -u scripts/eval.py --spatial $spatial --temporal $temporal --lr $lr\
            --dataset $dataset --num_epochs $num_epochs\
            $detach_sample --rotation_pattern $rotation_pattern\
            --spatial_num_heads $spatial_num_heads --spatial_num_layers $spatial_num_layers\
            --decode_style $decode_style \
            --spatial_num_heads_edges $spatial_num_heads_edges --random_seed $random_seed $ghost\
            $deterministic --lstm_hidden_size $lstm_hidden_size\
            --batch_size $batch_size \
            --init_temp $init_temp $only_observe_full_period\
            | tee -a logs/eval_sparse.txt
    done
done