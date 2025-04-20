model_type='LightGTS_4M_quantiles'
model_name='_monash_decoder_2task_wo_img_256_528_48_rope_quantiles'
dset_path='/home/Decoder_version_1/data/pretrain_datasets/monash_csv_downsmp'
context_points=528
target_points=96
e_layers=3
d_layers=3
d_model=256
d_ff=512
mask_mode='freq_multi'
n_epochs_pretrain=50
img_size=64

if [ ! -d "logs" ]; then
    mkdir logs
fi

if [ ! -d "logs/Pretrain" ]; then
    mkdir logs/Pretrain
fi
if [ ! -d "logs/Pretrain/$model_type" ]; then
    mkdir logs/Pretrain/$model_type
fi
if [ ! -d "logs/Pretrain/$model_type/$model_name" ]; then
    mkdir logs/Pretrain/$model_type/$model_name
fi


python -u pretrain_quantile.py \
    --context_points $context_points \
    --target_points $target_points \
    --batch_size 8192\
    --dset_path $dset_path \
    --num_workers 8\
    --features M\
    --patch_len 48\
    --stride 48\
    --revin 1 \
    --e_layers $e_layers\
    --d_layers $d_layers\
    --n_heads 8 \
    --d_model $d_model \
    --img_size $img_size \
    --d_ff $d_ff\
    --dropout 0.2\
    --head_drop 0.2 \
    --mask_ratio 0.3\
    --mask_mode $mask_mode\
    --mask_nums 4\
    --n_epochs_pretrain $n_epochs_pretrain\
    --pretrained_model_id 1\
    --lr 1e-4 \
    --model_type $model_type\
    --is_half 0\
    --is_all 1\
    --model_name $model_name\
    --one_channel 0\
    --is_checkpoints True\
    --checkpoints_freq 1 >logs/Pretrain/$model_type/$model_name/'input_length_'$context_points'_pred_len_'$target_points'_e_layer_'$e_layers'_d_layers_'$d_layers'_d_model_'$d_model'_d_ff_'$d_ff'_'$mask_mode.log 






