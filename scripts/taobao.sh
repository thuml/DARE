# Explain of the input parameters can also be found at the end of the README.md
# DARE
CUDA_VISIBLE_DEVICES=4 python train_pytorch.py -train_dataset_path 'data/taobao/train_{}_{}.npy' -val_dataset_path 'data/taobao/val.npy' -test_dataset_path 'data/taobao/test.npy' -max_length 200 -item_n 4068791 -cate_n 9408 -long_seq_split '0:200' -short_model_type DIN -long_model_type DIN -use_cross_feature True -attn_func scaled_dot_product -hard_or_soft soft -top_k 20 -use_aux_loss True -use_time_mode concat -time_embedding_dim 16 -epoch 2 -category_embedding_dim 16 -batch_size 2048 -learning_rate 0.01 -weight_decay 0.000001 -seed 1 2 3 4 5 -log_dir log/taobao/model_DARE -test_interval 150 -log_interval 50 -model_name DARE

# TWIN
CUDA_VISIBLE_DEVICES=4 python train_pytorch.py -train_dataset_path 'data/taobao/train_{}_{}.npy' -val_dataset_path 'data/taobao/val.npy' -test_dataset_path 'data/taobao/test.npy' -max_length 200 -item_n 4068791 -cate_n 9408 -long_seq_split '0:200' -short_model_type DIN -long_model_type DIN -use_cross_feature True -attn_func scaled_dot_product -hard_or_soft soft -top_k 20 -use_aux_loss True -use_time_mode concat -time_embedding_dim 16 -epoch 2 -category_embedding_dim 16 -batch_size 2048 -learning_rate 0.01 -weight_decay 0.000001 -seed 1 2 3 4 5 -log_dir log/taobao/model_twin -test_interval 150 -log_interval 50 -model_name twin

# DIN
CUDA_VISIBLE_DEVICES=4 python train_pytorch.py -train_dataset_path 'data/taobao/train_{}_{}.npy' -val_dataset_path 'data/taobao/val.npy' -test_dataset_path 'data/taobao/test.npy' -max_length 200 -item_n 4068791 -cate_n 9408 -long_seq_split '0:200' -short_model_type DIN -long_model_type DIN -use_cross_feature True -attn_func learnable -hard_or_soft soft -top_k 20 -use_aux_loss True -use_time_mode concat -time_embedding_dim 16 -epoch 2 -category_embedding_dim 16 -batch_size 2048 -learning_rate 0.01 -weight_decay 0.000001 -seed 1 2 3 4 5 -log_dir log/taobao/model_DIN -test_interval 150 -log_interval 50 -model_name DIN -use_time

# TWIN-4E
CUDA_VISIBLE_DEVICES=4 python train_pytorch.py -train_dataset_path 'data/taobao/train_{}_{}.npy' -val_dataset_path 'data/taobao/val.npy' -test_dataset_path 'data/taobao/test.npy' -max_length 200 -item_n 4068791 -cate_n 9408 -long_seq_split '0:200' -short_model_type DIN -long_model_type DIN -use_cross_feature True -attn_func scaled_dot_product -hard_or_soft soft -top_k 20 -use_aux_loss True -use_time_mode concat -time_embedding_dim 16 -epoch 2 -category_embedding_dim 16 -batch_size 2048 -learning_rate 0.01 -weight_decay 0.000001 -seed 1 2 3 4 5 -log_dir log/taobao/model_twin_4e -test_interval 150 -log_interval 50 -model_name four_embedding

# TWIN w/proj
CUDA_VISIBLE_DEVICES=4 python train_pytorch.py -train_dataset_path 'data/taobao/train_{}_{}.npy' -val_dataset_path 'data/taobao/val.npy' -test_dataset_path 'data/taobao/test.npy' -max_length 200 -item_n 4068791 -cate_n 9408 -long_seq_split '0:200' -short_model_type DIN -long_model_type DIN -use_cross_feature True -attn_func scaled_dot_product -hard_or_soft soft -top_k 20 -use_aux_loss True -use_time_mode concat -time_embedding_dim 16 -epoch 2 -category_embedding_dim 16 -batch_size 2048 -learning_rate 0.01 -weight_decay 0.000001 -seed 1 2 3 4 5 -log_dir log/taobao/model_twin_with_projection -test_interval 150 -log_interval 50 -model_name projection -mlp_position_after_concat

# TWIN hard
CUDA_VISIBLE_DEVICES=4 python train_pytorch.py -train_dataset_path 'data/taobao/train_{}_{}.npy' -val_dataset_path 'data/taobao/val.npy' -test_dataset_path 'data/taobao/test.npy' -max_length 200 -item_n 4068791 -cate_n 9408 -long_seq_split '0:200' -short_model_type DIN -long_model_type DIN -use_cross_feature True -attn_func scaled_dot_product -hard_or_soft hard -top_k 20 -use_aux_loss True -use_time_mode concat -time_embedding_dim 16 -epoch 2 -category_embedding_dim 16 -batch_size 2048 -learning_rate 0.01 -weight_decay 0.000001 -seed 1 2 3 4 5 -log_dir log/taobao/model_twin_hard -test_interval 150 -log_interval 50 -model_name twin

# TWIN no TR
CUDA_VISIBLE_DEVICES=4 python train_pytorch.py -train_dataset_path 'data/taobao/train_{}_{}.npy' -val_dataset_path 'data/taobao/val.npy' -test_dataset_path 'data/taobao/test.npy' -max_length 200 -item_n 4068791 -cate_n 9408 -long_seq_split '0:200' -short_model_type DIN -long_model_type DIN -attn_func scaled_dot_product -hard_or_soft soft -top_k 20 -use_aux_loss True -use_time_mode concat -time_embedding_dim 16 -epoch 2 -category_embedding_dim 16 -batch_size 2048 -learning_rate 0.01 -weight_decay 0.000001 -seed 1 2 3 4 5 -log_dir log/taobao/model_twin_no_TR -test_interval 150 -log_interval 50 -model_name twin


# We have tried some other decoupling methods, but they prove to be failure. If you are interest in our research process, you can run these models by
# assigning the model name to 'three_embedding', 'projection_two_linear', 'projection_with_mlp', 'projection_no_item', 'projection_decouple_by_category',
# 'projection_only_time', 'projection_only_category', 'projection_only_item', 'projection_large_emb_small_output', or change the parameters according to
# the information in README.md.
