CUDA_VISIBLE_DEVICES=3 python collect_repr.py -mode train -test_dataset_path '../../data/taobao/test.npy' -max_length 200 -item_n 4068791 -cate_n 9408 -long_seq_split '0:200' -short_model_type DIN -long_model_type DIN -use_cross_feature True -attn_func scaled_dot_product -hard_or_soft soft -top_k 20 -use_aux_loss True -use_time_mode concat -time_embedding_dim 16 -epoch 2 -category_embedding_dim 16 -batch_size 2048 -learning_rate 0.01 -weight_decay 0.000001 -seed 1 2 3 4 5 -model_log_dir '../../log/taobao/model_DARE' -log_dir 'analysis_result/taobao/model_DARE' -test_interval 150 -log_interval 50 -model_name DARE

CUDA_VISIBLE_DEVICES=3 python analysis.py --log_dir 'analysis_result/taobao/model_DARE'

CUDA_VISIBLE_DEVICES=3 python collect_repr.py -mode train -test_dataset_path '../../data/taobao/test.npy' -max_length 200 -item_n 4068791 -cate_n 9408 -long_seq_split '0:200' -short_model_type DIN -long_model_type DIN -use_cross_feature True -attn_func scaled_dot_product -hard_or_soft soft -top_k 20 -use_aux_loss True -use_time_mode concat -time_embedding_dim 16 -epoch 2 -category_embedding_dim 16 -batch_size 2048 -learning_rate 0.01 -weight_decay 0.000001 -seed 1 2 3 4 5 -model_log_dir '../../log/taobao/model_twin' -log_dir 'analysis_result/taobao/model_twin' -test_interval 150 -log_interval 50 -model_name twin

CUDA_VISIBLE_DEVICES=3 python analysis.py --log_dir 'analysis_result/taobao/model_twin'


CUDA_VISIBLE_DEVICES=3 python collect_repr.py -mode train -test_dataset_path '../../data/taobao/test.npy' -max_length 200 -item_n 4068791 -cate_n 9408 -long_seq_split '0:200' -short_model_type DIN -long_model_type DIN -use_cross_feature True -attn_func learnable -hard_or_soft soft -top_k 20 -use_aux_loss True -use_time_mode concat -time_embedding_dim 16 -epoch 2 -category_embedding_dim 16 -batch_size 2048 -learning_rate 0.01 -weight_decay 0.000001 -seed 1 2 3 4 5 -model_log_dir '../../log/taobao/model_DIN' -log_dir 'analysis_result/taobao/model_DIN' -test_interval 150 -log_interval 50 -model_name DIN -use_time

CUDA_VISIBLE_DEVICES=3 python analysis.py --log_dir 'analysis_result/taobao/model_DIN'

CUDA_VISIBLE_DEVICES=3 python collect_repr.py -mode train -test_dataset_path '../../data/taobao/test.npy' -max_length 200 -item_n 4068791 -cate_n 9408 -long_seq_split '0:200' -short_model_type DIN -long_model_type DIN -use_cross_feature True -attn_func scaled_dot_product -hard_or_soft soft -top_k 20 -use_aux_loss True -use_time_mode concat -time_embedding_dim 16 -epoch 2 -category_embedding_dim 16 -batch_size 2048 -learning_rate 0.01 -weight_decay 0.000001 -seed 1 2 3 4 5 -model_log_dir '../../log/taobao/model_twin_with_projection' -log_dir 'analysis_result/taobao/model_twin_with_projection' -test_interval 150 -log_interval 50 -model_name projection -mlp_position_after_concat

CUDA_VISIBLE_DEVICES=3 python analysis.py --log_dir 'analysis_result/taobao/model_twin_with_projection'

CUDA_VISIBLE_DEVICES=3 python collect_repr.py -mode train -test_dataset_path '../../data/taobao/test.npy' -max_length 200 -item_n 4068791 -cate_n 9408 -long_seq_split '0:200' -short_model_type DIN -long_model_type DIN -use_cross_feature True -attn_func scaled_dot_product -hard_or_soft soft -top_k 20 -use_aux_loss True -use_time_mode concat -time_embedding_dim 16 -epoch 2 -category_embedding_dim 16 -batch_size 2048 -learning_rate 0.01 -weight_decay 0.000001 -seed 1 2 3 4 5 -model_log_dir '../../log/taobao/model_twin_4e' -log_dir 'analysis_result/taobao/model_twin_4e' -test_interval 150 -log_interval 50 -model_name four_embedding

CUDA_VISIBLE_DEVICES=3 python analysis.py --log_dir 'analysis_result/taobao/model_twin_4e'



CUDA_VISIBLE_DEVICES=3 python collect_repr.py -mode train -test_dataset_path '../../data/taobao/test.npy' -max_length 200 -item_n 4068791 -cate_n 9408 -long_seq_split '0:200' -short_model_type DIN -long_model_type DIN -attn_func scaled_dot_product -hard_or_soft soft -top_k 20 -use_aux_loss True -use_time_mode concat -time_embedding_dim 16 -epoch 2 -category_embedding_dim 16 -batch_size 2048 -learning_rate 0.01 -weight_decay 0.000001 -seed 1 2 3 4 5 -model_log_dir '../../log/taobao/model_DARE_no_TR' -log_dir 'analysis_result/taobao/model_DARE_no_TR' -test_interval 150 -log_interval 50 -model_name DARE

CUDA_VISIBLE_DEVICES=3 python analysis.py --log_dir 'analysis_result/taobao/model_DARE_no_TR'

CUDA_VISIBLE_DEVICES=3 python collect_repr.py -mode train -test_dataset_path '../../data/taobao/test.npy' -max_length 200 -item_n 4068791 -cate_n 9408 -long_seq_split '0:200' -short_model_type DIN -long_model_type DIN -attn_func scaled_dot_product -hard_or_soft soft -top_k 20 -use_aux_loss True -use_time_mode concat -time_embedding_dim 16 -epoch 2 -category_embedding_dim 16 -batch_size 2048 -learning_rate 0.01 -weight_decay 0.000001 -seed 1 2 3 4 5 -model_log_dir '../../log/taobao/model_twin_no_TR' -log_dir 'analysis_result/taobao/model_twin_no_TR' -test_interval 150 -log_interval 50 -model_name twin

CUDA_VISIBLE_DEVICES=3 python analysis.py --log_dir 'analysis_result/taobao/model_twin_no_TR'

