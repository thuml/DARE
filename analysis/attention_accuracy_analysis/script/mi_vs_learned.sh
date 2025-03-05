# Taobao
python calc_gt.py --train_dataset_path '../../data/taobao/train_{}_{}.npy' --test_dataset_path None --n_category 9408 --save_dir 'visualization_result/taobao' --n_most_common 10 --vis_nc 10 --vis_np 10 --figsize 5 5 --normalize_mi_mat

python calc_learned.py --model gsu --c_embedding_path '../../log/taobao/model_twin/learned_embedding/attention_for_behavior_category.npy' --p_embedding_path '../../log/taobao/model_twin/learned_embedding/attention_for_behavior_time.npy' --most_common_cid_path 'visualization_result/taobao/most_common_10_category.npy' --vis_nc 10 --vis_np 10 --figsize 5 5 --save_dir 'visualization_result/taobao/twin_learned'

python calc_learned.py --model gsu --c_embedding_path '../../log/taobao/model_DARE/learned_embedding/attention_for_behavior_category.npy' --p_embedding_path '../../log/taobao/model_DARE/learned_embedding/attention_for_behavior_time.npy' --most_common_cid_path 'visualization_result/taobao/most_common_10_category.npy' --vis_nc 10 --vis_np 10 --figsize 5 5 --save_dir 'visualization_result/taobao/DARE_learned'

# Tmall
python calc_gt.py --train_dataset_path '../../data/tmall/train_{}_{}.npy' --test_dataset_path None --n_category 1493 --save_dir 'visualization_result/tmall' --n_most_common 10 --vis_nc 10 --vis_np 10 --figsize 5 5 --normalize_mi_mat

python calc_learned.py --model gsu --c_embedding_path '../../log/tmall/model_twin/learned_embedding/attention_for_behavior_category.npy' --p_embedding_path '../../log/tmall/model_twin/learned_embedding/attention_for_behavior_time.npy' --most_common_cid_path 'visualization_result/tmall/most_common_10_category.npy' --vis_nc 10 --vis_np 10 --figsize 5 5 --save_dir 'visualization_result/tmall/twin_learned'

python calc_learned.py --model gsu --c_embedding_path '../../log/tmall/model_DARE/learned_embedding/attention_for_behavior_category.npy' --p_embedding_path '../../log/tmall/model_DARE/learned_embedding/attention_for_behavior_time.npy' --most_common_cid_path 'visualization_result/tmall/most_common_10_category.npy' --vis_nc 10 --vis_np 10 --figsize 5 5 --save_dir 'visualization_result/tmall/DARE_learned'
