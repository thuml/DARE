# Taobao
CUDA_VISIBLE_DEVICES=5 python gsu_performance.py --gt_save_dir visualization_result/taobao --test_dataset ../../data/taobao/test.npy --record_path ./gsu_result/taobao --select_num 20 --dataset_name taobao --model_save_dir ../../log/taobao/model_DARE ../../log/taobao/model_twin_4e ../../log/taobao/model_twin ../../log/taobao/model_twin_with_projection ../../log/taobao/model_DIN

# Tmall
CUDA_VISIBLE_DEVICES=5 python gsu_performance.py --gt_save_dir visualization_result/tmall --test_dataset ../../data/tmall/test.npy --record_path ./gsu_result/tmall --select_num 20 --dataset_name tmall --model_save_dir ../../log/tmall/model_DARE ../../log/tmall/model_twin_4e ../../log/tmall/model_twin ../../log/tmall/model_twin_with_projection ../../log/tmall/model_DIN



