python draw_analysis_result.py --dataset_name "Taobao" --twin_store_path "analysis_result/taobao/model_twin" --two_embedding_path "analysis_result/taobao/model_DARE" --four_embedding_path "analysis_result/taobao/model_twin_4e" --DIN_path "analysis_result/taobao/model_DIN" --twin_linear_path "analysis_result/taobao/model_twin_with_projection"

python draw_analysis_result.py --dataset_name "Tmall" --twin_store_path "analysis_result/tmall/model_twin" --two_embedding_path "analysis_result/tmall/model_DARE" --four_embedding_path "analysis_result/tmall/model_twin_4e" --DIN_path "analysis_result/tmall/model_DIN" --twin_linear_path "analysis_result/tmall/model_twin_with_projection"


python draw_odot_analysis_result.py --dataset_name "Taobao" --twin_without_odot_path "analysis_result/taobao/model_twin_no_TR" --twin_only_odot_path "analysis_result/taobao/model_twin" --two_embedding_without_odot_path "analysis_result/taobao/model_DARE_no_TR" --two_embedding_only_odot_path "analysis_result/taobao/model_DARE"

python draw_odot_analysis_result.py --dataset_name "Tmall" --twin_without_odot_path "analysis_result/tmall/model_twin_no_TR" --twin_only_odot_path "analysis_result/tmall/model_twin" --two_embedding_without_odot_path "analysis_result/tmall/model_DARE_no_TR" --two_embedding_only_odot_path "analysis_result/tmall/model_DARE"
