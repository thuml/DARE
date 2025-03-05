import os
import argparse
import numpy as np
import tqdm
import warnings
import sys

from model.din_pytorch import DIN
import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_save_dir', type=str, default='None')
    parser.add_argument('--test_dataset', type=str, default='None')
    parser.add_argument('--model_list', type=str, nargs='+', default=['two embedding', 'four embedding', 'twin', 'twin with projection', 'DIN', 'gt'])
    parser.add_argument('--model_save_dir', type=str, nargs='+', default=[])
    parser.add_argument('--record_path', type=str, default='None')
    parser.add_argument('--select_num', type=int, default=20)
    parser.add_argument('--dataset_name', type=str, default="taobao")
    args = parser.parse_args()

    assert len(args.model_list) == len(args.model_save_dir) + 1

    parent_dir = os.path.dirname(args.record_path)
    grand_parent_dir = os.path.dirname(parent_dir)
    assert os.path.exists(grand_parent_dir), grand_parent_dir
    if not os.path.exists(parent_dir):
        os.mkdir(parent_dir)
    if not os.path.exists(args.record_path):
        os.mkdir(args.record_path)

    if not os.path.exists(os.path.join(args.record_path, "simulate_gsu_case_study")):
        os.mkdir(os.path.join(args.record_path, "simulate_gsu_case_study"))

    save_dir = args.gt_save_dir
    adj_p = np.load(os.path.join(save_dir, 'adj_p.npy'), allow_pickle=True).item()
    adj_n = np.load(os.path.join(save_dir, 'adj_n.npy'), allow_pickle=True).item()
    target_p = np.load(os.path.join(save_dir, 'target_p.npy'), allow_pickle=True)
    target_n = np.load(os.path.join(save_dir, 'target_n.npy'), allow_pickle=True)

    print("loading test")

    test_dataset = np.load(args.test_dataset, allow_pickle=True)

    warnings.filterwarnings("error", category=RuntimeWarning)

    print("loading DIN")

    if 'DIN' in args.model_list:
        DIN_model = DIN(
            category_embedding_dim=16,
            item_embedding_dim=16,
            time_embedding_dim=16,
            attention_category_embedding_dim=-1,
            attention_time_embedding_dim=-1,
            attention_item_embedding_dim=-1,
            item_n=4068791 if args.dataset_name == 'taobao' else 1080667,
            cate_n=9408 if args.dataset_name == 'taobao' else 1493,
            batch_size=1,
            max_length=200,
            use_cross_feature=True,
            attn_func='learnable',
            use_aux_loss=False,
            use_time=False,
            use_time_mode='concat',
            short_seq_split=None,
            long_seq_split='0:200',
            soft_search=True,
            top_k=20,
            use_long_seq_average=False,
            model_name='DIN',
            mlp_position_after_concat=False,
            only_odot=False,
            no_batch_norm_if_one_batchsize=False
        )
        DIN_model.load(os.path.join(args.model_save_dir[4], "model", "best_DIN.pth"))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        DIN_model = DIN_model.to(device)

    for index, model_name in enumerate(args.model_list):
        print("loading", model_name)
        if model_name == 'two embedding':
            model_save_dir = args.model_save_dir[index]
            c_embedding_two_embedding = np.load(os.path.join(model_save_dir, 'learned_embedding', 'attention_for_behavior_category.npy'))
            p_embedding_two_embedding = np.load(os.path.join(model_save_dir, 'learned_embedding', 'attention_for_behavior_time.npy'))
        elif model_name == 'four embedding':
            model_save_dir = args.model_save_dir[index]
            c_embedding_behavior_four_embedding = np.load(os.path.join(model_save_dir, 'learned_embedding', 'attention_for_behavior_category.npy'))
            p_embedding_behavior_four_embedding = np.load(os.path.join(model_save_dir, 'learned_embedding', 'attention_for_behavior_time.npy'))
            c_embedding_target_four_embedding = np.load(os.path.join(model_save_dir, 'learned_embedding', 'attention_for_target_category.npy'))
            p_embedding_target_four_embedding = np.load(os.path.join(model_save_dir, 'learned_embedding', 'attention_for_target_time.npy'))
        elif model_name == 'twin with projection':
            model_save_dir = args.model_save_dir[index]
            c_embedding_twin_projection = np.load(os.path.join(model_save_dir, 'learned_embedding', 'attention_for_behavior_category.npy'))
            p_embedding_twin_projection = np.load(os.path.join(model_save_dir, 'learned_embedding', 'attention_for_behavior_time.npy'))
            category_target_mlp_twin_projection = np.load(os.path.join(model_save_dir, 'learned_embedding', 'q_category_mlp.npy'))
            time_target_mlp_twin_projection = np.load(os.path.join(model_save_dir, 'learned_embedding', 'q_time_mlp.npy'))
            category_behavior_mlp_twin_projection = np.load(os.path.join(model_save_dir, 'learned_embedding', 'k_category_mlp.npy'))
            time_behavior_mlp_twin_projection = np.load(os.path.join(model_save_dir, 'learned_embedding', 'k_time_mlp.npy'))
        elif model_name == 'twin':
            model_save_dir = args.model_save_dir[index]
            c_embedding_twin = np.load(os.path.join(model_save_dir, 'learned_embedding', 'attention_for_behavior_category.npy'))
            p_embedding_twin = np.load(os.path.join(model_save_dir, 'learned_embedding', 'attention_for_behavior_time.npy'))

    aim_record = {}
    norm_selected_quality = {}
    for model_name in args.model_list:
        aim_record[model_name] = {}
        norm_selected_quality[model_name] = {}
        for i in range(20):
            aim_record[model_name][i] = []
            norm_selected_quality[model_name][i] = []
    skip_bad_quality_data_num = 0
    data_index = 0
    for one_data in tqdm.tqdm(test_dataset):
        data_index += 1
        _, _, target_cid, _, _, cid_hist, = one_data[:6]
        gt_mul_information = []

        t_p = target_p[target_cid]
        t_n = target_n[target_cid]
        s = t_p + t_n

        if s < 700:
            skip_bad_quality_data_num += 1
            if skip_bad_quality_data_num % 100 == 0:
                print("haven skip bad_quality_data num: "+str(skip_bad_quality_data_num))
            continue
        try:
            bad_quality_behaviour_num = 0
            for position, one_behaviour_cid in enumerate(reversed(cid_hist)):
                if one_behaviour_cid==0:
                    gt_mul_information.append(-(2**31))
                    continue

                # check quality
                if (target_cid, one_behaviour_cid, position+1) in adj_p:
                    x_p = adj_p[(target_cid, one_behaviour_cid, position+1)]
                else:
                    x_p = 1e-5
                if (target_cid, one_behaviour_cid, position+1) in adj_n:
                    x_n = adj_n[(target_cid, one_behaviour_cid, position+1)]
                else:
                    x_n = 1e-5

                if x_p < 50 or x_n < 50:
                    bad_quality_behaviour_num += 1

                m1 = x_p / s * np.log2(x_p * s / ((x_p + x_n) * t_p))  # x_i = 1, y = 1
                m2 = x_n / s * np.log2(x_n * s / ((x_p + x_n) * t_n))  # x_i = 1, y = 0
                m3 = (t_p - x_p) / s * np.log2((t_p - x_p) * s / ((t_p + t_n - x_p - x_n) * t_p))  # x_i = 0, y = 1
                m4 = (t_n - x_n) / s * np.log2((t_n - x_n) * s / ((t_p + t_n - x_p - x_n) * t_n))  # x_i = 0, y = 0
                m = m1 + m2 + m3 + m4
                gt_mul_information.append(m)

            if bad_quality_behaviour_num <= 20:
                numpy_gt_mul_information = np.array(gt_mul_information)
                gt_selected_index = np.argsort(numpy_gt_mul_information)[::-1][:args.select_num]
                scores_record = {}
                for index, model_name in enumerate(args.model_list):  # 每个模型
                    learned_mul_information = []
                    if model_name == 'two embedding':
                        model_save_dir = args.model_save_dir[index]
                        c_embedding = c_embedding_two_embedding
                        p_embedding = p_embedding_two_embedding
                        for position, one_behaviour_cid in enumerate(reversed(cid_hist)):
                            semantics_score = (c_embedding[target_cid] * c_embedding[one_behaviour_cid]) / ((len(c_embedding[target_cid])*2) ** 0.5)
                            time_score = (p_embedding[0] * p_embedding[position + 1]) / (len(p_embedding[0]) ** 0.5)
                            total_score = np.sum(semantics_score) + np.sum(time_score)
                            learned_mul_information.append(total_score)
                        numpy_learned_mul_information = np.array(learned_mul_information)
                        scores_record[model_name] = numpy_learned_mul_information
                    elif model_name == 'twin':
                        model_save_dir = args.model_save_dir[index]
                        c_embedding = c_embedding_twin
                        p_embedding = p_embedding_twin
                        for position, one_behaviour_cid in enumerate(reversed(cid_hist)):
                            semantics_score = (c_embedding[target_cid] * c_embedding[one_behaviour_cid]) / ((len(c_embedding[target_cid]) * 2) ** 0.5)
                            time_score = (p_embedding[0] * p_embedding[position + 1]) / (len(p_embedding[0]) ** 0.5)
                            total_score = np.sum(semantics_score) + np.sum(time_score)
                            learned_mul_information.append(total_score)
                        numpy_learned_mul_information = np.array(learned_mul_information)
                        scores_record[model_name] = numpy_learned_mul_information
                    elif model_name == 'four embedding':
                        model_save_dir = args.model_save_dir[index]
                        c_embedding_behavior = c_embedding_behavior_four_embedding
                        p_embedding_behavior = p_embedding_behavior_four_embedding
                        c_embedding_target = c_embedding_target_four_embedding
                        p_embedding_target = p_embedding_target_four_embedding
                        for position, one_behaviour_cid in enumerate(reversed(cid_hist)):
                            semantics_score = (c_embedding_target[target_cid] * c_embedding_behavior[one_behaviour_cid]) / ((len(c_embedding_target[target_cid])*2) ** 0.5)
                            time_score = (p_embedding_target[0] * p_embedding_behavior[position + 1]) / (len(p_embedding_target[0]) ** 0.5)
                            total_score = np.sum(semantics_score) + np.sum(time_score)
                            learned_mul_information.append(total_score)
                        numpy_learned_mul_information = np.array(learned_mul_information)
                        scores_record[model_name] = numpy_learned_mul_information
                    elif model_name == 'twin with projection':
                        model_save_dir = args.model_save_dir[index]
                        c_embedding = c_embedding_twin_projection
                        p_embedding = p_embedding_twin_projection
                        category_target_mlp = category_target_mlp_twin_projection
                        time_target_mlp = time_target_mlp_twin_projection
                        category_behavior_mlp = category_behavior_mlp_twin_projection
                        time_behavior_mlp = time_behavior_mlp_twin_projection
                        for position, one_behaviour_cid in enumerate(reversed(cid_hist)):
                            c_mlp_target = np.squeeze(np.dot(c_embedding[target_cid][np.newaxis, :], category_target_mlp))
                            c_mlp_behavior = np.squeeze(np.dot(c_embedding[one_behaviour_cid][np.newaxis, :], category_behavior_mlp))
                            p_mlp_target = np.squeeze(np.dot(p_embedding[0][np.newaxis, :], time_target_mlp))
                            p_mlp_behavior = np.squeeze(np.dot(p_embedding[position + 1][np.newaxis, :], time_behavior_mlp))
                            semantics_score = (c_mlp_target * c_mlp_behavior) / ((len(c_mlp_target)*2) ** 0.5)
                            time_score = (p_mlp_target * p_mlp_behavior) / (len(p_embedding[0]) ** 0.5)
                            total_score = np.sum(semantics_score) + np.sum(time_score)
                            learned_mul_information.append(total_score)
                        numpy_learned_mul_information = np.array(learned_mul_information)
                        scores_record[model_name] = numpy_learned_mul_information
                    elif model_name == 'DIN':
                        model_save_dir = args.model_save_dir[index]
                        uid = [one_data[0]]
                        iid = [one_data[1]]
                        cid = [one_data[2]]
                        target = [one_data[3]]
                        hist_iid = [one_data[4][-200:]]
                        hist_cid = [one_data[5][-200:]]

                        uid, iid, cid, target, hist_iid, hist_cid = list(
                            map(np.array, [uid, iid, cid, target, hist_iid, hist_cid]))
                        mask = np.greater(hist_iid, 0).astype(np.float32)

                        tid = np.zeros(iid.shape, dtype=np.int32)
                        hist_tid = np.array([[200 - x for x in range(200)]])
                        # hist_tid = np.repeat(hist_tid, 1, axis=0)

                        input_data = {
                            'uid': uid,
                            'iid': iid,
                            'cid': cid,
                            'tid': tid,
                            'hist_iid': hist_iid,
                            'hist_cid': hist_cid,
                            'hist_tid': hist_tid,
                            'mask': mask,
                            'target': target
                        }
                        numpy_learned_mul_information = np.squeeze(DIN_model.forward(input_data, only_return_scores=True).detach().cpu().numpy())
                        numpy_learned_mul_information = numpy_learned_mul_information[::-1]
                        scores_record[model_name] = numpy_learned_mul_information
                    elif model_name == 'gt':
                        learned_mul_information = gt_mul_information
                        numpy_learned_mul_information = np.array(learned_mul_information)
                        scores_record[model_name] = numpy_learned_mul_information

                    for j in range(20):
                        learned_selected_index = np.argsort(numpy_learned_mul_information)[::-1][:j+1]
                        result = np.isin(learned_selected_index, gt_selected_index)
                        aim_num = np.sum(result)
                        aim_record[model_name][j].append(aim_num)

                    learned_selected_index = np.argsort(numpy_learned_mul_information)[::-1][:args.select_num]
                    # print(model_name, learned_selected_index)
                    # input("press to continue")
                    gt_selected_mul_information = 0
                    learned_selected_mul_information = 0
                    for i, one_gt_index in enumerate(gt_selected_index):
                        gt_selected_mul_information += gt_mul_information[one_gt_index] / np.log2(i+2)
                    for i, one_learned_index in enumerate(learned_selected_index):
                        learned_selected_mul_information += gt_mul_information[one_learned_index] / np.log2(i+2)

                        if gt_selected_mul_information > 0:
                            norm_selected_quality[model_name][i].append(learned_selected_mul_information / gt_selected_mul_information)
                        else:
                            print("unexpected case: gt_selected_mul_information is not positive", gt_selected_mul_information)
                    scores_record[model_name+"score"] = learned_selected_mul_information / gt_selected_mul_information
                if scores_record["two embedding"+"score"] > scores_record["twin"+"score"] + 0.3 and scores_record["two embedding"+"score"] > scores_record["DIN"+"score"] + 0.3:
                    aim_list = [cid_hist]
                    for model_name in args.model_list:
                        aim_list.append(scores_record[model_name])
                    record_numpy = np.stack(aim_list)
                    np.save(os.path.join(args.record_path, "simulate_gsu_case_study", "index_"+str(data_index)+"_aim_"+str(target_cid)+".npy"), record_numpy)
                    print("save_one_example")
            else:
                skip_bad_quality_data_num += 1
                if skip_bad_quality_data_num % 100 == 0:
                    print("haven skip bad_quality_data num: " + str(skip_bad_quality_data_num))
        except Warning as e:
            print("meet runtime warning, skip one data")

    with open(os.path.join(args.record_path, "aim_result_ndcg.txt"), "a") as file:
        for model_name in args.model_list:
            file.write(model_name)
            file.write(":\n")
            file.write("hit: ")
            for i in range(20):
                file.write(str(sum(aim_record[model_name][i])/len(aim_record[model_name][i])))
                if i != 19:
                    file.write(" ")
            file.write("\nndcg: ")
            for i in range(20):
                file.write(str(sum(norm_selected_quality[model_name][i]) / len(norm_selected_quality[model_name][i])))
                if i != 19:
                    file.write(" ")
            file.write("\n\n")


