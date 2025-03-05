import os
import time
import random
import sys

import numpy as np
import torch

from dataset import SeqRecDataset
from tools.logger import CompleteLogger
from utils import calc_auc
import logging
from model.din_pytorch import DIN
import torch.optim as optim
import torch.nn as nn


def create_model(model_type, args):
    if model_type == "DIN":
        model = DIN(
            category_embedding_dim=args.category_embedding_dim,
            item_embedding_dim=args.item_embedding_dim,
            time_embedding_dim=args.time_embedding_dim,
            attention_category_embedding_dim=args.attention_category_embedding_dim,
            attention_time_embedding_dim=args.attention_time_embedding_dim,
            attention_item_embedding_dim=args.attention_item_embedding_dim,
            item_n=args.item_n,
            cate_n=args.cate_n,
            batch_size=args.batch_size,
            max_length=args.max_length,
            use_cross_feature=args.use_cross_feature,
            attn_func=args.attn_func,
            use_aux_loss=args.use_aux_loss,
            use_time=args.use_time,
            use_time_mode=args.use_time_mode,
            short_seq_split=args.short_seq_split,
            long_seq_split=args.long_seq_split,
            soft_search=(args.hard_or_soft == 'soft'),
            top_k=args.top_k,
            use_long_seq_average=args.use_long_seq_average,
            model_name=args.model_name,
            mlp_position_after_concat=args.mlp_position_after_concat,
            only_odot=args.only_odot,
            mlp_hidden_layer=args.mlp_hidden_layer,
            observe_attn_repr_grad=args.observe_attn_repr_grad
        )
    else:
        raise ValueError("Unknown model_type: %s" % model_type)

    return model


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_args():
    import argparse
    parser = argparse.ArgumentParser()

    '''
        The meaning of some key parameters：
        
        dataset parameters
        | parameter  | meaning                                       | 
        |------------|-----------------------------------------------|
        |max_length  | the maximum history behavior length of a user |
        |item_n      | total item number in the whole dataset        |
        |cate_n      | total category number in the whole dataset    |
        
        model config parameters
        | parameter                 | meaning                                                                                                                                                                                                                                                                                                                                                | 
        |---------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
        | short_seq_split           | if not None, the assigned short sequence will always be retrieved (will not be counted in the "top_k")                                                                                                                                                                                                                                                 |
        | short_model_type          | we only support DIN now. You can add other different basic models.                                                                                                                                                                                                                                                                                     |
        | long_model_type           | we only support DIN now.                                                                                                                                                                                                                                                                                                                               |
        | attn_func                 | "learnable" for DIN, and "scale dot product" for other models.                                                                                                                                                                                                                                                                                         |
        | hard_or_soft              | use hard search or soft search in the retrieval. We focus on soft search in our experiments.                                                                                                                                                                                                                                                           |
        | top_k                     | the model will retrieve -top_k behaviors from history behaviors.                                                                                                                                                                                                                                                                                       |
        | use_time                  | DIN do not use time information, while others use time information.                                                                                                                                                                                                                                                                                    |
        | use_time_mode             | We focus on "concat" in our experiments, meaning an item will be embedded into torch.cat([item_embed, category_embed, time_embed])                                                                                                                                                                                                                     |
        | model_name                | **An important parameter**. The model you use. Options include "DARE", "TWIN", "DIN", and so on.                                                                                                                                                                                                                                                       |
        | use_aux_loss              | a simple trick that may benefit model training. You can find it in model/din_pytorch.py.                                                                                                                                                                                                                                                               |
        | use_long_seq_average      | a simple trick that may benefit model training.                                                                                                                                                                                                                                                                                                        |
        | mlp_position_after_concat | if you use projection to decouple the modules, there are two options: <br> 1. projection(torch.cat([item_embed, category_embed, time_embed]))  (-mlp_position_after_concat = True) <br> 2. torch.cat([item_embed, category_embed, time_embed])  (-mlp_position_after_concat = False) <br> We use -mlp_position_after_concat = True in our experiments. |
        | avoid_domination          | one of the decoupling methods we tried. If set to True, we will manually update the gradients to keep their size the same.                                                                                                                                                                                                                             |                                                                                                                                                                                                                 
        | only_odot                 | by default, our input of the final MLP is [history, target, history \odot target]. If set to True, it will be [history \odot target].                                                                                                                                                                                                                  |                                                                                                                                                                                                     
        | use_cross_feature         | use target representation (the odot product) or not.                                                                                                                                                                                                                                                                                                 |                                                                                                                                                                                                                                                                                    

        hyperparameter parameters
        | parameter                                    | meaning                                                                                                                                   | 
        |----------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------|
        | attention_{time/category/time}_embedding_dim | This is the parameter specially for DARE, since DARE supports using different embedding dimension for attention and representation.       |
        | mlp_hidden_layer                             | this is the parameter specially for "projection_with_mlp" (changing the linear projection to MLP in TWIN w/ proj), defining the MLP size. |
    '''

    # dataset config
    parser.add_argument('-train_dataset_path', type=str, default=None)
    parser.add_argument('-val_dataset_path', type=str, default=None)
    parser.add_argument('-test_dataset_path', type=str, default=None)
    parser.add_argument("-max_length", type=int, default=200)
    parser.add_argument('-item_n', type=int, default=4068791)
    parser.add_argument('-cate_n', type=int, default=9408)

    # model config
    parser.add_argument("-short_seq_split", type=str, default=None)
    parser.add_argument("-long_seq_split", type=str, default=None)
    parser.add_argument("-short_model_type", type=str)
    parser.add_argument("-long_model_type", type=str)
    parser.add_argument('-attn_func', type=str, default='learnable')
    parser.add_argument("-hard_or_soft", type=str, default='soft')
    parser.add_argument("-top_k", type=int, default=20)
    parser.add_argument("-use_time", action="store_false")
    parser.add_argument("-use_time_mode", type=str, default='concat')

    parser.add_argument("-model_name", type=str, default="DARE")

    parser.add_argument("-use_aux_loss", type=bool, default=False)
    parser.add_argument('-use_long_seq_average', action="store_true")
    parser.add_argument('-mlp_position_after_concat', action="store_false")
    parser.add_argument("-avoid_domination", action='store_true')
    parser.add_argument("-only_odot", action='store_true')
    parser.add_argument('-use_cross_feature', type=bool, default=False)

    # hyperparameter
    parser.add_argument("-epoch", type=int, default=1)
    parser.add_argument("-batch_size", type=int, default=256)
    parser.add_argument("-category_embedding_dim", type=int, default=16)
    parser.add_argument("-item_embedding_dim", type=int, default=-1)
    parser.add_argument("-time_embedding_dim", type=int, default=4)
    parser.add_argument("-attention_time_embedding_dim", type=int, default=-1)
    parser.add_argument("-attention_category_embedding_dim", type=int, default=-1)
    parser.add_argument("-attention_item_embedding_dim", type=int, default=-1)
    parser.add_argument('-learning_rate', type=float, default=0.001)
    parser.add_argument('-weight_decay', type=float, default=1e-6)
    parser.add_argument("-seed", type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument("-mlp_hidden_layer", type=int, nargs='+', default=[32])

    # logging
    parser.add_argument("-level", type=str, default='INFO')
    parser.add_argument('-log_dir', type=str, default='log')
    parser.add_argument("-test_interval", type=int, default=1000)
    parser.add_argument("-log_interval", type=int, default=100)
    parser.add_argument("-stor_grad", action='store_true')
    parser.add_argument("-observe_attn_repr_grad", action='store_true')

    args = parser.parse_args()
    assert args.hard_or_soft in ['hard', 'soft']
    return args


def evaluate(model, eval_dataset, use_aux_loss):
    logging.info("evaluating starts.")
    loss_sum = 0.
    accuracy_sum = 0.
    iteration = 0
    stored_arr = []
    grouped_by_category = {}
    positive_polibility = []
    with torch.no_grad():
        total_eval_time = 0
        for batch_data in eval_dataset:
            iteration += 1
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            target = torch.tensor(batch_data['target'], dtype=torch.int32).to(device)
            category = torch.tensor(batch_data['cid'], dtype=torch.int32).to(device)
            user_id = torch.tensor(batch_data['uid'], dtype=torch.int32).to(device)
            start_one_eval_time = time.time()
            if use_aux_loss:
                predicted_probability, loss, _ = model.forward(batch_data, mode="evaluate")
            else:
                predicted_probability, loss = model.forward(batch_data, mode="evaluate")
            total_eval_time += time.time() - start_one_eval_time
            positive_polibility.append(predicted_probability[:, 0].detach().cpu().numpy())

            loss_sum += loss
            one_hot_predicted_result = torch.round(predicted_probability)
            correctness = torch.sum(torch.mul(one_hot_predicted_result, target), dim=1).float()
            accuracy = torch.mean(correctness)
            accuracy_sum += accuracy.item()
            for p, t, c, u in zip(predicted_probability[:, 0].tolist(), target[:, 0].tolist(), category.tolist(),
                                  user_id.tolist()):
                stored_arr.append([p, t])
                if c not in grouped_by_category:
                    grouped_by_category[c] = []
                grouped_by_category[c].append([p, t])

    positive_probability = np.stack(positive_polibility)
    negative_probability = 1 - positive_probability
    max_probability = np.max(np.stack([positive_probability, negative_probability]), axis=0)
    print("positive_probability mean and std: ", np.mean(positive_polibility), np.std(positive_polibility))
    print("max_probability mean: ", np.mean(max_probability), np.std(max_probability))

    validation_auc = calc_auc(stored_arr)
    total_test_num = 0
    category_average_auc = 0
    for category, group in grouped_by_category.items():
        group_len = len(group)
        try:
            group_auc = calc_auc(group)
            total_test_num += group_len
            category_average_auc += group_len * group_auc
        except:
            pass
    category_average_auc = category_average_auc / total_test_num

    accuracy_sum = accuracy_sum / iteration
    loss_sum = loss_sum / iteration
    result = {
        "auc": validation_auc,
        "loss": loss_sum,
        "accuracy": accuracy_sum,
        "total_val_num": len(stored_arr),
        "valid_val_num": total_test_num,
        "category_average_auc": category_average_auc,
        "eval_time": total_eval_time,
    }
    return result


def train(
        train_dataset,
        val_dataset,
        test_dataset,
        args,
        embedding_dim,
        learning_rate,
        weight_decay,
        seed
):
    setup_seed(seed)

    model_type = args.long_model_type
    if not os.path.exists(os.path.join(args.log_dir, 'model')):
        os.mkdir(os.path.join(args.log_dir, 'model'))
    latest_model_path = os.path.join(args.log_dir, 'model', 'latest_{}.pth'.format(model_type))
    best_model_path = os.path.join(args.log_dir, 'model', 'best_{}.pth'.format(model_type))
    test_interval = args.test_interval
    log_interval = args.log_interval
    batch_size = args.batch_size
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = create_model(model_type, args).to(device)
    for param_name, param in model.named_parameters():
        if 'bias' not in param_name:
            torch.nn.init.normal_(param, mean=0.0, std=0.01)
    if args.observe_attn_repr_grad or args.avoid_domination:
        model.time_id_embedding_layer_for_attention_of_target.weight = nn.Parameter(
            model.time_id_embedding_layer_for_representation_of_target.weight.clone())
        model.item_id_embedding_layer_for_attention_of_target.weight = nn.Parameter(
            model.item_id_embedding_layer_for_representation_of_target.weight.clone())
        model.category_id_embedding_layer_for_attention_of_target.weight = nn.Parameter(
            model.category_id_embedding_layer_for_representation_of_target.weight.clone())

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    sys.stdout.flush()
    logging.info('Training starts.')
    sys.stdout.flush()

    start_time = time.time()
    iteration = 0
    best_auc = 0.0
    loss_sum = 0.0
    accuracy_sum = 0.
    aux_loss_sum = 0.
    epoch = args.epoch
    eval_auc_list = []

    train_iterations = []
    test_iterations = []
    train_loss_record = []
    test_loss_record = []
    train_accuracy_record = []
    test_accuracy_record = []

    train_attn_category_grad_record = []
    train_attn_time_grad_record = []
    train_repr_category_grad_record = []
    train_repr_time_grad_record = []
    query_linear_grad_record = []
    key_linear_grad_record = []
    value_linear_grad_record = []

    for cur_epoch in range(epoch):
        logging.info("Epoch: " + str(cur_epoch))

        start_total_time = time.time()
        sum_forward_time = 0
        sum_total_time = 0

        train_dataset.reset()
        for batch_data in train_dataset:
            target = torch.tensor(batch_data['target'], dtype=torch.int32).to(device)
            start_forward_time = time.time()
            if args.use_aux_loss:
                predicted_probability, loss, aux_loss = model.forward(batch_data)
            else:
                predicted_probability, loss = model.forward(batch_data)
                aux_loss = torch.zeros([1])
            l2_penalty_loss = 0.0
            for param_name, param in model.named_parameters():
                if 'bias' not in param_name and 'embedding' not in param_name:  # TODO: check!
                    l2_penalty_loss += torch.sum(torch.pow(param, 2))
            loss += weight_decay * l2_penalty_loss

            # update params
            optimizer.zero_grad()
            loss.backward()
            forward_time = time.time() - start_forward_time

            if args.stor_grad:
                train_attn_time_grad_record.append(
                    model.time_id_embedding_layer_for_attention_of_target.weight.grad.detach().cpu().numpy())
                train_attn_category_grad_record.append(
                    model.category_id_embedding_layer_for_attention_of_target.weight.grad.detach().cpu().numpy())
                train_repr_time_grad_record.append(
                    model.time_id_embedding_layer_for_representation_of_target.weight.grad.detach().cpu().numpy())
                train_repr_category_grad_record.append(
                    model.category_id_embedding_layer_for_representation_of_target.weight.grad.detach().cpu().numpy())

                if args.model_name == "projection":
                    query_linear_grad_record.append(
                        model.mlp_for_attention_of_target.weight.grad.detach().cpu().numpy())
                    key_linear_grad_record.append(
                        model.mlp_for_attention_of_behavior.weight.grad.detach().cpu().numpy())
                    value_linear_grad_record.append(
                        model.mlp_for_representation_of_behavior.weight.grad.detach().cpu().numpy())

            if args.observe_attn_repr_grad:
                assert model.time_id_embedding_layer_for_attention_of_target.weight.equal(model.time_id_embedding_layer_for_representation_of_target.weight)
                assert model.item_id_embedding_layer_for_attention_of_target.weight.equal(model.item_id_embedding_layer_for_representation_of_target.weight)
                assert model.category_id_embedding_layer_for_attention_of_target.weight.equal(model.category_id_embedding_layer_for_representation_of_target.weight)

                model.time_id_embedding_layer_for_attention_of_target.weight.grad += model.time_id_embedding_layer_for_representation_of_target.weight.grad.clone()
                model.time_id_embedding_layer_for_representation_of_target.weight.grad = model.time_id_embedding_layer_for_attention_of_target.weight.grad.clone()
                model.item_id_embedding_layer_for_attention_of_target.weight.grad += model.item_id_embedding_layer_for_representation_of_target.weight.grad.clone()
                model.item_id_embedding_layer_for_representation_of_target.weight.grad = model.item_id_embedding_layer_for_attention_of_target.weight.grad.clone()
                model.category_id_embedding_layer_for_attention_of_target.weight.grad += model.category_id_embedding_layer_for_representation_of_target.weight.grad.clone()
                model.category_id_embedding_layer_for_representation_of_target.weight.grad = model.category_id_embedding_layer_for_attention_of_target.weight.grad.clone()

            if args.avoid_domination:
                assert model.time_id_embedding_layer_for_attention_of_target.weight.equal(model.time_id_embedding_layer_for_representation_of_target.weight)
                assert model.item_id_embedding_layer_for_attention_of_target.weight.equal(model.item_id_embedding_layer_for_representation_of_target.weight)
                assert model.category_id_embedding_layer_for_attention_of_target.weight.equal(model.category_id_embedding_layer_for_representation_of_target.weight)

                representation_time_norm = torch.norm(model.time_id_embedding_layer_for_representation_of_target.weight.grad.clone(), p=2)
                attention_time_norm = torch.norm(model.time_id_embedding_layer_for_attention_of_target.weight.grad.clone(), p=2)
                model.time_id_embedding_layer_for_attention_of_target.weight.grad *= representation_time_norm / attention_time_norm
                representation_item_norm = torch.norm(model.item_id_embedding_layer_for_representation_of_target.weight.grad.clone(), p=2)
                attention_item_norm = torch.norm(model.item_id_embedding_layer_for_attention_of_target.weight.grad.clone(), p=2)
                model.item_id_embedding_layer_for_attention_of_target.weight.grad *= representation_item_norm / attention_item_norm
                representation_cate_norm = torch.norm(model.category_id_embedding_layer_for_representation_of_target.weight.grad.clone(), p=2)
                attention_cate_norm = torch.norm(model.category_id_embedding_layer_for_attention_of_target.weight.grad.clone(), p=2)
                model.category_id_embedding_layer_for_attention_of_target.weight.grad *= representation_cate_norm / attention_cate_norm

                model.time_id_embedding_layer_for_attention_of_target.weight.grad += model.time_id_embedding_layer_for_representation_of_target.weight.grad.clone()
                model.time_id_embedding_layer_for_representation_of_target.weight.grad = model.time_id_embedding_layer_for_attention_of_target.weight.grad.clone()
                model.item_id_embedding_layer_for_attention_of_target.weight.grad += model.item_id_embedding_layer_for_representation_of_target.weight.grad.clone()
                model.item_id_embedding_layer_for_representation_of_target.weight.grad = model.item_id_embedding_layer_for_attention_of_target.weight.grad.clone()
                model.category_id_embedding_layer_for_attention_of_target.weight.grad += model.category_id_embedding_layer_for_representation_of_target.weight.grad.clone()
                model.category_id_embedding_layer_for_representation_of_target.weight.grad = model.category_id_embedding_layer_for_attention_of_target.weight.grad.clone()

            optimizer.step()

            loss_sum += loss.item()
            aux_loss_sum += aux_loss.item()
            # calculate accuracy
            with torch.no_grad():
                one_hot_predicted_result = torch.round(predicted_probability)
                correctness = torch.sum(torch.mul(one_hot_predicted_result, target), dim=1).float()
                accuracy = torch.mean(correctness)
            accuracy_sum += accuracy.item()

            iteration += 1

            sum_forward_time += forward_time
            sum_total_time += time.time() - start_total_time
            start_total_time = time.time()

            if (iteration % log_interval) == 0:
                test_time = time.time()
                info_string = 'Training: epoch=%d, iteration=%d, train_loss=%.4f, train_aux_loss=%.4f, train_accuracy=%.4f, total_time=%.4f ms, sess_time=%.4f ms, train_time=%.4f s, forward_time=%.4f ms' % (
                    cur_epoch, iteration,
                    loss_sum / log_interval,
                    aux_loss_sum / log_interval,
                    accuracy_sum / log_interval,
                    (1000 * sum_total_time) / (batch_size * log_interval),
                    sum_forward_time * 1000 / (batch_size * log_interval),
                    test_time - start_time,
                    forward_time)
                logging.info(info_string)
                train_iterations.append(iteration)
                train_loss_record.append((loss_sum - aux_loss_sum) / log_interval)
                train_accuracy_record.append(accuracy_sum / log_interval)
                loss_sum = 0.0
                accuracy_sum = 0.0
                aux_loss_sum = 0.
                sum_forward_time = 0

            if (iteration % test_interval) == 0:
                logging.info('Save latest model.')
                model.save(latest_model_path)

                val_dataset.reset()
                eval_result = evaluate(model, val_dataset, args.use_aux_loss)
                eval_result['epoch'] = cur_epoch
                eval_result['iteration'] = iteration
                test_info_string = 'Testing finishes: epoch={epoch}, iteration={iteration}, test_auc={auc:.4f}, test_loss={loss:.4f}, test_accuracy={accuracy:.4f}, eval_time={eval_time:.4f}'.format(
                    **eval_result)
                logging.info(test_info_string)
                test_iterations.append(iteration)
                test_loss_record.append(eval_result['loss'].item())
                test_accuracy_record.append(eval_result['accuracy'])

                if eval_result['auc'] > best_auc:
                    logging.info('Save best model.')
                    model.save(best_model_path)
                    best_auc = eval_result['auc']

                eval_auc_list.append(eval_result['auc'])

        logging.info("Epoch {0} train ends.".format(cur_epoch))

    if args.stor_grad:
        grad_save_path = os.path.join(args.log_dir, 'grad_during_training')
        if not os.path.exists(grad_save_path):
            os.mkdir(grad_save_path)
        attn_category_grad = np.stack(train_attn_category_grad_record)
        attn_time_grad = np.stack(train_attn_time_grad_record)
        repr_category_grad = np.stack(train_repr_category_grad_record)
        repr_time_grad = np.stack(train_repr_time_grad_record)
        np.save(os.path.join(grad_save_path, 'attn_category_grad.npy'), attn_category_grad)
        np.save(os.path.join(grad_save_path, 'attn_time_grad.npy'), attn_time_grad)
        np.save(os.path.join(grad_save_path, 'repr_category_grad.npy'), repr_category_grad)
        np.save(os.path.join(grad_save_path, 'repr_time_grad.npy'), repr_time_grad)
        if args.model_name == "projection":
            query_grad = np.stack(query_linear_grad_record)
            key_grad = np.stack(key_linear_grad_record)
            value_grad = np.stack(value_linear_grad_record)
            np.save(os.path.join(grad_save_path, 'query_grad.npy'), query_grad)
            np.save(os.path.join(grad_save_path, 'key_grad.npy'), key_grad)
            np.save(os.path.join(grad_save_path, 'value_grad.npy'), value_grad)

    process_save_path = os.path.join(args.log_dir, 'performance_during_training')
    if not os.path.exists(process_save_path):
        os.mkdir(process_save_path)

    with open(os.path.join(process_save_path, "record.txt"), "a") as file:
        file.write("train_iterations:\n")
        file.write(str(train_iterations))
        file.write("\ntrain_accuracy:\n")
        file.write(str(train_accuracy_record))
        file.write("\ntrain_loss:\n")
        file.write(str(train_loss_record))
        file.write("\ntest_iterations\n")
        file.write(str(test_iterations))
        file.write("\ntest_accuracy:\n")
        file.write(str(test_accuracy_record))
        file.write("\ntest_loss:\n")
        file.write(str(test_loss_record))

    logging.info(" ".join(['{:.4f}'.format(x) for x in eval_auc_list]))
    model.load(best_model_path)
    model.to(device)
    test_dataset.reset()
    test_result = evaluate(model, test_dataset, args.use_aux_loss)
    final_test_auc_string = 'Final test auc: {:.4f}.'.format(test_result['auc'])
    final_test_logloss_string = 'Final test logloss: {:.4f}'.format(test_result['loss'])
    final_test_total_num_string = 'Final test num: {:d}/{:d}'.format(test_result['valid_val_num'],
                                                                     test_result['total_val_num'])
    final_test_category_average_auc_string = 'Final test category average auc: {:.4f}.'.format(
        test_result['category_average_auc'])
    logging.info(final_test_auc_string)
    logging.info(final_test_logloss_string)
    logging.info(final_test_total_num_string)
    logging.info(final_test_category_average_auc_string)
    print(final_test_auc_string)
    print(final_test_logloss_string)

    if model_type == "DIN":
        os.makedirs(os.path.join(args.log_dir, 'learned_embedding'), exist_ok=True)
        attention_for_target_time_embedding_weights = model.time_id_embedding_layer_for_attention_of_target.weight.detach().cpu().numpy()
        np.save(os.path.join(args.log_dir, 'learned_embedding', 'attention_for_target_time.npy'), attention_for_target_time_embedding_weights)
        attention_for_target_item_embedding_weights = model.item_id_embedding_layer_for_attention_of_target.weight.detach().cpu().numpy()
        np.save(os.path.join(args.log_dir, 'learned_embedding', 'attention_for_target_item.npy'), attention_for_target_item_embedding_weights)
        attention_for_target_category_embedding_weights = model.category_id_embedding_layer_for_attention_of_target.weight.detach().cpu().numpy()
        np.save(os.path.join(args.log_dir, 'learned_embedding', 'attention_for_target_category.npy'), attention_for_target_category_embedding_weights)
        representation_for_target_time_embedding_weights = model.time_id_embedding_layer_for_representation_of_target.weight.detach().cpu().numpy()
        np.save(os.path.join(args.log_dir, 'learned_embedding', 'representation_for_target_time.npy'), representation_for_target_time_embedding_weights)
        representation_for_target_item_embedding_weights = model.item_id_embedding_layer_for_representation_of_target.weight.detach().cpu().numpy()
        np.save(os.path.join(args.log_dir, 'learned_embedding', 'representation_for_target_item.npy'), representation_for_target_item_embedding_weights)
        representation_for_target_category_embedding_weights = model.category_id_embedding_layer_for_representation_of_target.weight.detach().cpu().numpy()
        np.save(os.path.join(args.log_dir, 'learned_embedding', 'representation_for_target_category.npy'), representation_for_target_category_embedding_weights)
        attention_for_behavior_time_embedding_weights = model.time_id_embedding_layer_for_attention_of_behavior.weight.detach().cpu().numpy()
        np.save(os.path.join(args.log_dir, 'learned_embedding', 'attention_for_behavior_time.npy'), attention_for_behavior_time_embedding_weights)
        attention_for_behavior_item_embedding_weights = model.item_id_embedding_layer_for_attention_of_behavior.weight.detach().cpu().numpy()
        np.save(os.path.join(args.log_dir, 'learned_embedding', 'attention_for_behavior_item.npy'), attention_for_behavior_item_embedding_weights)
        attention_for_behavior_category_embedding_weights = model.category_id_embedding_layer_for_attention_of_behavior.weight.detach().cpu().numpy()
        np.save(os.path.join(args.log_dir, 'learned_embedding', 'attention_for_behavior_category.npy'), attention_for_behavior_category_embedding_weights)
        representation_for_behavior_time_embedding_weights = model.time_id_embedding_layer_for_representation_of_behavior.weight.detach().cpu().numpy()
        np.save(os.path.join(args.log_dir, 'learned_embedding', 'representation_for_behavior_time.npy'), representation_for_behavior_time_embedding_weights)
        representation_for_behavior_item_embedding_weights = model.item_id_embedding_layer_for_representation_of_behavior.weight.detach().cpu().numpy()
        np.save(os.path.join(args.log_dir, 'learned_embedding', 'representation_for_behavior_item.npy'), representation_for_behavior_item_embedding_weights)
        representation_for_behavior_category_embedding_weights = model.category_id_embedding_layer_for_representation_of_behavior.weight.detach().cpu().numpy()
        np.save(os.path.join(args.log_dir, 'learned_embedding', 'representation_for_behavior_category.npy'), representation_for_behavior_category_embedding_weights)
        if args.model_name == "projection":
            if args.mlp_position_after_concat:
                query_mlp = model.mlp_for_attention_of_target.weight.detach().cpu().numpy()
                np.save(os.path.join(args.log_dir, 'learned_embedding', 'q_mlp.npy'), query_mlp)
                key_mlp = model.mlp_for_attention_of_behavior.weight.detach().cpu().numpy()
                np.save(os.path.join(args.log_dir, 'learned_embedding', 'k_mlp.npy'), key_mlp)
                u_mlp = model.mlp_for_representation_of_target.weight.detach().cpu().numpy()
                np.save(os.path.join(args.log_dir, 'learned_embedding', 'u_mlp.npy'), u_mlp)
                value_mlp = model.mlp_for_representation_of_behavior.weight.detach().cpu().numpy()
                np.save(os.path.join(args.log_dir, 'learned_embedding', 'v_mlp.npy'), value_mlp)
            else:
                query_time_mlp = model.mlp_for_time_attention_of_target.weight.detach().cpu().numpy()
                np.save(os.path.join(args.log_dir, 'learned_embedding', 'q_time_mlp.npy'), query_time_mlp)
                query_category_mlp = model.mlp_for_category_attention_of_target.weight.detach().cpu().numpy()
                np.save(os.path.join(args.log_dir, 'learned_embedding', 'q_category_mlp.npy'), query_category_mlp)

                key_time_mlp = model.mlp_for_time_attention_of_behavior.weight.detach().cpu().numpy()
                np.save(os.path.join(args.log_dir, 'learned_embedding', 'k_time_mlp.npy'), key_time_mlp)
                key_category_mlp = model.mlp_for_category_attention_of_behavior.weight.detach().cpu().numpy()
                np.save(os.path.join(args.log_dir, 'learned_embedding', 'k_category_mlp.npy'), key_category_mlp)

                u_time_mlp = model.mlp_for_time_representation_of_target.weight.detach().cpu().numpy()
                np.save(os.path.join(args.log_dir, 'learned_embedding', 'u_time_mlp.npy'), u_time_mlp)
                u_category_mlp = model.mlp_for_category_representation_of_target.weight.detach().cpu().numpy()
                np.save(os.path.join(args.log_dir, 'learned_embedding', 'u_category_mlp.npy'), u_category_mlp)

                value_time_mlp = model.mlp_for_time_representation_of_behavior.weight.detach().cpu().numpy()
                np.save(os.path.join(args.log_dir, 'learned_embedding', 'v_time_mlp.npy'), value_time_mlp)
                value_category_mlp = model.mlp_for_category_representation_of_behavior.weight.detach().cpu().numpy()
                np.save(os.path.join(args.log_dir, 'learned_embedding', 'v_category_mlp.npy'), value_category_mlp)

    return test_result['auc'], test_result['loss']


def main():
    args = get_args()
    logger = CompleteLogger(args.log_dir)
    logging.basicConfig(format="[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d] %(message)s",
                        level=args.level, stream=sys.stderr)
    logging.info(args)

    # prepare data
    train_dataset_path = args.train_dataset_path
    test_dataset_path = args.test_dataset_path
    batch_size = args.batch_size

    train_dataset = SeqRecDataset(
        train_dataset_path,
        batch_size,
        max_length=args.max_length,
        apply_hard_search=(args.hard_or_soft == 'hard')
    )
    test_dataset = SeqRecDataset(
        test_dataset_path,
        batch_size,
        max_length=args.max_length,
        apply_hard_search=(args.hard_or_soft == 'hard')
    )
    val_dataset = SeqRecDataset(
        args.val_dataset_path,
        batch_size,
        max_length=args.max_length,
        apply_hard_search=(args.hard_or_soft == 'hard')
    )

    grid_search = []
    for seed in args.seed:
        test_auc, test_logloss = train(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
            args=args,
            embedding_dim=args.category_embedding_dim,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=seed
        )
        grid_search.append({
            'embedding_dim': args.category_embedding_dim,
            'learning_rate': args.learning_rate,
            'weight_decay': args.weight_decay,
            'seed': seed,
            'test_auc': test_auc,
            'test_logloss': test_logloss
        })

    np.save(os.path.join(args.log_dir, 'grid_search.npy'), grid_search)
    logger.close()


if __name__ == '__main__':
    main()
