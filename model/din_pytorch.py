import torch
import torch.nn as nn
import numpy as np

from .base_model_pytorch import BaseModel
from .attention_pytorch import attention


class DIN(BaseModel):
    def __init__(self,
                 *args,
                 soft_search=False,
                 top_k=10,
                 **kwargs,
                 ):
        super(DIN, self).__init__(*args, **kwargs)
        self.soft_search = soft_search
        self.top_k = top_k

    def forward(self, batch_data, mode="train", only_return_representation=False, only_return_scores=False):
        # get data
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        item_ids = torch.tensor(batch_data['iid'], dtype=torch.int32).to(device)
        category_ids = torch.tensor(batch_data['cid'], dtype=torch.int32).to(device)
        time_ids = torch.tensor(batch_data['tid'], dtype=torch.int32).to(device)
        history_item_ids = torch.tensor(batch_data['hist_iid'], dtype=torch.int32).to(device)
        history_category_ids = torch.tensor(batch_data['hist_cid'], dtype=torch.int32).to(device)
        history_time_ids = torch.tensor(batch_data['hist_tid'], dtype=torch.int32).to(device)
        mask = batch_data['mask']
        if not self.soft_search:
            num_ones = np.sum(mask, axis=1)
            for i in range(mask.shape[0]):
                if num_ones[i] > self.top_k:
                    one_incides = np.where(mask[i] == 1)[0]
                    aim_position = one_incides[round(num_ones[i] - self.top_k)]
                    mask[i][:aim_position] = 0
                elif num_ones[i] < self.top_k:
                    zero_incides = np.where(mask[i] == 0)[0]
                    aim_position = zero_incides[round(-(self.top_k - num_ones[i]))]
                    mask[i][aim_position:] = 1
            assert np.all(np.sum(mask, axis=1) == self.top_k)
        mask = torch.tensor(mask, dtype=torch.int32).to(device)
        target = torch.tensor(batch_data['target'], dtype=torch.int32).to(device)

        # embedding
        embedded_item_for_attention_of_target = self.item_id_embedding_layer_for_attention_of_target(item_ids)
        embedded_category_for_attention_of_target = self.category_id_embedding_layer_for_attention_of_target(category_ids)
        embedded_time_for_attention_of_target = self.time_id_embedding_layer_for_attention_of_target(time_ids)

        embedded_item_for_representation_of_target = self.item_id_embedding_layer_for_representation_of_target(item_ids)
        embedded_category_for_representation_of_target = self.category_id_embedding_layer_for_representation_of_target(category_ids)
        embedded_time_for_representation_of_target = self.time_id_embedding_layer_for_representation_of_target(time_ids)

        embedded_item_for_attention_of_behavior = self.item_id_embedding_layer_for_attention_of_behavior(history_item_ids)
        embedded_category_for_attention_of_behavior = self.category_id_embedding_layer_for_attention_of_behavior(history_category_ids)
        embedded_time_for_attention_of_behavior = self.time_id_embedding_layer_for_attention_of_behavior(history_time_ids)

        embedded_item_for_representation_of_behavior = self.item_id_embedding_layer_for_representation_of_behavior(history_item_ids)
        embedded_category_for_representation_of_behavior = self.category_id_embedding_layer_for_representation_of_behavior(history_category_ids)
        embedded_time_for_representation_of_behavior = self.time_id_embedding_layer_for_representation_of_behavior(history_time_ids)

        # mlp_transform
        if not self.mlp_position_after_concat:
            if self.mlp_for_item_attention_of_target is not None:
                embedded_item_for_attention_of_target_result = self.mlp_for_item_attention_of_target(embedded_item_for_attention_of_target)
            else:
                embedded_item_for_attention_of_target_result = embedded_item_for_attention_of_target

            if self.mlp_for_category_attention_of_target is not None:
                embedded_category_for_attention_of_target_result = self.mlp_for_category_attention_of_target(embedded_category_for_attention_of_target)
            else:
                embedded_category_for_attention_of_target_result = embedded_category_for_attention_of_target

            if self.mlp_for_time_attention_of_target is not None:
                embedded_time_for_attention_of_target_result = self.mlp_for_time_attention_of_target(embedded_time_for_attention_of_target)
            else:
                embedded_time_for_attention_of_target_result = embedded_time_for_attention_of_target

            if self.mlp_for_item_representation_of_target is not None:
                embedded_item_for_representation_of_target_result = self.mlp_for_item_representation_of_target(embedded_item_for_representation_of_target)
            else:
                embedded_item_for_representation_of_target_result = embedded_item_for_representation_of_target

            if self.mlp_for_category_representation_of_target is not None:
                embedded_category_for_representation_of_target_result = self.mlp_for_category_representation_of_target(embedded_category_for_representation_of_target)
            else:
                embedded_category_for_representation_of_target_result = embedded_category_for_representation_of_target

            if self.mlp_for_time_representation_of_target is not None:
                embedded_time_for_representation_of_target_result = self.mlp_for_time_representation_of_target(embedded_time_for_representation_of_target)
            else:
                embedded_time_for_representation_of_target_result = embedded_time_for_representation_of_target

            if self.mlp_for_item_attention_of_behavior is not None:
                embedded_item_for_attention_of_behavior_result = self.mlp_for_item_attention_of_behavior(embedded_item_for_attention_of_behavior)
            else:
                embedded_item_for_attention_of_behavior_result = embedded_item_for_attention_of_behavior

            if self.mlp_for_category_attention_of_behavior is not None:
                embedded_category_for_attention_of_behavior_result = self.mlp_for_category_attention_of_behavior(embedded_category_for_attention_of_behavior)
            else:
                embedded_category_for_attention_of_behavior_result = embedded_category_for_attention_of_behavior

            if self.mlp_for_time_attention_of_behavior is not None:
                embedded_time_for_attention_of_behavior_result = self.mlp_for_time_attention_of_behavior(embedded_time_for_attention_of_behavior)
            else:
                embedded_time_for_attention_of_behavior_result = embedded_time_for_attention_of_behavior

            if self.mlp_for_item_representation_of_behavior is not None:
                embedded_item_for_representation_of_behavior_result = self.mlp_for_item_representation_of_behavior(embedded_item_for_representation_of_behavior)
            else:
                embedded_item_for_representation_of_behavior_result = embedded_item_for_representation_of_behavior

            if self.mlp_for_category_representation_of_behavior is not None:
                embedded_category_for_representation_of_behavior_result = self.mlp_for_category_representation_of_behavior(embedded_category_for_representation_of_behavior)
            else:
                embedded_category_for_representation_of_behavior_result = embedded_category_for_representation_of_behavior

            if self.mlp_for_time_representation_of_behavior is not None:
                embedded_time_for_representation_of_behavior_result = self.mlp_for_time_representation_of_behavior(embedded_time_for_representation_of_behavior)
            else:
                embedded_time_for_representation_of_behavior_result = embedded_time_for_representation_of_behavior

            if self.use_time:
                if self.use_time_mode == "concat":
                    result_for_attention_of_target = torch.cat((embedded_item_for_attention_of_target_result, embedded_category_for_attention_of_target_result, embedded_time_for_attention_of_target_result), dim=1)
                    result_for_representation_of_target = torch.cat((embedded_item_for_representation_of_target_result, embedded_category_for_representation_of_target_result, embedded_time_for_representation_of_target_result), dim=1)
                    result_for_attention_of_behavior = torch.cat((embedded_item_for_attention_of_behavior_result, embedded_category_for_attention_of_behavior_result, embedded_time_for_attention_of_behavior_result), dim=2)
                    result_for_representation_of_behavior = torch.cat((embedded_item_for_representation_of_behavior_result, embedded_category_for_representation_of_behavior_result, embedded_time_for_representation_of_behavior_result), dim=2)
                else:
                    result_for_attention_of_target = torch.cat((embedded_item_for_attention_of_target_result, embedded_category_for_attention_of_target_result), dim=1) + embedded_time_for_attention_of_target_result
                    result_for_representation_of_target = torch.cat((embedded_item_for_representation_of_target_result, embedded_category_for_representation_of_target_result), dim=1) + embedded_time_for_representation_of_target_result
                    result_for_attention_of_behavior = torch.cat((embedded_item_for_attention_of_behavior_result, embedded_category_for_attention_of_behavior_result), dim=2) + embedded_time_for_attention_of_behavior_result
                    result_for_representation_of_behavior = torch.cat((embedded_item_for_representation_of_behavior_result, embedded_category_for_representation_of_behavior_result), dim=2) + embedded_time_for_representation_of_behavior_result
            else:
                result_for_attention_of_target = torch.cat((embedded_item_for_attention_of_target_result, embedded_category_for_attention_of_target_result), dim=1)
                result_for_representation_of_target = torch.cat((embedded_item_for_representation_of_target_result, embedded_category_for_representation_of_target_result), dim=1)
                result_for_attention_of_behavior = torch.cat((embedded_item_for_attention_of_behavior_result, embedded_category_for_attention_of_behavior_result), dim=2)
                result_for_representation_of_behavior = torch.cat((embedded_item_for_representation_of_behavior_result, embedded_category_for_representation_of_behavior_result), dim=2)
        else:
            if self.use_time:
                if self.use_time_mode == "concat":
                    embedded_result_for_attention_of_target = torch.cat((embedded_item_for_attention_of_target, embedded_category_for_attention_of_target, embedded_time_for_attention_of_target), dim=1)
                    embedded_result_for_representation_of_target = torch.cat((embedded_item_for_representation_of_target, embedded_category_for_representation_of_target, embedded_time_for_representation_of_target), dim=1)
                    embedded_result_for_attention_of_behavior = torch.cat((embedded_item_for_attention_of_behavior, embedded_category_for_attention_of_behavior, embedded_time_for_attention_of_behavior), dim=2)
                    embedded_result_for_representation_of_behavior = torch.cat((embedded_item_for_representation_of_behavior, embedded_category_for_representation_of_behavior, embedded_time_for_representation_of_behavior), dim=2)
                else:
                    embedded_result_for_attention_of_target = torch.cat((embedded_item_for_attention_of_target, embedded_category_for_attention_of_target), dim=1) + embedded_time_for_attention_of_target
                    embedded_result_for_representation_of_target = torch.cat((embedded_item_for_representation_of_target, embedded_category_for_representation_of_target), dim=1) + embedded_time_for_representation_of_target
                    embedded_result_for_attention_of_behavior = torch.cat((embedded_item_for_attention_of_behavior, embedded_category_for_attention_of_behavior), dim=2) + embedded_time_for_attention_of_behavior
                    embedded_result_for_representation_of_behavior = torch.cat((embedded_item_for_representation_of_behavior, embedded_category_for_representation_of_behavior), dim=2) + embedded_time_for_representation_of_behavior
            else:
                embedded_result_for_attention_of_target = torch.cat((embedded_item_for_attention_of_target, embedded_category_for_attention_of_target), dim=1)
                embedded_result_for_representation_of_target = torch.cat((embedded_item_for_representation_of_target, embedded_category_for_representation_of_target), dim=1)
                embedded_result_for_attention_of_behavior = torch.cat((embedded_item_for_attention_of_behavior, embedded_category_for_attention_of_behavior), dim=2)
                embedded_result_for_representation_of_behavior = torch.cat((embedded_item_for_representation_of_behavior, embedded_category_for_representation_of_behavior), dim=2)

            if self.mlp_for_attention_of_target is not None:
                result_for_attention_of_target = self.mlp_for_attention_of_target(embedded_result_for_attention_of_target)
            else:
                result_for_attention_of_target = embedded_result_for_attention_of_target

            if self.mlp_for_representation_of_target is not None:
                result_for_representation_of_target = self.mlp_for_representation_of_target(embedded_result_for_representation_of_target)
            else:
                result_for_representation_of_target = embedded_result_for_representation_of_target

            if self.mlp_for_attention_of_behavior is not None:
                result_for_attention_of_behavior = self.mlp_for_attention_of_behavior(embedded_result_for_attention_of_behavior)
            else:
                result_for_attention_of_behavior = embedded_result_for_attention_of_behavior

            if self.mlp_for_representation_of_behavior is not None:
                result_for_representation_of_behavior = self.mlp_for_representation_of_behavior(embedded_result_for_representation_of_behavior)
            else:
                result_for_representation_of_behavior = embedded_result_for_representation_of_behavior

        assert len(result_for_attention_of_target.shape) == 2
        assert len(result_for_representation_of_target.shape) == 2
        assert len(result_for_attention_of_behavior.shape) == 3
        assert len(result_for_representation_of_behavior.shape) == 3

        if not self.only_odot:
            features = [result_for_representation_of_target]
        else:
            features = []
        if self.short_seq_split:
            seq_split = [(int(x.split(":")[0]), int(x.split(":")[1])) for x in self.short_seq_split.split(",")]
            for idx, (left_idx, right_idx) in enumerate(seq_split):
                short_seq_mask = mask[:, left_idx:right_idx]
                short_seq_embed = attention(result_for_attention_of_target, result_for_attention_of_behavior[:, left_idx:right_idx], result_for_representation_of_behavior[:, left_idx:right_idx], short_seq_mask, attn_func=self.attn_func, attn_model=self.attention_model, time_embedding_dim=self.attention_time_embedding_dim, use_time_mode=self.use_time_mode, model_name=self.model_name, ETA_H_metirc=self.ETA_H_metrics)
                if not self.only_odot:
                    features.append(short_seq_embed)
                if self.use_cross_feature:
                    features.append(torch.mul(result_for_representation_of_target, short_seq_embed))

        if self.long_seq_split:
            seq_split = [(int(x.split(":")[0]), int(x.split(":")[1])) for x in self.long_seq_split.split(",")]
            for idx, (left_idx, right_idx) in enumerate(seq_split):
                long_seq_mask = mask[:, left_idx:right_idx]
                long_seq_average_embed = torch.sum(torch.mul(result_for_representation_of_behavior[:, left_idx:right_idx], long_seq_mask.unsqueeze(2)), dim=1) / (torch.sum(long_seq_mask, dim=1, keepdim=True) + 1.0)

                long_seq_embed, original_attn_scores, attention_distribution = attention(result_for_attention_of_target, result_for_attention_of_behavior[:, left_idx:right_idx], result_for_representation_of_behavior[:, left_idx:right_idx], long_seq_mask, attn_func=self.attn_func, attn_model=self.attention_model, time_embedding_dim=self.attention_time_embedding_dim, return_scores=True, use_time_mode=self.use_time_mode, soft_search=self.soft_search)
                if only_return_scores:
                    return original_attn_scores
                # get aux_loss by auxiliary_mlp
                if self.use_cross_feature:
                    if self.only_odot:
                        aux_input_embed = torch.mul(result_for_representation_of_target, long_seq_embed)
                    else:
                        aux_input_embed = torch.cat((result_for_representation_of_target, long_seq_embed,
                                                     torch.mul(result_for_representation_of_target, long_seq_embed)),
                                                    dim=1)
                else:
                    aux_input_embed = torch.cat((result_for_representation_of_target, long_seq_embed), dim=1)
                assert aux_input_embed.shape == (item_ids.shape[0], self.auxiliary_mlp_input_dimension)
                aux_y_hat = self.auxiliary_mlp(aux_input_embed)
                assert aux_y_hat.shape == (item_ids.shape[0], 2)
                aux_predicted_probability = nn.functional.softmax(aux_y_hat, dim=1) + 1e-8
                aux_loss = -torch.mean(torch.log(torch.sum(torch.mul(aux_predicted_probability, target), dim=1)))

                if self.soft_search:
                    top_k_scores, top_k_indices = torch.topk(original_attn_scores, self.top_k, dim=1)
                    mask_for_soft_search = torch.zeros_like(long_seq_mask)
                    for k in range(top_k_indices.shape[0]):
                        mask_for_soft_search[k, top_k_indices[k]] = 1
                    assert torch.all(torch.sum(mask_for_soft_search, dim=1) == self.top_k)
                    _, scores_for_soft_search, _ = attention(result_for_attention_of_target, result_for_attention_of_behavior[:, left_idx:right_idx], result_for_representation_of_behavior[:, left_idx:right_idx], mask_for_soft_search, attn_func=self.attn_func, attn_model=self.attention_model, time_embedding_dim=self.attention_time_embedding_dim, return_scores=True, use_time_mode=self.use_time_mode)

                    attention_distribution_for_soft_search = nn.functional.softmax(scores_for_soft_search, dim=1)
                    soft_search_long_seq_embed = torch.matmul(attention_distribution_for_soft_search.unsqueeze(1), result_for_representation_of_behavior[:, left_idx:right_idx])
                    long_seq_embed = soft_search_long_seq_embed.squeeze(1)

                if not self.only_odot:
                    features.append(long_seq_embed)
                    if self.use_long_seq_average:
                        features.append(long_seq_average_embed)
                if self.use_cross_feature:
                    features.append(torch.mul(result_for_representation_of_target, long_seq_embed))
                    if self.use_long_seq_average:
                        features.append(torch.mul(result_for_representation_of_target, long_seq_average_embed))

        # logging.info(features)
        features = torch.cat(features, dim=1)
        if only_return_representation:
            return features
        main_predicted_y = self.main_mlp(features)
        main_predicted_probility = nn.functional.softmax(main_predicted_y, dim=1) + 1e-8

        ctr_loss = -torch.mean(torch.log(torch.sum(torch.mul(main_predicted_probility, target), dim=1)))

        if self.use_aux_loss:
            return main_predicted_probility, ctr_loss+aux_loss, aux_loss
        else:
            return main_predicted_probility, ctr_loss
