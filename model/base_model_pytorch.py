import torch
import torch.nn as nn

from .attention_pytorch import AttentionModel


class BaseModel(nn.Module):
    def __init__(self,
                 item_n=-1,
                 cate_n=-1,
                 category_embedding_dim=16,
                 batch_size=2048,
                 max_length=200,
                 use_cross_feature=False,
                 attn_func='learnable',
                 use_aux_loss=False,
                 use_time=True,
                 use_time_mode='concat',
                 time_embedding_dim=16,
                 short_seq_split=None,
                 long_seq_split=None,
                 use_long_seq_average=True,
                 model_name=None,
                 item_embedding_dim=-1,
                 mlp_position_after_concat=False,
                 attention_category_embedding_dim=-1,
                 attention_time_embedding_dim=-1,
                 attention_item_embedding_dim=-1,
                 only_odot=False,
                 mlp_hidden_layer=None,
                 main_mlp_hidden_1=200,
                 main_mlp_hidden_2=80,
                 observe_attn_repr_grad=False,
                 no_batch_norm_if_one_batchsize=True,
                 ):
        super().__init__()
        self.use_cross_feature = use_cross_feature
        self.attn_func = attn_func
        self.use_aux_loss = use_aux_loss
        self.batch_size = batch_size
        self.max_length = max_length
        self.use_time = use_time
        self.use_time_mode = use_time_mode
        self.short_seq_split = short_seq_split
        self.long_seq_split = long_seq_split
        self.category_embedding_dim = category_embedding_dim
        if item_embedding_dim == -1:
            item_embedding_dim = category_embedding_dim
        self.item_embedding_dim = item_embedding_dim
        self.time_embedding_dim = time_embedding_dim
        self.use_long_seq_average = use_long_seq_average
        self.model_name = model_name
        self.mlp_position_after_concat = mlp_position_after_concat
        if attention_time_embedding_dim != -1:
            self.attention_time_embedding_dim = attention_time_embedding_dim
        else:
            self.attention_time_embedding_dim = self.time_embedding_dim
        if attention_category_embedding_dim != -1:
            self.attention_category_embedding_dim = attention_category_embedding_dim
        else:
            self.attention_category_embedding_dim = self.category_embedding_dim
        if attention_item_embedding_dim != -1:
            self.attention_item_embedding_dim = attention_item_embedding_dim
        else:
            self.attention_item_embedding_dim = self.item_embedding_dim
        if attention_item_embedding_dim != -1 or attention_category_embedding_dim != -1 or attention_time_embedding_dim != -1:
            assert model_name in ['DARE', 'three_embedding', 'four_embedding']
        if self.use_time and self.use_time_mode == "add":
            assert self.item_embedding_dim + self.category_embedding_dim == self.time_embedding_dim
        if model_name in ['DARE', 'three_embedding', 'four_embedding', 'twin', 'projection',
                          'projection_two_linear', 'projection_with_mlp', 'projection_no_item',
                          'projection_decouple_by_category', 'projection_only_time', 'projection_only_category',
                          'projection_only_item', 'projection_large_emb_small_output']:
            assert self.attn_func == 'scaled_dot_product'
        elif model_name in ['DIN']:
            assert self.attn_func == 'learnable'
            assert self.use_time == False
        else:
            assert False , f'model_name {model_name} not expected'
        self.only_odot = only_odot
        if only_odot:
            assert self.use_cross_feature

        # embedding for (attention/representation) of (behaviour/target)
        if model_name in ["twin", "projection", "DIN", 'projection_two_linear',
                          'projection_with_mlp', 'projection_no_item',
                          'projection_only_time', 'projection_only_category', 'projection_only_item',
                          'projection_large_emb_small_output'] and not observe_attn_repr_grad:
            self.item_id_embedding_layer_for_attention_of_target = nn.Embedding(item_n, item_embedding_dim)
            self.category_id_embedding_layer_for_attention_of_target = nn.Embedding(cate_n, category_embedding_dim)
            self.time_id_embedding_layer_for_attention_of_target = nn.Embedding(2000, time_embedding_dim)

            self.item_id_embedding_layer_for_representation_of_target = self.item_id_embedding_layer_for_attention_of_target
            self.category_id_embedding_layer_for_representation_of_target = self.category_id_embedding_layer_for_attention_of_target
            self.time_id_embedding_layer_for_representation_of_target = self.time_id_embedding_layer_for_attention_of_target

            self.item_id_embedding_layer_for_attention_of_behavior = self.item_id_embedding_layer_for_attention_of_target
            self.category_id_embedding_layer_for_attention_of_behavior = self.category_id_embedding_layer_for_attention_of_target
            self.time_id_embedding_layer_for_attention_of_behavior = self.time_id_embedding_layer_for_attention_of_target

            self.item_id_embedding_layer_for_representation_of_behavior = self.item_id_embedding_layer_for_attention_of_target
            self.category_id_embedding_layer_for_representation_of_behavior = self.category_id_embedding_layer_for_attention_of_target
            self.time_id_embedding_layer_for_representation_of_behavior = self.time_id_embedding_layer_for_attention_of_target
        elif model_name == "DARE" or observe_attn_repr_grad:
            self.item_id_embedding_layer_for_attention_of_target = nn.Embedding(item_n,
                                                                                self.attention_item_embedding_dim)
            self.category_id_embedding_layer_for_attention_of_target = nn.Embedding(cate_n,
                                                                                    self.attention_category_embedding_dim)
            self.time_id_embedding_layer_for_attention_of_target = nn.Embedding(2000, self.attention_time_embedding_dim)

            self.item_id_embedding_layer_for_representation_of_target = nn.Embedding(item_n, item_embedding_dim)
            self.category_id_embedding_layer_for_representation_of_target = nn.Embedding(cate_n, category_embedding_dim)
            self.time_id_embedding_layer_for_representation_of_target = nn.Embedding(2000, time_embedding_dim)

            self.item_id_embedding_layer_for_attention_of_behavior = self.item_id_embedding_layer_for_attention_of_target
            self.category_id_embedding_layer_for_attention_of_behavior = self.category_id_embedding_layer_for_attention_of_target
            self.time_id_embedding_layer_for_attention_of_behavior = self.time_id_embedding_layer_for_attention_of_target

            self.item_id_embedding_layer_for_representation_of_behavior = self.item_id_embedding_layer_for_representation_of_target
            self.category_id_embedding_layer_for_representation_of_behavior = self.category_id_embedding_layer_for_representation_of_target
            self.time_id_embedding_layer_for_representation_of_behavior = self.time_id_embedding_layer_for_representation_of_target
        elif model_name == "four_embedding":
            self.item_id_embedding_layer_for_attention_of_target = nn.Embedding(item_n,
                                                                                self.attention_item_embedding_dim)
            self.category_id_embedding_layer_for_attention_of_target = nn.Embedding(cate_n,
                                                                                    self.attention_category_embedding_dim)
            self.time_id_embedding_layer_for_attention_of_target = nn.Embedding(2000, self.attention_time_embedding_dim)

            self.item_id_embedding_layer_for_representation_of_target = nn.Embedding(item_n, item_embedding_dim)
            self.category_id_embedding_layer_for_representation_of_target = nn.Embedding(cate_n, category_embedding_dim)
            self.time_id_embedding_layer_for_representation_of_target = nn.Embedding(2000, time_embedding_dim)

            self.item_id_embedding_layer_for_attention_of_behavior = nn.Embedding(item_n,
                                                                                  self.attention_item_embedding_dim)
            self.category_id_embedding_layer_for_attention_of_behavior = nn.Embedding(cate_n,
                                                                                      self.attention_category_embedding_dim)
            self.time_id_embedding_layer_for_attention_of_behavior = nn.Embedding(2000,
                                                                                  self.attention_time_embedding_dim)

            self.item_id_embedding_layer_for_representation_of_behavior = nn.Embedding(item_n, item_embedding_dim)
            self.category_id_embedding_layer_for_representation_of_behavior = nn.Embedding(cate_n,
                                                                                           category_embedding_dim)
            self.time_id_embedding_layer_for_representation_of_behavior = nn.Embedding(2000, time_embedding_dim)
        elif model_name == "three_embedding":
            self.item_id_embedding_layer_for_attention_of_target = nn.Embedding(item_n,
                                                                                self.attention_item_embedding_dim)
            self.category_id_embedding_layer_for_attention_of_target = nn.Embedding(cate_n,
                                                                                    self.attention_category_embedding_dim)
            self.time_id_embedding_layer_for_attention_of_target = nn.Embedding(2000, self.attention_time_embedding_dim)

            self.item_id_embedding_layer_for_representation_of_target = nn.Embedding(item_n, item_embedding_dim)
            self.category_id_embedding_layer_for_representation_of_target = nn.Embedding(cate_n, category_embedding_dim)
            self.time_id_embedding_layer_for_representation_of_target = nn.Embedding(2000, time_embedding_dim)

            self.item_id_embedding_layer_for_attention_of_behavior = nn.Embedding(item_n,
                                                                                  self.attention_item_embedding_dim)
            self.category_id_embedding_layer_for_attention_of_behavior = nn.Embedding(cate_n,
                                                                                      self.attention_category_embedding_dim)
            self.time_id_embedding_layer_for_attention_of_behavior = nn.Embedding(2000,
                                                                                  self.attention_time_embedding_dim)

            self.item_id_embedding_layer_for_representation_of_behavior = self.item_id_embedding_layer_for_representation_of_target
            self.category_id_embedding_layer_for_representation_of_behavior = self.category_id_embedding_layer_for_representation_of_target
            self.time_id_embedding_layer_for_representation_of_behavior = self.time_id_embedding_layer_for_representation_of_target
        else:
            assert False, "unexpected case: model_name is {}".format(model_name)

        if model_name == 'projection_large_emb_small_output':
            embedding_result_dimension = 48
            attention_embedding_result_dimension = 48
        else:
            if use_time and use_time_mode == 'concat':
                embedding_result_dimension = category_embedding_dim + item_embedding_dim + time_embedding_dim
                attention_embedding_result_dimension = self.attention_item_embedding_dim + self.attention_category_embedding_dim + self.attention_time_embedding_dim
            else:
                embedding_result_dimension = category_embedding_dim + item_embedding_dim
                attention_embedding_result_dimension = self.attention_item_embedding_dim + self.attention_category_embedding_dim

        # mlp for (attention/representation) of (behaviour/target)
        if not mlp_position_after_concat:
            self.mlp_for_item_attention_of_target = None
            self.mlp_for_category_attention_of_target = None
            self.mlp_for_time_attention_of_target = None
            self.mlp_for_item_representation_of_target = None
            self.mlp_for_category_representation_of_target = None
            self.mlp_for_time_representation_of_target = None
            self.mlp_for_item_attention_of_behavior = None
            self.mlp_for_category_attention_of_behavior = None
            self.mlp_for_time_attention_of_behavior = None
            self.mlp_for_item_representation_of_behavior = None
            self.mlp_for_category_representation_of_behavior = None
            self.mlp_for_time_representation_of_behavior = None
            if model_name in ['DARE', 'three_embedding', 'four_embedding', 'twin', 'DIN']:
                pass
            elif model_name == 'projection':
                self.mlp_for_item_attention_of_target = nn.Linear(self.attention_item_embedding_dim,
                                                                  self.attention_item_embedding_dim, bias=False)
                self.mlp_for_category_attention_of_target = nn.Linear(self.attention_category_embedding_dim,
                                                                      self.attention_category_embedding_dim, bias=False)
                self.mlp_for_time_attention_of_target = nn.Linear(self.attention_time_embedding_dim,
                                                                  self.attention_time_embedding_dim, bias=False)
                self.mlp_for_item_representation_of_target = nn.Linear(item_embedding_dim, item_embedding_dim,
                                                                       bias=False)
                self.mlp_for_category_representation_of_target = nn.Linear(category_embedding_dim,
                                                                           category_embedding_dim, bias=False)
                self.mlp_for_time_representation_of_target = nn.Linear(time_embedding_dim, time_embedding_dim,
                                                                       bias=False)
                self.mlp_for_item_attention_of_behavior = nn.Linear(self.attention_item_embedding_dim,
                                                                    self.attention_item_embedding_dim, bias=False)
                self.mlp_for_category_attention_of_behavior = nn.Linear(self.attention_category_embedding_dim,
                                                                        self.attention_category_embedding_dim,
                                                                        bias=False)
                self.mlp_for_time_attention_of_behavior = nn.Linear(self.attention_time_embedding_dim,
                                                                    self.attention_time_embedding_dim, bias=False)
                self.mlp_for_item_representation_of_behavior = nn.Linear(item_embedding_dim, item_embedding_dim,
                                                                         bias=False)
                self.mlp_for_category_representation_of_behavior = nn.Linear(category_embedding_dim,
                                                                             category_embedding_dim, bias=False)
                self.mlp_for_time_representation_of_behavior = nn.Linear(time_embedding_dim, time_embedding_dim,
                                                                         bias=False)
            elif model_name == 'projection_with_mlp':
                assert mlp_hidden_layer is not None
                if len(mlp_hidden_layer) == 1:
                    self.mlp_for_item_attention_of_target = nn.Sequential(
                        nn.Linear(self.attention_item_embedding_dim, mlp_hidden_layer[0]),
                        nn.PReLU(num_parameters=1, init=0.1),
                        nn.Linear(mlp_hidden_layer[0], self.attention_item_embedding_dim)
                    )
                    self.mlp_for_category_attention_of_target = nn.Sequential(
                        nn.Linear(self.attention_category_embedding_dim, mlp_hidden_layer[0]),
                        nn.PReLU(num_parameters=1, init=0.1),
                        nn.Linear(mlp_hidden_layer[0], self.attention_category_embedding_dim)
                    )
                    self.mlp_for_time_attention_of_target = nn.Sequential(
                        nn.Linear(self.attention_time_embedding_dim, mlp_hidden_layer[0]),
                        nn.PReLU(num_parameters=1, init=0.1),
                        nn.Linear(mlp_hidden_layer[0], self.attention_time_embedding_dim)
                    )
                    self.mlp_for_item_representation_of_target = nn.Sequential(
                        nn.Linear(item_embedding_dim, mlp_hidden_layer[0]),
                        nn.PReLU(num_parameters=1, init=0.1),
                        nn.Linear(mlp_hidden_layer[0], item_embedding_dim)
                    )
                    self.mlp_for_category_representation_of_target = nn.Sequential(
                        nn.Linear(category_embedding_dim, mlp_hidden_layer[0]),
                        nn.PReLU(num_parameters=1, init=0.1),
                        nn.Linear(mlp_hidden_layer[0], category_embedding_dim)
                    )
                    self.mlp_for_time_representation_of_target = nn.Sequential(
                        nn.Linear(time_embedding_dim, mlp_hidden_layer[0]),
                        nn.PReLU(num_parameters=1, init=0.1),
                        nn.Linear(mlp_hidden_layer[0], time_embedding_dim)
                    )
                    self.mlp_for_item_attention_of_behavior = self.mlp_for_item_attention_of_target
                    self.mlp_for_category_attention_of_behavior = self.mlp_for_category_attention_of_target
                    self.mlp_for_time_attention_of_behavior = self.mlp_for_time_attention_of_target
                    self.mlp_for_item_representation_of_behavior = self.mlp_for_item_representation_of_target
                    self.mlp_for_category_representation_of_behavior = self.mlp_for_category_representation_of_target
                    self.mlp_for_time_representation_of_behavior = self.mlp_for_time_representation_of_target
                elif len(mlp_hidden_layer) == 2:
                    self.mlp_for_item_attention_of_target = nn.Sequential(
                        nn.Linear(self.attention_item_embedding_dim, mlp_hidden_layer[0]),
                        nn.PReLU(num_parameters=1, init=0.1),
                        nn.Linear(mlp_hidden_layer[0], mlp_hidden_layer[1]),
                        nn.PReLU(num_parameters=1, init=0.1),
                        nn.Linear(mlp_hidden_layer[1], self.attention_item_embedding_dim)
                    )
                    self.mlp_for_category_attention_of_target = nn.Sequential(
                        nn.Linear(self.attention_category_embedding_dim, mlp_hidden_layer[0]),
                        nn.PReLU(num_parameters=1, init=0.1),
                        nn.Linear(mlp_hidden_layer[0], mlp_hidden_layer[1]),
                        nn.PReLU(num_parameters=1, init=0.1),
                        nn.Linear(mlp_hidden_layer[1], self.attention_category_embedding_dim)
                    )
                    self.mlp_for_time_attention_of_target = nn.Sequential(
                        nn.Linear(self.attention_time_embedding_dim, mlp_hidden_layer[0]),
                        nn.PReLU(num_parameters=1, init=0.1),
                        nn.Linear(mlp_hidden_layer[0], mlp_hidden_layer[1]),
                        nn.PReLU(num_parameters=1, init=0.1),
                        nn.Linear(mlp_hidden_layer[1], self.attention_time_embedding_dim)
                    )
                    self.mlp_for_item_representation_of_target = nn.Sequential(
                        nn.Linear(item_embedding_dim, mlp_hidden_layer[0]),
                        nn.PReLU(num_parameters=1, init=0.1),
                        nn.Linear(mlp_hidden_layer[0], mlp_hidden_layer[1]),
                        nn.PReLU(num_parameters=1, init=0.1),
                        nn.Linear(mlp_hidden_layer[1], item_embedding_dim)
                    )
                    self.mlp_for_category_representation_of_target = nn.Sequential(
                        nn.Linear(category_embedding_dim, mlp_hidden_layer[0]),
                        nn.PReLU(num_parameters=1, init=0.1),
                        nn.Linear(mlp_hidden_layer[0], mlp_hidden_layer[1]),
                        nn.PReLU(num_parameters=1, init=0.1),
                        nn.Linear(mlp_hidden_layer[1], category_embedding_dim)
                    )
                    self.mlp_for_time_representation_of_target = nn.Sequential(
                        nn.Linear(time_embedding_dim, mlp_hidden_layer[0]),
                        nn.PReLU(num_parameters=1, init=0.1),
                        nn.Linear(mlp_hidden_layer[0], mlp_hidden_layer[1]),
                        nn.PReLU(num_parameters=1, init=0.1),
                        nn.Linear(mlp_hidden_layer[1], time_embedding_dim)
                    )
                    self.mlp_for_item_attention_of_behavior = self.mlp_for_item_attention_of_target
                    self.mlp_for_category_attention_of_behavior = self.mlp_for_category_attention_of_target
                    self.mlp_for_time_attention_of_behavior = self.mlp_for_time_attention_of_target
                    self.mlp_for_item_representation_of_behavior = self.mlp_for_item_representation_of_target
                    self.mlp_for_category_representation_of_behavior = self.mlp_for_category_representation_of_target
                    self.mlp_for_time_representation_of_behavior = self.mlp_for_time_representation_of_target
                else:
                    assert False, 'we only support one or two mlp_hidden_layer'
            elif model_name == 'projection_no_item':
                self.mlp_for_category_attention_of_target = nn.Linear(self.attention_category_embedding_dim, self.attention_category_embedding_dim, bias=False)
                self.mlp_for_time_attention_of_target = nn.Linear(self.attention_time_embedding_dim, self.attention_time_embedding_dim, bias=False)
                self.mlp_for_category_representation_of_target = nn.Linear(category_embedding_dim, category_embedding_dim, bias=False)
                self.mlp_for_time_representation_of_target = nn.Linear(time_embedding_dim, time_embedding_dim, bias=False)
                self.mlp_for_category_attention_of_behavior = self.mlp_for_category_attention_of_target
                self.mlp_for_time_attention_of_behavior = self.mlp_for_time_attention_of_target
                self.mlp_for_category_representation_of_behavior = self.mlp_for_category_representation_of_target
                self.mlp_for_time_representation_of_behavior = self.mlp_for_time_representation_of_target
            elif model_name == 'projection_only_time':
                self.mlp_for_time_attention_of_target = nn.Linear(self.attention_time_embedding_dim, self.attention_time_embedding_dim, bias=False)
                self.mlp_for_time_representation_of_target = nn.Linear(time_embedding_dim, time_embedding_dim, bias=False)
                self.mlp_for_time_attention_of_behavior = self.mlp_for_time_attention_of_target
                self.mlp_for_time_representation_of_behavior = self.mlp_for_time_representation_of_target
            elif model_name == 'projection_only_category':
                self.mlp_for_category_attention_of_target = nn.Linear(self.attention_category_embedding_dim, self.attention_category_embedding_dim, bias=False)
                self.mlp_for_category_representation_of_target = nn.Linear(category_embedding_dim, category_embedding_dim, bias=False)
                self.mlp_for_category_attention_of_behavior = self.mlp_for_category_attention_of_target
                self.mlp_for_category_representation_of_behavior = self.mlp_for_category_representation_of_target
            elif model_name == 'projection_only_item':
                self.mlp_for_item_attention_of_target = nn.Linear(self.attention_item_embedding_dim, self.attention_item_embedding_dim, bias=False)
                self.mlp_for_item_representation_of_target = nn.Linear(item_embedding_dim, item_embedding_dim, bias=False)
                self.mlp_for_item_attention_of_behavior = self.mlp_for_item_attention_of_target
                self.mlp_for_item_representation_of_behavior = self.mlp_for_item_representation_of_target
            elif model_name == 'projection_large_emb_small_output':
                self.mlp_for_item_attention_of_target = nn.Linear(self.attention_item_embedding_dim, 16, bias=False)
                self.mlp_for_category_attention_of_target = nn.Linear(self.attention_category_embedding_dim, 16, bias=False)
                self.mlp_for_time_attention_of_target = nn.Linear(self.attention_time_embedding_dim, 16, bias=False)
                self.mlp_for_item_representation_of_target = nn.Linear(item_embedding_dim, 16, bias=False)
                self.mlp_for_category_representation_of_target = nn.Linear(category_embedding_dim, 16, bias=False)
                self.mlp_for_time_representation_of_target = nn.Linear(time_embedding_dim, 16, bias=False)
                self.mlp_for_item_attention_of_behavior = self.mlp_for_item_attention_of_target
                self.mlp_for_category_attention_of_behavior = self.mlp_for_category_attention_of_target
                self.mlp_for_time_attention_of_behavior = self.mlp_for_time_attention_of_target
                self.mlp_for_item_representation_of_behavior = self.mlp_for_item_representation_of_target
                self.mlp_for_category_representation_of_behavior = self.mlp_for_category_representation_of_target
                self.mlp_for_time_representation_of_behavior = self.mlp_for_time_representation_of_target
            else:
                assert False, "unexpected case: model_name is {}".format(model_name)
        else:
            if model_name in ['DARE', 'three_embedding', 'four_embedding', 'twin', 'DIN']:
                self.mlp_for_attention_of_target = None
                self.mlp_for_representation_of_target = None
                self.mlp_for_attention_of_behavior = None
                self.mlp_for_representation_of_behavior = None
            elif model_name == 'projection':
                self.mlp_for_attention_of_target = nn.Linear(attention_embedding_result_dimension, attention_embedding_result_dimension)
                self.mlp_for_representation_of_target = nn.Linear(embedding_result_dimension, embedding_result_dimension)
                self.mlp_for_attention_of_behavior = nn.Linear(attention_embedding_result_dimension, attention_embedding_result_dimension)
                self.mlp_for_representation_of_behavior = nn.Linear(embedding_result_dimension, embedding_result_dimension)
            elif model_name == 'projection_two_linear':
                self.mlp_for_attention_of_target = nn.Linear(attention_embedding_result_dimension, attention_embedding_result_dimension)
                self.mlp_for_representation_of_target = nn.Linear(embedding_result_dimension, embedding_result_dimension)
                self.mlp_for_attention_of_behavior = self.mlp_for_attention_of_target
                self.mlp_for_representation_of_behavior = self.mlp_for_representation_of_target
            else:
                assert False, "unexpected case: model_name is {}".format(model_name)

        # calculate the dimension of input
        target_embedding_dimension = embedding_result_dimension

        if self.short_seq_split:
            short_seq_embedding_dimension = target_embedding_dimension
            if use_cross_feature and not self.only_odot:
                short_seq_embedding_dimension *= 2
        else:
            short_seq_embedding_dimension = 0

        if self.long_seq_split:
            long_seq_embedding_dimension = target_embedding_dimension
            if self.use_long_seq_average:
                long_seq_embedding_dimension *= 2
            if use_cross_feature and not self.only_odot:
                long_seq_embedding_dimension *= 2
        else:
            long_seq_embedding_dimension = 0

        if self.only_odot:
            target_embedding_dimension = 0
        self.main_mlp_input_dimension = target_embedding_dimension + short_seq_embedding_dimension + long_seq_embedding_dimension
        # the following equation is correct by coincidence, the true meaning of auxiliary_mlp is in base_model.py
        if self.use_long_seq_average:
            self.auxiliary_mlp_input_dimension = target_embedding_dimension + int(long_seq_embedding_dimension / 2)
        else:
            self.auxiliary_mlp_input_dimension = target_embedding_dimension + long_seq_embedding_dimension

        if batch_size == 1 and no_batch_norm_if_one_batchsize:
            self.main_mlp = nn.Sequential(
                nn.Linear(self.main_mlp_input_dimension, main_mlp_hidden_1),
                nn.PReLU(num_parameters=1, init=0.1),
                nn.Linear(main_mlp_hidden_1, main_mlp_hidden_2),
                nn.PReLU(num_parameters=1, init=0.1),
                nn.Linear(main_mlp_hidden_2, 2)
            )
        else:
            self.main_mlp = nn.Sequential(
                nn.BatchNorm1d(self.main_mlp_input_dimension),
                nn.Linear(self.main_mlp_input_dimension, main_mlp_hidden_1),
                nn.PReLU(num_parameters=1, init=0.1),
                nn.Linear(main_mlp_hidden_1, main_mlp_hidden_2),
                nn.PReLU(num_parameters=1, init=0.1),
                nn.Linear(main_mlp_hidden_2, 2)
            )

        if batch_size == 1 and no_batch_norm_if_one_batchsize:
            self.auxiliary_mlp = nn.Sequential(
                nn.Linear(self.auxiliary_mlp_input_dimension, 200),
                nn.PReLU(num_parameters=1, init=0.1),
                nn.Linear(200, 2)
            )
        else:
            self.auxiliary_mlp = nn.Sequential(
                nn.BatchNorm1d(self.auxiliary_mlp_input_dimension),
                nn.Linear(self.auxiliary_mlp_input_dimension, 200),
                nn.PReLU(num_parameters=1, init=0.1),
                nn.Linear(200, 2)
            )

        if self.attn_func == 'learnable':
            self.attention_model = AttentionModel(attention_embedding_result_dimension * 4)
        else:
            self.attention_model = None

    def save(self, path):
        torch.save(self.state_dict(), path)
        print('model saved at %s' % path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
        print('model restored from %s' % path)
