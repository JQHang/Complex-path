import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn import TransformerEncoder, TransformerEncoderLayer

# 残差网络，用于转换特征，以及输出最终预测结果
class DNN(nn.Module):
    def __init__(self, input_size, output_size, dropout = 0.5):
        super().__init__()
        self.dense = nn.Linear(input_size, output_size)
        self.LayerNorm = nn.LayerNorm(output_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        return hidden_states

class Res_DNN_layer(nn.Module):
    def __init__(self, hidden_size, dropout, num_DNN):
        super().__init__()
        self.multi_DNN = nn.ModuleList([DNN(hidden_size, hidden_size, dropout) for _ in range(num_DNN)])
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.activation = nn.Tanh()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, hidden_states):
        
        hidden_states_shortcut = hidden_states
        for i,layer_module in enumerate(self.multi_DNN):
            hidden_states = layer_module(hidden_states)
        hidden_states = hidden_states_shortcut + hidden_states
        
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        return hidden_states

class Res_DNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout, num_Res_DNN, num_DNN):
        super().__init__()
        # 先将数据降维
        self.prepare = nn.Linear(input_size, hidden_size) 
        
        # 再导入两轮3层Res_DNN
        self.multi_Res_DNN = nn.ModuleList([Res_DNN_layer(hidden_size, dropout, num_DNN) for _ in range(num_Res_DNN)])
        
        # 输出层，简单的一个线性层，从hidden_size映射到num_labels
        self.classifier = nn.Linear(hidden_size, output_size)
        
    def forward(self, input_ids):
        hidden_states = self.prepare(input_ids)
        
        for i,layer_module in enumerate(self.multi_Res_DNN):
            hidden_states = layer_module(hidden_states)
        
        hidden_states = self.classifier(hidden_states)
    
        return hidden_states

# 添加position embedding(先加入pos embedding，再标准化，再dropout)
class PositionEmbeddings(nn.Module):
    def __init__(self, nfeat, seq_length, dropout):
        super().__init__()
        
        self.seq_length = seq_length
        
        self.position_embeddings = nn.Embedding(seq_length, nfeat)
        
        self.LayerNorm = nn.LayerNorm(nfeat)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, features_embeddings):
        
        position_embeddings = self.position_embeddings.weight.unsqueeze(1).expand(self.seq_length, features_embeddings.size(1), 
                                                          features_embeddings.size(2))
        
        embeddings = features_embeddings + position_embeddings
        
        embeddings = self.LayerNorm(embeddings)
        
        embeddings = self.dropout(embeddings)
        
        return embeddings

class TransformerLayer(nn.Module):
    def __init__(self, nfeat, nhead, nhid, nout, nlayers, seq_length, dropout=0.5):
        super().__init__()
        
        self.feature_Linear = nn.Linear(nfeat, nhid)
        
#         self.get_embed = PositionEmbeddings(nfeat = nhid, seq_length = seq_length, dropout = dropout)
        
        encoder_layers = TransformerEncoderLayer(nhid, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        
        self.decoder = nn.Linear(nhid*seq_length, nout)
        
        self.LayerNorm = nn.LayerNorm(nhid)
            
    def forward(self, h):
        h = self.feature_Linear(h)
        
#         h = self.get_embed(h)
        
        h = self.transformer_encoder(h)

        batch_num = h.size(1)
        h = self.decoder(h.permute(1, 0, 2).reshape(batch_num, -1))
        
        h = self.LayerNorm(h)
    
        return h
    
class Complex_Path_Transformer(nn.Module):
    def __init__(self, Node_Type_to_Feature_len_dict, Meta_Path_to_Complex_Path_dict, Meta_Path_Column_Type_dict, node_feature_hid_len, 
             metapath_level_nhid, metapath_level_nhead, metapath_level_nlayers, semantic_level_nhid, semantic_level_nhead, 
             semantic_level_nlayers, num_Res_DNN, each_Res_DNN_num, dropout=0.5):
        super().__init__()
        
        # 随机dropout函数，防止过拟合
        self.dropout = nn.Dropout(dropout)
        
        # 对注意力权重的激活函数
        self.att_Activation = nn.LeakyReLU()
        
        # 对原始特征进行映射
        self.node_feature_transform_dict = {}
        for tmp_node_type in Node_Type_to_Feature_len_dict:
#             tmp = nn.Linear(Node_Type_to_Feature_len_dict[tmp_node_type], node_feature_hid_len)
            tmp = Res_DNN(Node_Type_to_Feature_len_dict[tmp_node_type], node_feature_hid_len, node_feature_hid_len, dropout, 
                      num_Res_DNN, each_Res_DNN_num)
            
            self.node_feature_transform_dict[tmp_node_type] = tmp
            self.add_module('Node_feature_transform_{}'.format(tmp_node_type), tmp)
        
        self.Meta_Path_Column_Type_dict = Meta_Path_Column_Type_dict
            
        # 生成各元路径的细分transformer
        self.metapath_level_transformer_dict = {}
        self.complexpath_level_att_dict = {}
        for tmp_meta_path_name in Meta_Path_to_Complex_Path_dict.keys():
            self.metapath_level_transformer_dict[tmp_meta_path_name] = {}
            
            tmp = TransformerLayer(node_feature_hid_len, metapath_level_nhead, metapath_level_nhid, metapath_level_nhid,
                            metapath_level_nlayers, len(Meta_Path_Column_Type_dict[tmp_meta_path_name].keys()) + 1, dropout)
                
            self.metapath_level_transformer_dict[tmp_meta_path_name] = tmp
                
            self.add_module('Metapath_level_transformer_{}'.format(tmp_meta_path_name), tmp)
            
            for tmp_complex_path in Meta_Path_to_Complex_Path_dict[tmp_meta_path_name]:
                # 各复杂路基于注意力机制的权重
                tmp_att_W = nn.Linear(metapath_level_nhid, 1)

                self.complexpath_level_att_dict[tmp_complex_path] = tmp_att_W

                self.add_module('Complexpath_level_att_{}'.format(tmp_complex_path), tmp_att_W)
            
#             if len(Meta_Path_to_Complex_Path_dict[tmp_meta_path_name]) > 1:
#                 tmp = TransformerLayer(node_feature_hid_len, metapath_level_nhead, metapath_level_nhid, metapath_level_nhid,
#                                     metapath_level_nlayers, len(Meta_Path_to_Complex_Path_dict[tmp_meta_path_name]), dropout)
#             else:
#                 tmp = Res_DNN(metapath_level_nhid, metapath_level_nhid, metapath_level_nhid, dropout, num_Res_DNN, each_Res_DNN_num)
                
#             self.metapath_level_att_dict[tmp_meta_path_name] = tmp

#             self.add_module('Metapath_level_att_{}'.format(tmp_meta_path_name), tmp)
        
        # 最后的输出函数
        self.metapath_activation = nn.Tanh()
        self.metapath_LayerNorm = nn.LayerNorm(metapath_level_nhid * (len(Meta_Path_to_Complex_Path_dict.keys()) + 1))
        self.output_dense = Res_DNN(metapath_level_nhid * (len(Meta_Path_to_Complex_Path_dict.keys()) + 1), metapath_level_nhid, 1, dropout,
                           num_Res_DNN, each_Res_DNN_num)
#         self.output_dense = nn.Linear(semantic_level_nhid, 1)
        
        self.activation = nn.Sigmoid()
        
    def forward(self, input_feature_dict):
        # 获取各关系对应的特征
        metapath_h_feature_list = []
        
        # 先转化目标节点本身特征
        tmp_aim_node_transferred_h = self.node_feature_transform_dict['Start_Node'](input_feature_dict['Start_Node_Feature'])
        metapath_h_feature_list.append(tmp_aim_node_transferred_h)
    
        # 再对元路径特征进行转换
        for tmp_meta_path_name in input_feature_dict['Complex_Path_Feature']:
            complexpath_h_feature_list = []
            complexpath_h_att_list = []
            
            for tmp_complex_path_name in input_feature_dict['Complex_Path_Feature'][tmp_meta_path_name]:
                tmp_transferred_h_list = [tmp_aim_node_transferred_h]

                for tmp_index in input_feature_dict['Complex_Path_Feature'][tmp_meta_path_name][tmp_complex_path_name]:
                    tmp_node_type = self.Meta_Path_Column_Type_dict[tmp_meta_path_name][tmp_index]

                    tmp_transferred_h = self.node_feature_transform_dict[tmp_node_type](input_feature_dict['Complex_Path_Feature'][tmp_meta_path_name][tmp_complex_path_name][tmp_index])

                    tmp_transferred_h_list.append(tmp_transferred_h)

                # 合并转换后的原始特征
                tmp_transferred_h_stack = torch.stack(tmp_transferred_h_list, 0)
                
                # 通过metapath_level_transformer
                tmp_transferred_h_stack = self.metapath_level_transformer_dict[tmp_meta_path_name](tmp_transferred_h_stack)
                
                complexpath_h_feature_list.append(tmp_transferred_h_stack)
                
                # 计算注意力权重
                tmp_complexpath_att = self.complexpath_level_att_dict[tmp_complex_path_name](tmp_transferred_h_stack)
                tmp_complexpath_att = self.att_Activation(tmp_complexpath_att)
                complexpath_h_att_list.append(tmp_complexpath_att)
            
            if len(complexpath_h_feature_list) > 1:
                # 合并特征
                tmp_complexpath_h_feature_stack = torch.stack(complexpath_h_feature_list, dim = 1)
                tmp_complexpath_h_feature_stack = self.dropout(tmp_complexpath_h_feature_stack)
                
                # 合并注意力权重
                tmp_complexpath_attention = torch.cat(complexpath_h_att_list, dim = 1)
                
                # 对注意力权重进行归一化
                tmp_complexpath_attention = torch.softmax(tmp_complexpath_attention, dim = 1)
                
#                 print(tmp_complexpath_attention[0,:])
                
                # 扩充注意力权重维度
                tmp_complexpath_attention = tmp_complexpath_attention.unsqueeze(2)
                tmp_complexpath_attention = tmp_complexpath_attention.expand(tmp_complexpath_attention.shape[0],
                                                         tmp_complexpath_attention.shape[1],
                                                         tmp_complexpath_h_feature_stack.shape[-1])
                
                # 基于权重的结果对特征进行加权平均
                tmp_complexpath_h_feature_stack = tmp_complexpath_attention * tmp_complexpath_h_feature_stack
                tmp_complexpath_h_feature_stack = torch.mean(tmp_complexpath_h_feature_stack, dim=1)
                
                metapath_h_feature_list.append(tmp_complexpath_h_feature_stack)
            else:
                metapath_h_feature_list.append(complexpath_h_feature_list[0])
            
#             if len(complexpath_h_feature_list) > 1:
#                 tmp_complexpath_h_feature_stack = torch.stack(complexpath_h_feature_list, 0)
#             else:
#                 tmp_complexpath_h_feature_stack = complexpath_h_feature_list[0]
            
#             tmp_transferred_h_stack = self.metapath_level_att_dict[tmp_meta_path_name](tmp_complexpath_h_feature_stack)
#             metapath_h_feature_list.append(tmp_transferred_h_stack)
            
        ###################################################################################################################
        # 合并各元路径的结果
        tmp_metapath_h_feature_stack = torch.cat(metapath_h_feature_list, 1)
        
        # 输出最终结果
#         tmp_metapath_h_feature_stack = self.dropout(tmp_metapath_h_feature_stack)
#         tmp_metapath_h_feature_stack = self.metapath_activation(tmp_metapath_h_feature_stack)
#         tmp_metapath_h_feature_stack = self.metapath_LayerNorm(tmp_metapath_h_feature_stack)

        h_output = self.output_dense(tmp_metapath_h_feature_stack)
        h_output = h_output.squeeze()
        h_output = self.activation(h_output)

        return h_output