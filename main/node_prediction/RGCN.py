#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import pandas as pd
from collections import defaultdict
import math
import os
import io
import time
import copy
import re
import gc
import sys
sys.path.append("..")

import random
from random import sample
import numpy as np
import pandas as pd

import scipy.sparse as sp
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from scipy.stats import norm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.optim as optim
import math

import sklearn
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.preprocessing import normalize

from tqdm import tqdm

from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta

from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc
import matplotlib.pyplot as plt


# In[2]:


#显示所有列
pd.set_option('display.max_columns', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)

plt.rcParams["font.sans-serif"]=["SimHei"] #设置字体
plt.rcParams["axes.unicode_minus"]=False #该语句解决图像中的“-”负号的乱码问题


# In[3]:


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


# In[4]:


# 固定随机值
def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    
setup_seed(42)


# In[5]:


if torch.cuda.is_available():
    print('cuda')
    args_cuda = True
else:
    print('cpu')
    args_cuda = False


# # 预设参数

# In[6]:


from kg_lib.utils import read_json_config_file

# 选定用哪个标签表对应的结果
Label_Data_Config_file = './kg_config/Target_Node_Dataset_Config/Sign_Label_Train_Target_Data_Config_2023_05_20.json'
Label_Data_Config_dict = read_json_config_file(Label_Data_Config_file)

Model_Config_dict = {}
Model_Config_dict['train_sample_size'] = 3000
Model_Config_dict['eval_sample_size'] = 6000
Model_Config_dict['train_pos_sample_percent'] = 0.2
Model_Config_dict['train_epoch'] = 1000
Model_Config_dict['sample_num_for_eval'] = 100
Model_Config_dict['early_stop'] = 30
Model_Config_dict['learning_rate'] = 0.0001
Model_Config_dict['weight_decay'] = 0

Model_Config_dict['GAT_hid_len'] = 128
Model_Config_dict['DNN_hid_len'] = 128
Model_Config_dict['num_Res_DNN'] = 2
Model_Config_dict['each_Res_DNN_num'] = 3
Model_Config_dict['dropout'] = 0


# In[8]:


from kg_lib.utils import divid_range_list_to_monthly_list

# 目标时间及月份(左闭右开)
KG_train_time_range_list = [datetime(2023, 4, 1), datetime(2023, 4, 5)]
KG_validation_time_range_list = [datetime(2023, 4, 5), datetime(2023, 4, 10)]
KG_test_time_range_list = [datetime(2023, 4, 10), datetime(2023, 4, 15)]

KG_train_time_monthly_range_list = divid_range_list_to_monthly_list(KG_train_time_range_list)
print('KG_train_time_monthly_range_list:', KG_train_time_monthly_range_list)

KG_validation_time_monthly_range_list = divid_range_list_to_monthly_list(KG_validation_time_range_list)
print('KG_validation_time_monthly_range_list:', KG_validation_time_monthly_range_list)

KG_test_time_monthly_range_list = divid_range_list_to_monthly_list(KG_test_time_range_list)
print('KG_test_time_monthly_range_list:', KG_test_time_monthly_range_list)

# 全部要计算的时间区间
all_aim_time_monthly_range_list = (KG_train_time_monthly_range_list + KG_validation_time_monthly_range_list + 
                                   KG_test_time_monthly_range_list)

data_source_description_str = '06_13-签约标签4月subgraph复杂度测试数据'
print(data_source_description_str)

Subgraph_Hop_num = 2


# # 预处理ML格式的数据

# In[9]:


from kg_lib.Get_Subgraph_Required_Data import get_subgraph_required_pandas_data

time_range_to_Processed_Subgraph_Data_dict = {}
all_relation_list = []
node_type_to_feature_len_dict = {}
for tmp_aim_time_monthly_range in all_aim_time_monthly_range_list:
    time_range_str = ('Time_Range:' + str(tmp_aim_time_monthly_range))
    Processed_Subgraph_Data_dict = get_subgraph_required_pandas_data(data_source_description_str, time_range_str,
                                                                     aim_node_type = Label_Data_Config_dict['Node_Type'],
                                                                     subgraph_hop_num = Subgraph_Hop_num, regenerate = True, 
                                                                     Relation_drop_list = ['mobile_to_contact_company_Aim_Head',
                                                                                           'mobile_to_contact_user_pin_Aim_Tail',
                                                                                           'contact_mobile_to_road_by_receive_Aim_Tail',
                                                                                           'mobile_to_road_by_receive_Aim_Tail',
                                                                                           'mobile_to_user_pin_Aim_Tail',
                                                                                           'company_related_to_l4_industry_Aim_Tail'], 
                                                                     Feature_Type_list = ['Norm'])
    
    time_range_to_Processed_Subgraph_Data_dict[time_range_str] = Processed_Subgraph_Data_dict
    
    if len(all_relation_list) == 0:
        for tmp_hop in Processed_Subgraph_Data_dict['Adj']:
            all_relation_list.extend(list(Processed_Subgraph_Data_dict['Adj'][tmp_hop].keys()))
        all_relation_list = list(set(all_relation_list))
    
    if len(node_type_to_feature_len_dict.keys()) == 0:
        for tmp_node_type in Processed_Subgraph_Data_dict['Feature']:
            node_type_to_feature_len_dict[tmp_node_type] = Processed_Subgraph_Data_dict['Feature'][tmp_node_type].shape[1]
    
print('all_relation_list:', all_relation_list)


# # 采样函数

# In[10]:


def sample_random_index_with_portion(Source_Data_Dict, sample_size, positive_percent):
    
    tmp_pos_sample_size = math.ceil(sample_size * positive_percent)
    tmp_neg_sample_size = (sample_size - tmp_pos_sample_size)

    # 获取正样本的序号
    tmp_pos_sample_index_np = np.argwhere(Source_Data_Dict['Target_Node_Label'] == 1).T[0]

    # 随机选取指定数目的正样本的序号
    tmp_sub_pos_sample_index_np = np.random.choice(tmp_pos_sample_index_np, size = tmp_pos_sample_size, replace = False)

    # 获取负样本的序号
    tmp_neg_sample_index_np = np.argwhere(Source_Data_Dict['Target_Node_Label'] == 0).T[0]

    # 随机选取指定数目的负样本的序号
    tmp_sub_neg_sample_index_np = np.random.choice(tmp_neg_sample_index_np, size = tmp_neg_sample_size, replace = False)

    # 合并两组序号
    tmp_sampled_label_index = np.concatenate((tmp_sub_pos_sample_index_np, tmp_sub_neg_sample_index_np))
    
    return tmp_sampled_label_index


# In[11]:


def sample_sub_graph_with_label_index(Processed_Data_dict, tmp_target_node_type, tmp_sampled_label_index):
    sub_graph_data_dict = {}
    sub_graph_data_dict['Feature'] = {}
    sub_graph_data_dict['Adj'] = {}
    sub_graph_data_dict['Seed_Node'] = {}
    
    # 取出对应的标签、转为tensor，并看情况放入cuda
    sub_graph_data_dict['Label'] = torch.FloatTensor(Processed_Data_dict['Target_Node_Label'][tmp_sampled_label_index])
    if args_cuda:
        sub_graph_data_dict['Label'] = sub_graph_data_dict['Label'].cuda()
        
    # 取出目标点对应的index号
    tmp_sampled_index = Processed_Data_dict['Target_Node_Index'][tmp_sampled_label_index].astype(int)

    #####################################################################################################
    # 将目标点设为第0跳的seed node
    sub_graph_data_dict['Seed_Node'][0] = {}
    sub_graph_data_dict['Seed_Node'][0][tmp_target_node_type] = tmp_sampled_index
    
    # 获取目标节点的特征，作为第0跳的seed node的特征
    sub_graph_data_dict['Feature'][0] = {}
    tmp_node_feature = Processed_Data_dict['Feature'][tmp_target_node_type][tmp_sampled_index]
    tmp_node_feature = torch.FloatTensor(tmp_node_feature)
    if args_cuda:
        tmp_node_feature = tmp_node_feature.cuda()
    sub_graph_data_dict['Feature'][0][tmp_target_node_type] = tmp_node_feature
    
    #####################################################################################################
    # 依次处理各跳的数据
    for tmp_hop in Processed_Data_dict['Adj'].keys():
        # 记录当前跳的邻接表，以及终止点的特征和节点
        sub_graph_data_dict['Feature'][tmp_hop + 1] = {}
        sub_graph_data_dict['Adj'][tmp_hop] = {}
        sub_graph_data_dict['Seed_Node'][tmp_hop + 1] = {}
        
        # 依次处理各条边，只保留有关联的边
        for tmp_relation_name_with_aim in Processed_Data_dict['Adj'][tmp_hop]:
            # 查看首列的节点类型
            tmp_head_node_type = Processed_Data_dict['Adj'][tmp_hop][tmp_relation_name_with_aim]['head_type']

            # 查看尾列的节点类型
            tmp_tail_node_type = Processed_Data_dict['Adj'][tmp_hop][tmp_relation_name_with_aim]['tail_type']
                
            # 只保留起始点被采样了的相关边(起始点一定在seed node中，不然就有bug)
            tmp_sampled_adj_mask = np.isin(Processed_Data_dict['Adj'][tmp_hop][tmp_relation_name_with_aim]['Adj'][0],
                                           sub_graph_data_dict['Seed_Node'][tmp_hop][tmp_head_node_type])
            tmp_sampled_adj = Processed_Data_dict['Adj'][tmp_hop][tmp_relation_name_with_aim]['Adj'][:, tmp_sampled_adj_mask]
            
            sub_graph_data_dict['Adj'][tmp_hop][tmp_relation_name_with_aim] = {}
            sub_graph_data_dict['Adj'][tmp_hop][tmp_relation_name_with_aim]['Edges'] = tmp_sampled_adj
            sub_graph_data_dict['Adj'][tmp_hop][tmp_relation_name_with_aim]['Head_type'] = tmp_head_node_type
            sub_graph_data_dict['Adj'][tmp_hop][tmp_relation_name_with_aim]['Tail_type'] = tmp_tail_node_type
            
            # 如果总边数大于0
            if tmp_sampled_adj.shape[1] > 0:
                # 获取尾列涉及的全部节点
                tmp_tail_node_index = np.unique(tmp_sampled_adj[1])
                tmp_tail_node_index.sort()
                
                # 添加进入下一跳的seed_node
                if tmp_tail_node_type not in sub_graph_data_dict['Seed_Node'][tmp_hop + 1]:
                    sub_graph_data_dict['Seed_Node'][tmp_hop + 1][tmp_tail_node_type] = tmp_tail_node_index
                else:
                    tmp_concat_seed_node = sub_graph_data_dict['Seed_Node'][tmp_hop + 1][tmp_tail_node_type]
                    tmp_concat_seed_node = np.concatenate([tmp_concat_seed_node, tmp_tail_node_index])
                    tmp_concat_seed_node = np.unique(tmp_concat_seed_node)
                    tmp_concat_seed_node.sort()
                    
                    sub_graph_data_dict['Seed_Node'][tmp_hop + 1][tmp_tail_node_type] = tmp_concat_seed_node
                    
        # 获取涉及的节点的特征（分别获取seed_node和new_seed_node的）
        for tmp_node_type in sub_graph_data_dict['Seed_Node'][tmp_hop + 1]:
            tmp_sampled_index = sub_graph_data_dict['Seed_Node'][tmp_hop + 1][tmp_node_type]
            
            tmp_node_feature = Processed_Data_dict['Feature'][tmp_node_type][tmp_sampled_index]
            
            tmp_node_feature = torch.FloatTensor(tmp_node_feature)
            if args_cuda:
                tmp_node_feature = tmp_node_feature.cuda()
            sub_graph_data_dict['Feature'][tmp_hop + 1][tmp_node_type] = tmp_node_feature
            
        # 给涉及的节点新的序号（seed_node和new_seed_node分别设置新的index）
        for tmp_relation_name_with_aim in Processed_Data_dict['Adj'][tmp_hop]:
            tmp_sampled_adj = sub_graph_data_dict['Adj'][tmp_hop][tmp_relation_name_with_aim]['Edges']
            
            # 先查看是否有边
            if tmp_sampled_adj.shape[1] == 0:
                # 保存空的邻接表（做运算时就会特征自动全给0）
                tmp_sampled_adj_new_index = torch.LongTensor(np.array([[],[]]))
                
            else:
                # 查看首列的节点类型
                tmp_head_node_type = Processed_Data_dict['Adj'][tmp_hop][tmp_relation_name_with_aim]['head_type']

                # 查看尾列的节点类型
                tmp_tail_node_type = Processed_Data_dict['Adj'][tmp_hop][tmp_relation_name_with_aim]['tail_type']
                
                # 将起始点序号转化为其在全部采样点中的序号
                tmp_index_trans_dict = dict(zip(sub_graph_data_dict['Seed_Node'][tmp_hop][tmp_head_node_type], 
                                                range(len(sub_graph_data_dict['Seed_Node'][tmp_hop][tmp_head_node_type]))))
                tmp_head_new_index = np.vectorize(tmp_index_trans_dict.get)(tmp_sampled_adj[0])

                # 将终止点序号转化为其在全部终止点中的序号
                tmp_index_trans_dict = dict(zip(sub_graph_data_dict['Seed_Node'][tmp_hop + 1][tmp_tail_node_type], 
                                                range(len(sub_graph_data_dict['Seed_Node'][tmp_hop + 1][tmp_tail_node_type]))))
                tmp_tail_new_index = np.vectorize(tmp_index_trans_dict.get)(tmp_sampled_adj[1])

                tmp_sampled_adj_new_index = torch.LongTensor(np.array([tmp_head_new_index, tmp_tail_new_index]))

            if args_cuda:
                tmp_sampled_adj_new_index = tmp_sampled_adj_new_index.cuda()
            
            sub_graph_data_dict['Adj'][tmp_hop][tmp_relation_name_with_aim]['Edges'] = tmp_sampled_adj_new_index
                
    return sub_graph_data_dict


# In[1]:


# from tqdm import tqdm

# # 计算消耗内存
# sub_graph_data_dict = sample_sub_graph_with_label_index(Processed_Subgraph_Data_dict, Label_Data_Config_dict['Node_Type'], 
#                                                         np.arange(0, Processed_Subgraph_Data_dict['Target_Node_Label'].shape[0]))

# tmp_feat_mem = 0
# for tmp_hop in range(Subgraph_Hop_num + 1):
#     for tmp_path in sub_graph_data_dict['Feature'][tmp_hop]:
#         tmp_feat_mem = tmp_feat_mem + sub_graph_data_dict['Feature'][tmp_hop][tmp_path].element_size() * sub_graph_data_dict['Feature'][tmp_hop][tmp_path].nelement()

# tmp_adj_mem = 0
# for tmp_hop in range(Subgraph_Hop_num):
#     for tmp_path in sub_graph_data_dict['Adj'][tmp_hop]:
#         tmp_adj_mem = tmp_adj_mem + sub_graph_data_dict['Adj'][tmp_hop][tmp_path]['Edges'].element_size() * sub_graph_data_dict['Adj'][tmp_hop][tmp_path]['Edges'].nelement()

# print('Feature Memory:', tmp_feat_mem/(1024**2))
# print('Adj Memory:', tmp_adj_mem/(1024**2))
# print('All Memory:', (tmp_feat_mem + tmp_adj_mem)/(1024**2))

# tmp_time_list = []

# for tmp in tqdm(range(20)):
#     # 计算消耗时间
#     curr_time = datetime.now()

#     for tmp_i in range(3):
#         sampled_label_index = sample_random_index_with_portion(Processed_Subgraph_Data_dict, 4096, 0.5)
#         Sampled_Data_Dict = sample_sub_graph_with_label_index(Processed_Subgraph_Data_dict, Label_Data_Config_dict['Node_Type'], 
#                                                           sampled_label_index)

#     curr_time2 = datetime.now()
    
# #     print(curr_time2 - curr_time)

#     tmp_time_list.append((curr_time2-curr_time).total_seconds())
    
# print(np.mean(tmp_time_list), np.var(tmp_time_list))


# # 模型

# In[11]:


# 残差网络，用于转换特征，以及输出最终预测结果
class DNN(nn.Module):
    def __init__(self, input_size, output_size, dropout = 0.5):
        super().__init__()
        self.dense = nn.Linear(input_size, output_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(output_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

class Res_DNN_layer(nn.Module):
    def __init__(self, hidden_size, dropout, num_DNN):
        super().__init__()
        self.multi_DNN = nn.ModuleList([DNN(hidden_size, hidden_size, dropout) for _ in range(num_DNN)])
        self.activation = nn.Tanh()
        self.LayerNorm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states):
        
        hidden_states_shortcut = hidden_states
        for i,layer_module in enumerate(self.multi_DNN):
            hidden_states = layer_module(hidden_states)
        hidden_states = hidden_states_shortcut + hidden_states
        hidden_states = self.activation(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        
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


# In[12]:


# 针对source-target结构的GAT模型
class GATConv(nn.Module):
    def __init__(self, in_features, hid_features, dropout, bias=True):
        super(GATConv, self).__init__()

        self.in_features = in_features
        
        self.h_dropout = nn.Dropout(dropout)
        
        self.feat_linear = nn.Linear(in_features, hid_features)
        
        self.att = nn.Linear(2*hid_features, 1)
        
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(hid_features))
        else:
            self.register_parameter('bias', None)
            
        self.reset_parameters()

    def reset_parameters(self):
        if self.bias is not None:
            self.bias.data.fill_(0)
        
    # 由source到target
    def forward(self, source_h, target_h, edge_list):
        source_h = self.h_dropout(source_h)
        target_h = self.h_dropout(target_h)
        
        source_h = self.feat_linear(source_h)
        target_h = self.feat_linear(target_h)
        
        source_idx, target_idx = edge_list
        
        a_input = torch.cat([source_h[source_idx], target_h[target_idx]], dim=1)
        
        e = torch.tanh(self.att(a_input))
    
        # 稀疏矩阵
        attention = torch.sparse.FloatTensor(edge_list, e[:, 0], (source_h.size(0), target_h.size(0)))
    
        attention = torch.sparse.softmax(attention, dim=1)
        
        h_prime = torch.sparse.mm(attention, target_h)
        
        if self.bias is not None:
            h_prime = h_prime + self.bias

        return h_prime


# In[13]:


class RGAT(nn.Module):
    def __init__(self, node_feature_hid_len, all_relation_list, dropout=0.5):
        super().__init__()
        
        self.h_dropout = nn.Dropout(dropout)
        
        # 特征转化
        self.Node_Transform_list = {}
        for tmp_node_type in node_type_to_feature_len_dict:
            tmp_linear = nn.Linear(node_type_to_feature_len_dict[tmp_node_type], node_feature_hid_len)
            self.Node_Transform_list[tmp_node_type] = tmp_linear
            self.add_module('{}_Node_Transform'.format(tmp_node_type), self.Node_Transform_list[tmp_node_type])
        
        # 有多少种类型的元路径，每种元路径有多少条，就生成多少个GAT
        self.edge_GAT = {}
        for tmp_relation_name in all_relation_list:
            tmp_attention = GATConv(in_features = node_feature_hid_len, hid_features = node_feature_hid_len, dropout = dropout)
            self.edge_GAT[tmp_relation_name] = tmp_attention
            self.add_module('Edge_GAT_{}'.format(tmp_relation_name), tmp_attention)
        
        # 输出预测结果
        self.output_linear = Res_DNN(node_feature_hid_len, node_feature_hid_len, 1, dropout, 
                                          Model_Config_dict['num_Res_DNN'], 
                                          Model_Config_dict['each_Res_DNN_num'])
        
        self.activation = nn.Sigmoid()
        
        self.reset_parameters()

    def reset_parameters(self):
        for tmp_relation_name in self.edge_GAT:
            self.edge_GAT[tmp_relation_name].reset_parameters()

    def forward(self, sub_graph_data_dict):
        hop_num = len(sub_graph_data_dict['Adj'].keys())
        
        # 先将最后一跳涉及的节点特征都进行转换
        src_node_feature_dict = {}
        for tmp_node_type in sub_graph_data_dict['Feature'][hop_num]:
            transferred_node_feature = self.Node_Transform_list[tmp_node_type](sub_graph_data_dict['Feature'][hop_num][tmp_node_type])
            src_node_feature_dict[tmp_node_type] = transferred_node_feature
        
        # 从最后一跳往前进行运算
        for tmp_hop in range(hop_num - 1, -1, -1):
            src_node_feature_list_dict = {}
            
            # 先将上一跳节点涉及的特征进行转换
            for tmp_node_type in sub_graph_data_dict['Feature'][tmp_hop]:
                transferred_node_feature = self.Node_Transform_list[tmp_node_type](sub_graph_data_dict['Feature'][tmp_hop][tmp_node_type])
                sub_graph_data_dict['Feature'][tmp_hop][tmp_node_type] = transferred_node_feature
                
                # 保存每个节点本身的特征
                src_node_feature_list_dict[tmp_head_node_type] = [transferred_node_feature]
                
            # 再对每个关系运算GAT
            for tmp_relation_name in sub_graph_data_dict['Adj'][tmp_hop]:
                # 获取该关系头节点类型
                tmp_head_node_type = sub_graph_data_dict['Adj'][tmp_hop][tmp_relation_name]['Head_type']
                
                # 获取该关系尾节点类型
                tmp_tail_node_type = sub_graph_data_dict['Adj'][tmp_hop][tmp_relation_name]['Tail_type']
                
                if sub_graph_data_dict['Adj'][tmp_hop][tmp_relation_name]['Edges'].size()[1] > 0:
                    # 导入GAT获取结果
                    subgraph_based_h = self.edge_GAT[tmp_relation_name](sub_graph_data_dict['Feature'][tmp_hop][tmp_head_node_type], 
                                                                        src_node_feature_dict[tmp_tail_node_type], 
                                                                        sub_graph_data_dict['Adj'][tmp_hop][tmp_relation_name]['Edges'])
                else:
                    subgraph_based_h = torch.zeros(sub_graph_data_dict['Feature'][tmp_hop][tmp_head_node_type].size()[0],
                                                   sub_graph_data_dict['Feature'][tmp_hop][tmp_head_node_type].size()[1])
                    if args_cuda:
                        subgraph_based_h = subgraph_based_h.cuda()
                
                if tmp_head_node_type not in src_node_feature_list_dict:
                    src_node_feature_list_dict[tmp_head_node_type] = []
                src_node_feature_list_dict[tmp_head_node_type].append(subgraph_based_h)
                
            # aggreate 各关系的结果
            src_node_feature_dict = {}
            for tmp_node_type in src_node_feature_list_dict:
                src_node_feature_dict[tmp_node_type] = torch.mean(torch.stack(src_node_feature_list_dict[tmp_node_type], 0), 0)
            
        # 取出最后一跳输出的结果(最后一跳只应该有一种类型的点)
        target_node_type = list(src_node_feature_dict.keys())[0]
        h_prime = src_node_feature_dict[target_node_type]
        
        # 转化为概率，并返回预测结果
        output = self.activation(self.output_linear(h_prime))
        
        return output


# In[14]:


# 建立模型
model = RGAT(Model_Config_dict['GAT_hid_len'], all_relation_list, Model_Config_dict['dropout'])

if args_cuda:
    model.cuda()

# 优化器
optimizer = optim.Adam(model.parameters(), lr = Model_Config_dict['learning_rate'],
                            weight_decay = Model_Config_dict['weight_decay'])

# 损失函数
BCE_loss = torch.nn.BCELoss()


# # 训练函数

# In[15]:


def top_k_accuracy_score(y_true, y_score, k):
    sorted_pred = np.argsort(y_score)
    sorted_pred = sorted_pred[::-1]
    sorted_pred = sorted_pred[:k]

    hits = y_true[sorted_pred]
    
    return np.sum(hits)/k


# In[16]:


from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc

def evaluate_multi_time(model, tmp_time_range_list, need_transfer = True, print_figure = False):
    model.eval()
    with torch.no_grad():    
        h_output_squeeze_list = []
        source_data_label_list = []
        for tmp_time_range in tmp_time_range_list:
            # 分割成各个小数据
            time_range_str = ('Time_Range:' + str(tmp_time_range))
            source_Data_Dict = time_range_to_Processed_Subgraph_Data_dict[time_range_str]
            for sample_start in tqdm(range(0, source_Data_Dict['Target_Node_Label'].shape[0], Model_Config_dict['eval_sample_size'])):
                sample_end = sample_start + Model_Config_dict['eval_sample_size']
                if sample_end > source_Data_Dict['Target_Node_Label'].shape[0]:
                    sample_end = source_Data_Dict['Target_Node_Label'].shape[0]
            
                Sampled_Data_Dict = sample_sub_graph_with_label_index(source_Data_Dict, 
                                                                      Label_Data_Config_dict['Node_Type'], 
                                                                      np.arange(sample_start,sample_end))

                h_output = model(Sampled_Data_Dict)
                h_output_squeeze = torch.squeeze(h_output)
                
                h_output_squeeze_list.append(h_output_squeeze)
            
            source_data_label = torch.FloatTensor(source_Data_Dict['Target_Node_Label'])
            if args_cuda:
                source_data_label = source_data_label.cuda()
            
            source_data_label_list.append(source_data_label)
            
        h_output_squeeze = torch.cat(h_output_squeeze_list)
        source_data_label = torch.cat(source_data_label_list)
        
    loss = BCE_loss(h_output_squeeze, source_data_label)
    
    # 计算roc-auc值和AVG_Pre值
    if args_cuda:
        source_data_label_np = source_data_label.data.cpu().numpy()
        h_output_squeeze_np = h_output_squeeze.data.cpu().numpy()
    else:
        source_data_label_np = source_data_label.data.numpy()
        h_output_squeeze_np = h_output_squeeze.data.numpy()
    
    fpr, tpr, thresholds = roc_curve(source_data_label_np, h_output_squeeze_np)
    roc_auc = auc(fpr, tpr)
    
    precision, recall, _ = precision_recall_curve(source_data_label_np, h_output_squeeze_np)
    average_precision = auc(recall, precision)
    
    if print_figure:
        plt.plot(fpr, tpr, 'b', label = 'Val AUC = %0.3f' % roc_auc)
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()

        plt.plot(precision, recall, label = 'Val AVG_PRECISION = %0.3f' % average_precision)
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('recall')
        plt.xlabel('precision')
        plt.show()
    
    top_k_acc_dict = {}
    for tmp_aim_k in [100, 500, 1000, 5000, 10000]:
        top_k_acc_dict[tmp_aim_k] = top_k_accuracy_score(source_data_label_np, h_output_squeeze_np, k = tmp_aim_k)
    
    return loss.item(), roc_auc, average_precision, top_k_acc_dict


# In[17]:


from kg_lib.utils import mkdir

localtime = time.strftime("%m-%d-%H:%M", time.localtime())

temp_train_list_name = ('Train_' + KG_train_time_range_list[0].strftime('%Y_%m_%d') + '-' 
                        + KG_train_time_range_list[1].strftime('%Y_%m_%d') + '/')

# 模型参数的输出文件夹
tmp_model_parameter_output_dir = '../Model_Parameter/RGAT/' + localtime + '_' + temp_train_list_name + '/'
mkdir('../Model_Parameter')
mkdir('../Model_Parameter/RGAT/')
mkdir(tmp_model_parameter_output_dir)

# 各评价指标的变化情况
metric_list_dict = {}
metric_list_dict['train'] = {'loss':[], 'roc_auc':[], 'avg_precision':[], 'top_k_acc_dict':[]}
metric_list_dict['val'] = {'loss':[], 'roc_auc':[], 'avg_precision':[], 'top_k_acc_dict':[]}
metric_list_dict['test'] = {'loss':[], 'roc_auc':[], 'avg_precision':[], 'top_k_acc_dict':[]}

# 最优roc_auc
best_roc_auc = 0

# 累计未优化次数
early_stop_count = 0


# In[18]:


for epoch in range(Model_Config_dict['train_epoch']):
    # 多少轮查看一次效果
    pbar = tqdm(range(Model_Config_dict['sample_num_for_eval']))
    for sample_index in pbar:
        # 先采样
        time_range_str = ('Time_Range:' + str(KG_train_time_monthly_range_list[sample_index%len(KG_train_time_monthly_range_list)]))
        sampled_label_index = sample_random_index_with_portion(time_range_to_Processed_Subgraph_Data_dict[time_range_str],
                                                               Model_Config_dict['train_sample_size'], 
                                                               Model_Config_dict['train_pos_sample_percent'])
        Sampled_Data_Dict = sample_sub_graph_with_label_index(time_range_to_Processed_Subgraph_Data_dict[time_range_str],
                                                              Label_Data_Config_dict['Node_Type'], 
                                                              sampled_label_index)

        # 再训练模型
        model.train()

        h_output = model(Sampled_Data_Dict)
        h_output_squeeze = torch.squeeze(h_output)
        
        loss = BCE_loss(h_output_squeeze, Sampled_Data_Dict['Label'])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 查看其他指标
        if args_cuda:
            source_data_label_np = Sampled_Data_Dict['Label'].data.cpu().numpy()
            h_output_squeeze_np = h_output_squeeze.data.cpu().numpy()
        else:
            source_data_label_np = Sampled_Data_Dict['Label'].data.numpy()
            h_output_squeeze_np = h_output_squeeze.data.numpy()
        
        roc_auc = roc_auc_score(source_data_label_np, h_output_squeeze_np)
        average_precision = average_precision_score(source_data_label_np, h_output_squeeze_np)
        top_k_acc_dict = {}
        for tmp_aim_k in [100, 500, 1000, 5000, 10000]:
            top_k_acc_dict[tmp_aim_k] = top_k_accuracy_score(source_data_label_np, h_output_squeeze_np, k = tmp_aim_k)
        
        pbar.set_postfix({'loss':loss.item(), 'roc_auc': roc_auc, 'avg_precision': average_precision})
    
    # 查看效果
    val_loss, val_roc_auc, val_average_precision, val_top_k_acc_dict = evaluate_multi_time(model, KG_validation_time_monthly_range_list)
    test_loss, test_roc_auc, test_average_precision, test_top_k_acc_dict = evaluate_multi_time(model, KG_test_time_monthly_range_list)
    
    metric_list_dict['val']['loss'].append(val_loss)
    metric_list_dict['val']['roc_auc'].append(val_roc_auc)
    metric_list_dict['val']['avg_precision'].append(val_average_precision)
    metric_list_dict['val']['top_k_acc_dict'].append(val_top_k_acc_dict)
    
    metric_list_dict['test']['loss'].append(test_loss)
    metric_list_dict['test']['roc_auc'].append(test_roc_auc)
    metric_list_dict['test']['avg_precision'].append(test_average_precision)
    metric_list_dict['test']['top_k_acc_dict'].append(test_top_k_acc_dict)
    
    print('Epoch:', epoch)
    print('Validation - loss:', val_loss, 'roc_auc:',val_roc_auc, 'avg_precision:',val_average_precision, 
          'top_k_acc_dict:',val_top_k_acc_dict)
    print('Test - loss:', test_loss, 'roc_auc:', test_roc_auc, 'avg_precision:', test_average_precision, 
          'top_k_acc_dict:',test_top_k_acc_dict)
    
    # 达到最优效果时存储一次模型
    if val_roc_auc > best_roc_auc:
        early_stop_count = 0
        best_roc_auc = val_roc_auc
        
        torch.save(model.state_dict(), tmp_model_parameter_output_dir + 'model_parameter' + '_best_roc_auc_' + ("%.4f" % best_roc_auc) + '.pt')
    else:
        early_stop_count = early_stop_count + 1
        print("Early Stop Count:", early_stop_count)
        
        if early_stop_count >= Model_Config_dict['early_stop']:
            break


# # 打印趋势

# In[19]:


plt.plot(range(len(metric_list_dict['val']['loss'])), metric_list_dict['val']['loss'])
plt.plot(range(len(metric_list_dict['test']['loss'])), metric_list_dict['test']['loss'])
plt.show()


# In[20]:


plt.plot(range(len(metric_list_dict['val']['roc_auc'])), metric_list_dict['val']['roc_auc'])
plt.plot(range(len(metric_list_dict['test']['roc_auc'])), metric_list_dict['test']['roc_auc'])
plt.show()


# In[21]:


plt.plot(range(len(metric_list_dict['val']['avg_precision'])), metric_list_dict['val']['avg_precision'])
plt.plot(range(len(metric_list_dict['test']['avg_precision'])), metric_list_dict['test']['avg_precision'])
plt.show()


# In[22]:


# 读取模型
temp_model_file_name_list = os.listdir(tmp_model_parameter_output_dir)
temp_model_file_name_list.sort(key = lambda i:int(re.search('_0.(\d+).pt',i).group(1)))
model_used_file_name = temp_model_file_name_list[-1]
print('Used Model:', model_used_file_name)

model.load_state_dict(torch.load(tmp_model_parameter_output_dir + model_used_file_name))


# In[23]:


val_loss, val_roc_auc, val_average_precision, val_top_k_acc_dict = evaluate_multi_time(model, KG_validation_time_monthly_range_list, print_figure = True)
print(val_roc_auc, val_average_precision, val_top_k_acc_dict)

test_loss, test_roc_auc, test_average_precision, test_top_k_acc_dict = evaluate_multi_time(model, KG_test_time_monthly_range_list, print_figure = True)
print(test_roc_auc, test_average_precision, test_top_k_acc_dict)


# In[ ]:




