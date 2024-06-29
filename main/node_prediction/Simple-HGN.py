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
    
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# # 预设参数

# In[6]:


from kg_lib.utils import read_json_config_file

Target_Node_Type = "Mobile_Node"

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


# In[7]:


from kg_lib.utils import divid_range_list_to_monthly_list

# 目标时间及月份(左闭右开)
KG_train_time_range_list = [datetime(2022, 11, 20), datetime(2022, 12, 10)]
KG_validation_time_range_list = [datetime(2022, 12, 10), datetime(2022, 12, 15)]
KG_test_time_range_list = [datetime(2022, 12, 15), datetime(2022, 12, 20)]

KG_train_time_monthly_range_list = divid_range_list_to_monthly_list(KG_train_time_range_list)
print('KG_train_time_monthly_range_list:', KG_train_time_monthly_range_list)

KG_validation_time_monthly_range_list = divid_range_list_to_monthly_list(KG_validation_time_range_list)
print('KG_validation_time_monthly_range_list:', KG_validation_time_monthly_range_list)

KG_test_time_monthly_range_list = divid_range_list_to_monthly_list(KG_test_time_range_list)
print('KG_test_time_monthly_range_list:', KG_test_time_monthly_range_list)

# 全部要计算的时间区间
all_aim_time_monthly_range_list = (KG_train_time_monthly_range_list + KG_validation_time_monthly_range_list + 
                                   KG_test_time_monthly_range_list)

# data_source_description_str = '06_13-签约标签4月subgraph复杂度测试数据'
# data_source_description_str = '06_19-签约标签2至4月subgraph数据'
data_source_description_str = '06_26-热线索标签11至12月subgraph数据'
print(data_source_description_str)

Subgraph_Hop_num = 2


# In[9]:


from kg_lib.Get_Subgraph_Required_Data import get_subgraph_required_pandas_data

time_range_to_Processed_Subgraph_Data_dict = {}
all_relation_list = []
node_type_to_feature_len_dict = {}
for tmp_aim_time_monthly_range in all_aim_time_monthly_range_list:
    time_range_str = ('Time_Range:' + str(tmp_aim_time_monthly_range))
    Processed_Subgraph_Data_dict = get_subgraph_required_pandas_data(data_source_description_str, time_range_str,
                                                                     aim_node_type = Target_Node_Type,
                                                                     subgraph_hop_num = Subgraph_Hop_num, regenerate = True, 
                                                                     Relation_drop_list = [], 
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


import dgl

# 把结果转pyg或dgl
def Trans_To_DGL(Source_Data_dict):
    # 合并各跳涉及到的同类型的边
    hg_adj_dict = {}
    for tmp_hop in Source_Data_dict['Adj']:
        for tmp_relation_name in Source_Data_dict['Adj'][tmp_hop]:
            tmp_adj = Source_Data_dict['Adj'][tmp_hop][tmp_relation_name]
            
            # 我们采样的边头结点对应target,尾结点对应source
            target_column_node_type = Source_Data_dict['Relation_Node_Type'][tmp_relation_name]['Head_type']
            source_column_node_type = Source_Data_dict['Relation_Node_Type'][tmp_relation_name]['Tail_type']

            tmp_relation_info_set = (source_column_node_type, tmp_relation_name, target_column_node_type)

            if tmp_relation_info_set not in hg_adj_dict:
                hg_adj_dict[tmp_relation_info_set] = tmp_adj
            else:
                hg_adj_dict[tmp_relation_info_set] = np.concatenate((hg_adj_dict[tmp_relation_info_set], tmp_adj), axis = 1)

            hg_adj_dict[tmp_relation_info_set] = np.unique(hg_adj_dict[tmp_relation_info_set], axis=1)

    for tmp_relation_info_set in hg_adj_dict:
        hg_adj_dict[tmp_relation_info_set] = (torch.tensor(hg_adj_dict[tmp_relation_info_set][1,:]), 
                                              torch.tensor(hg_adj_dict[tmp_relation_info_set][0,:]))
    
    ############################################################################################################
    # 为每个节点添加特征
    num_nodes_dict = {}
    hg_feat_dict = {}
    for tmp_node_type in Source_Data_dict['Feature']:
        hg_feat_dict[tmp_node_type] = torch.FloatTensor(Source_Data_dict['Feature'][tmp_node_type])
        num_nodes_dict[tmp_node_type] = Source_Data_dict['Feature'][tmp_node_type].shape[0]
        
    ############################################################################################################
    # 基于全部的点和边创建图
    hg_dgl = dgl.heterograph(hg_adj_dict, num_nodes_dict = num_nodes_dict)
    hg_dgl.ndata['feat'] = hg_feat_dict
    
    ############################################################################################################
    # 添加标签及标签对应的节点序号
    hg_dgl.ndata["label"] = {Target_Node_Type: torch.FloatTensor(Source_Data_dict['All_Target_Node_Label'])}
    
    return hg_dgl

# hg_dgl = Trans_To_DGL(Processed_Subgraph_Data_dict)


# In[11]:


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


# In[12]:


# 对部分关系，加入反转head和tail后的边
def add_reverse_hetero(g):
    relations = {}
    num_nodes_dict = {ntype: g.num_nodes(ntype) for ntype in g.ntypes}
    
    for metapath in g.canonical_etypes:
        # Original edges
        src, dst = g.all_edges(etype=metapath[1])
        relations[metapath] = (src, dst)

        reverse_metapath = (metapath[2], metapath[1] + '_by', metapath[0])
        assert reverse_metapath not in relations
        relations[reverse_metapath] = (dst, src)           # Reverse edges

    new_g = dgl.heterograph(relations, num_nodes_dict = num_nodes_dict)

    # copy_ndata:
    for ntype in g.ntypes:
        for k, v in g.nodes[ntype].data.items():
            new_g.nodes[ntype].data[k] = v.detach().clone()
    
    return new_g


# In[13]:


def sample_sub_graph_with_index(Processed_Data_dict, tmp_sampled_label_index):
    tmp_node_type_to_sampled_index = {}
    
    sub_graph_data_dict = {}
    sub_graph_data_dict['Feature'] = {}
    sub_graph_data_dict['Adj'] = {}
    sub_graph_data_dict['Relation_Node_Type'] = {}
    sub_graph_data_dict['Seed_Node'] = {}
    
    # 取出对应的标签、转为tensor，并看情况放入cuda
    if 'Target_Node_Label' in Processed_Data_dict:
        sub_graph_data_dict['Label'] = torch.FloatTensor(Processed_Data_dict['Target_Node_Label'][tmp_sampled_label_index])
        
    # 取出目标点对应的index号
    tmp_sampled_index = Processed_Data_dict['Target_Node_Index'][tmp_sampled_label_index].astype(int)
    
    # 保存结果
    sub_graph_data_dict['Target_Node_Index'] = tmp_sampled_index
    sub_graph_data_dict['Target_Node_Type'] = Target_Node_Type
    tmp_node_type_to_sampled_index[Target_Node_Type] = tmp_sampled_index
    #####################################################################################################
    # 将目标点设为第0跳的seed node
    sub_graph_data_dict['Seed_Node'][0] = {}
    sub_graph_data_dict['Seed_Node'][0][Target_Node_Type] = tmp_sampled_index
    
    # 依次处理各跳的数据获得各类型节点所涉及的全部目标点和边
    for tmp_hop in Processed_Data_dict['Adj'].keys():
        sub_graph_data_dict['Adj'][tmp_hop] = {}
        sub_graph_data_dict['Seed_Node'][tmp_hop + 1] = {}
        
        # 依次处理各跳中涉及的边，只保留和目标点有关联的边
        for tmp_relation_name_with_aim in Processed_Data_dict['Adj'][tmp_hop]:
            # 查看首列的节点类型
            tmp_head_node_type = Processed_Data_dict['Adj'][tmp_hop][tmp_relation_name_with_aim]['head_type']

            # 查看尾列的节点类型
            tmp_tail_node_type = Processed_Data_dict['Adj'][tmp_hop][tmp_relation_name_with_aim]['tail_type']
                
            # 只保留起始点被采样了的相关边(起始点一定在seed node中，不然就有bug)
            tmp_sampled_adj_mask = np.isin(Processed_Data_dict['Adj'][tmp_hop][tmp_relation_name_with_aim]['Adj'][0],
                                           sub_graph_data_dict['Seed_Node'][tmp_hop][tmp_head_node_type])
            tmp_sampled_adj = Processed_Data_dict['Adj'][tmp_hop][tmp_relation_name_with_aim]['Adj'][:, tmp_sampled_adj_mask]
            
            # 保存采样后的边
            sub_graph_data_dict['Adj'][tmp_hop][tmp_relation_name_with_aim] = tmp_sampled_adj
            
            if tmp_relation_name_with_aim not in sub_graph_data_dict['Relation_Node_Type']:
                sub_graph_data_dict['Relation_Node_Type'][tmp_relation_name_with_aim] = {}
                sub_graph_data_dict['Relation_Node_Type'][tmp_relation_name_with_aim]['Head_type'] = tmp_head_node_type
                sub_graph_data_dict['Relation_Node_Type'][tmp_relation_name_with_aim]['Tail_type'] = tmp_tail_node_type
            
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
                
                # 保存全部涉及到的节点
                if tmp_tail_node_type not in tmp_node_type_to_sampled_index:
                    tmp_node_type_to_sampled_index[tmp_tail_node_type] = tmp_tail_node_index
                else:
                    tmp_concat_seed_node = tmp_node_type_to_sampled_index[tmp_tail_node_type]
                    tmp_concat_seed_node = np.concatenate([tmp_concat_seed_node, tmp_tail_node_index])
                    tmp_concat_seed_node = np.unique(tmp_concat_seed_node)
                    tmp_concat_seed_node.sort()
                    
                    tmp_node_type_to_sampled_index[tmp_tail_node_type] = tmp_concat_seed_node
                    
    #####################################################################################################
    # 保存涉及的节点的特征及各节点采样前后的index的对应关系
    tmp_node_type_to_index_trans_dict = {}
    for tmp_node_type in tmp_node_type_to_sampled_index:
        tmp_node_sampled_index = tmp_node_type_to_sampled_index[tmp_node_type]

        tmp_node_feature = Processed_Data_dict['Feature'][tmp_node_type][tmp_node_sampled_index]

        tmp_node_feature = torch.FloatTensor(tmp_node_feature)
        
        sub_graph_data_dict['Feature'][tmp_node_type] = tmp_node_feature
        
        tmp_index_trans_dict = dict(zip(tmp_node_sampled_index, range(len(tmp_node_sampled_index))))
        
        tmp_node_type_to_index_trans_dict[tmp_node_type] = tmp_index_trans_dict
        
        # 如果有标签，则也转化下对应的标签的顺序
        if tmp_node_type == Target_Node_Type and 'All_Target_Node_Label' in Processed_Data_dict:
            
            sub_graph_data_dict['All_Target_Node_Label'] = Processed_Data_dict['All_Target_Node_Label'][tmp_node_sampled_index]
        
    #####################################################################################################
    # 转换起始点对应的index
    tmp_index_trans_dict = tmp_node_type_to_index_trans_dict[Target_Node_Type]
    sub_graph_data_dict['Target_Node_Index'] = np.vectorize(tmp_index_trans_dict.get)(sub_graph_data_dict['Target_Node_Index'])
    
    #####################################################################################################
    # 转化各个关系中的节点对应的index
    for tmp_hop in sub_graph_data_dict['Adj'].keys():
        for tmp_relation_name_with_aim in sub_graph_data_dict['Adj'][tmp_hop]:
            tmp_sampled_adj = sub_graph_data_dict['Adj'][tmp_hop][tmp_relation_name_with_aim]

            # 查看首列的节点类型
            tmp_head_node_type = sub_graph_data_dict['Relation_Node_Type'][tmp_relation_name_with_aim]['Head_type']

            # 查看尾列的节点类型
            tmp_tail_node_type = sub_graph_data_dict['Relation_Node_Type'][tmp_relation_name_with_aim]['Tail_type']
            
            # 先查看是否有边
            if tmp_sampled_adj.shape[1] == 0:
                # 保存空的邻接表（做运算时就会特征自动全给0）
                tmp_sampled_adj_new_index = torch.LongTensor(np.array([[],[]]))

            else:
                # 将起始点序号转化为其在全部采样点中的序号
                tmp_index_trans_dict = tmp_node_type_to_index_trans_dict[tmp_head_node_type]
                tmp_head_new_index = np.vectorize(tmp_index_trans_dict.get)(tmp_sampled_adj[0])

                # 将终止点序号转化为其在全部终止点中的序号
                tmp_index_trans_dict = tmp_node_type_to_index_trans_dict[tmp_tail_node_type]
                tmp_tail_new_index = np.vectorize(tmp_index_trans_dict.get)(tmp_sampled_adj[1])

                tmp_sampled_adj_new_index = torch.LongTensor(np.array([tmp_head_new_index, tmp_tail_new_index]))

            sub_graph_data_dict['Adj'][tmp_hop][tmp_relation_name_with_aim] = tmp_sampled_adj_new_index
    
    #####################################################################################################
    # 将对应的subgraph转为dgl格式的
    sub_hg_dgl = Trans_To_DGL(sub_graph_data_dict)
    
    # 补全反向的边 BUG
    sub_hg_dgl = add_reverse_hetero(sub_hg_dgl)
    
    if args_cuda:
        sub_hg_dgl = sub_hg_dgl.to(device)
    
    return sub_hg_dgl

sampled_label_index = sample_random_index_with_portion(Processed_Subgraph_Data_dict, 1024, 0.5)
sub_hg_dgl = sample_sub_graph_with_index(Processed_Subgraph_Data_dict, sampled_label_index)


# In[14]:


# import dgl.function as fn
# import dgl.nn as dglnn
# from dgl.dataloading import MultiLayerFullNeighborSampler, NodeDataLoader, DataLoader, GraphDataLoader
# # from dgl import apply_each

# hg_dgl = time_range_to_Processed_Subgraph_Data_dict[time_range_str]

# sampled_label_index = sample_random_index_for_dgl(hg_dgl, Model_Config_dict['train_sample_size'], 0.5)
# sampler = MultiLayerFullNeighborSampler(2)

# tmp_nhid_dict = {}
# for tmp_node_type in hg_dgl.ntypes:
#     if tmp_node_type == Target_Node_Type:
#         tmp_nhid_dict[tmp_node_type] = sampled_label_index
#     else:
#         tmp_nhid_dict[tmp_node_type] = hg_dgl.nodes(tmp_node_type)

# dataloader = NodeDataLoader(hg_dgl, tmp_nhid_dict, sampler, batch_size = Model_Config_dict['train_sample_size'], shuffle=True, 
#                             drop_last=False, num_workers=4)

# for blocks in dataloader:
# #     print(blocks)
#     # 只保留涉及到的点和边组成的graph
    
#     print(blocks)
    
#     break


# In[15]:


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

# # 计算消耗时间
# curr_time = datetime.now()

# for tmp_i in range(3):
#     sampled_label_index = sample_random_index_with_portion(Processed_Subgraph_Data_dict, 4096, 0.5)
#     Sampled_Data_Dict = sample_sub_graph_with_label_index(Processed_Subgraph_Data_dict, Label_Data_Config_dict['Node_Type'], 
#                                                           sampled_label_index)
    
# curr_time2 = datetime.now()
# print(curr_time2-curr_time) 


# # 模型

# In[16]:


# from importlib import reload 
# import kg_model
# reload(kg_model.HGT_dgl)

import torch.nn as nn
from kg_model.HGT_dgl import HGT

node_dict = {}
edge_dict = {}
for ntype in sub_hg_dgl.ntypes:
    node_dict[ntype] = len(node_dict)
for etype in sub_hg_dgl.etypes:
    edge_dict[etype] = len(edge_dict)
#     sub_hg_dgl.edges[etype].data['id'] = torch.ones(sub_hg_dgl.number_of_edges(etype), dtype=torch.long) * edge_dict[etype] 

# 建立模型
model = HGT(node_dict, edge_dict, node_type_to_feature_len_dict, Model_Config_dict['GAT_hid_len'], 1, 2, 8)

output_activation = nn.Sigmoid()

if args_cuda:
    model.cuda()

# 优化器
optimizer = optim.AdamW(model.parameters(), lr = Model_Config_dict['learning_rate'],
                            weight_decay = Model_Config_dict['weight_decay'])

# 损失函数
BCE_loss = torch.nn.BCELoss()


# # 训练函数

# In[17]:


def top_k_accuracy_score(y_true, y_score, k):
    sorted_pred = np.argsort(y_score)
    sorted_pred = sorted_pred[::-1]
    sorted_pred = sorted_pred[:k]
    
    hits = y_true[sorted_pred]
    
    return np.sum(hits)/k


# In[18]:


from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc

def evaluate_dgl_multi_time(model, tmp_time_range_list, need_transfer = True, print_figure = False):
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
            
                sampled_hg = sample_sub_graph_with_index(source_Data_Dict, np.arange(sample_start,sample_end))

                h_output = model(sampled_hg, Target_Node_Type)
                h_output_squeeze = torch.squeeze(h_output)
                
                h_output_squeeze_list.append(h_output_squeeze)
            
            source_data_label = torch.FloatTensor(source_Data_Dict['Target_Node_Label'])
            if args_cuda:
                source_data_label = source_data_label.cuda()
            
            source_data_label_list.append(source_data_label)
            
        h_output_squeeze = torch.cat(h_output_squeeze_list)
        source_data_label = torch.cat(source_data_label_list)
        
        h_output_squeeze = output_activation(h_output_squeeze)
        
        # 查询label为0或1的节点的序号
        source_data_label_index = (source_data_label != -1).nonzero()
        
        # 只保留指定index的数据
        h_output_squeeze = h_output_squeeze[source_data_label_index]
        source_data_label = source_data_label[source_data_label_index]
        
        h_output_squeeze = torch.squeeze(h_output_squeeze)
        source_data_label = torch.squeeze(source_data_label)
        
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


# In[19]:


from kg_lib.utils import mkdir

localtime = time.strftime("%m-%d-%H:%M", time.localtime())

temp_train_list_name = ('Train_' + KG_train_time_range_list[0].strftime('%Y_%m_%d') + '-' 
                        + KG_train_time_range_list[1].strftime('%Y_%m_%d') + '/')

# 模型参数的输出文件夹
tmp_model_parameter_output_dir = '../../Model_Parameter/HGT/' + localtime + '_' + temp_train_list_name + '/'
mkdir('../../Model_Parameter')
mkdir('../../Model_Parameter/HGT/')
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


# In[20]:


for epoch in range(Model_Config_dict['train_epoch']):
    # 多少轮查看一次效果
    pbar = tqdm(range(Model_Config_dict['sample_num_for_eval']))
    for sample_index in pbar:
        # 先采样
        time_range_str = ('Time_Range:' + str(KG_train_time_monthly_range_list[sample_index%len(KG_train_time_monthly_range_list)]))
        sampled_label_index = sample_random_index_with_portion(time_range_to_Processed_Subgraph_Data_dict[time_range_str],
                                                               Model_Config_dict['train_sample_size'], 
                                                               Model_Config_dict['train_pos_sample_percent'])
        sampled_hg = sample_sub_graph_with_index(time_range_to_Processed_Subgraph_Data_dict[time_range_str], sampled_label_index)

        # 再训练模型
        model.train()

        h_output = model(sampled_hg, Target_Node_Type)
        h_output_squeeze = torch.squeeze(h_output)
        h_output_squeeze = output_activation(h_output_squeeze)
        
        # 查询label为0或1的节点的序号
        tmp_true_label = sampled_hg.nodes[Target_Node_Type].data['label']
        tmp_true_label_index = (tmp_true_label != -1).nonzero()
        
        # 只保留指定index的数据
        h_output_squeeze = h_output_squeeze[tmp_true_label_index]
        tmp_true_label = tmp_true_label[tmp_true_label_index]
        
        loss = BCE_loss(h_output_squeeze, tmp_true_label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 查看其他指标
        if args_cuda:
            source_data_label_np = sampled_hg.nodes[Target_Node_Type].data['label'].data.cpu().numpy()
            h_output_squeeze_np = h_output_squeeze.data.cpu().numpy()
        else:
            source_data_label_np = sampled_hg.nodes[Target_Node_Type].data['label'].data.numpy()
            h_output_squeeze_np = h_output_squeeze.data.numpy()
        
        roc_auc = roc_auc_score(source_data_label_np, h_output_squeeze_np)
        average_precision = average_precision_score(source_data_label_np, h_output_squeeze_np)
        top_k_acc_dict = {}
        for tmp_aim_k in [100, 500, 1000, 5000, 10000]:
            top_k_acc_dict[tmp_aim_k] = top_k_accuracy_score(source_data_label_np, h_output_squeeze_np, k = tmp_aim_k)
        
        pbar.set_postfix({'loss':loss.item(), 'roc_auc': roc_auc, 'avg_precision': average_precision})
        
    # 查看效果
    train_loss, train_roc_auc, train_average_precision, train_top_k_acc_dict = evaluate_dgl_multi_time(model, KG_train_time_monthly_range_list)
    val_loss, val_roc_auc, val_average_precision, val_top_k_acc_dict = evaluate_dgl_multi_time(model, KG_validation_time_monthly_range_list)
    test_loss, test_roc_auc, test_average_precision, test_top_k_acc_dict = evaluate_dgl_multi_time(model, KG_test_time_monthly_range_list)
    
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

# In[21]:


plt.plot(range(len(metric_list_dict['train']['loss'])), metric_list_dict['train']['loss'])
plt.plot(range(len(metric_list_dict['val']['loss'])), metric_list_dict['val']['loss'])
plt.plot(range(len(metric_list_dict['test']['loss'])), metric_list_dict['test']['loss'])
plt.show()


# In[22]:


plt.plot(range(len(metric_list_dict['train']['roc_auc'])), metric_list_dict['train']['roc_auc'])
plt.plot(range(len(metric_list_dict['val']['roc_auc'])), metric_list_dict['val']['roc_auc'])
plt.plot(range(len(metric_list_dict['test']['roc_auc'])), metric_list_dict['test']['roc_auc'])
plt.show()


# In[23]:


plt.plot(range(len(metric_list_dict['train']['avg_precision'])), metric_list_dict['train']['avg_precision'])
plt.plot(range(len(metric_list_dict['val']['avg_precision'])), metric_list_dict['val']['avg_precision'])
plt.plot(range(len(metric_list_dict['test']['avg_precision'])), metric_list_dict['test']['avg_precision'])
plt.show()


# In[24]:


# 读取模型
temp_model_file_name_list = os.listdir(tmp_model_parameter_output_dir)
temp_model_file_name_list.sort(key = lambda i:int(re.search('_0.(\d+).pt',i).group(1)))
model_used_file_name = temp_model_file_name_list[-1]
print('Used Model:', model_used_file_name)

model.load_state_dict(torch.load(tmp_model_parameter_output_dir + model_used_file_name))


# In[25]:


val_loss, val_roc_auc, val_average_precision, val_top_k_acc_dict = evaluate_dgl_multi_time(model, KG_validation_time_monthly_range_list, print_figure = True)
print(val_roc_auc, val_average_precision, val_top_k_acc_dict)

test_loss, test_roc_auc, test_average_precision, test_top_k_acc_dict = evaluate_dgl_multi_time(model, KG_test_time_monthly_range_list, print_figure = True)
print(test_roc_auc, test_average_precision, test_top_k_acc_dict)


# In[ ]:




