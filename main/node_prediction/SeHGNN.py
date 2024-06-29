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

import math
import os
import sys
sys.path.append("..")

from tqdm import tqdm

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


# # 读取参数

# In[6]:


from kg_lib.utils import read_json_config_file

# 选定用哪个标签表对应的结果
Label_Data_Config_file = './kg_config/Target_Node_Dataset_Config/Sign_Label_Train_Target_Data_Config_2023_05_20.json'
Label_Data_Config_dict = read_json_config_file(Label_Data_Config_file)

Model_Config_dict = {}
Model_Config_dict['train_sample_size'] = 4096
Model_Config_dict['eval_sample_size'] = 100000
Model_Config_dict['train_pos_sample_percent'] = 0.1
Model_Config_dict['train_epoch'] = 1000
Model_Config_dict['sample_num_for_eval'] = 30
Model_Config_dict['learning_rate'] = 0.0005
Model_Config_dict['weight_decay'] = 0

Model_Config_dict['node_feature_hid_len'] = 128
Model_Config_dict['metapath_level_nhid'] = 128
Model_Config_dict['metapath_level_nhead'] = 1
Model_Config_dict['metapath_level_nlayers'] = 1
Model_Config_dict['num_Res_DNN'] = 1
Model_Config_dict['each_Res_DNN_num'] = 2
Model_Config_dict['dropout'] = 0

print(Model_Config_dict)


# In[7]:


# 数据来源描述
# Output_Columns_Type = "Head_And_Tail"
Output_Columns_Type = "ALL_Nodes"
# Output_Columns_Type = "ALL_Nodes_And_Edges"

Feature_Month_Range = 1

# data_source_description_str = '06_13-签约标签4月复杂度测试数据-过去' + str(Feature_Month_Range) + '个月的特征-' + Output_Columns_Type + '格式'
data_source_description_str = '06_19-签约标签2至4月数据-过去' + str(Feature_Month_Range) + '个月的特征-' + Output_Columns_Type + '格式'
print(data_source_description_str)


# In[8]:


from kg_lib.utils import divid_range_list_to_monthly_list

# 目标时间及月份(左闭右开)
KG_train_time_range_list = [datetime(2023, 2, 1), datetime(2023, 4, 10)]
KG_validation_time_range_list = [datetime(2023, 4, 10), datetime(2023, 4, 20)]
KG_test_time_range_list = [datetime(2023, 4, 20), datetime(2023, 5, 1)]

KG_train_time_monthly_range_list = divid_range_list_to_monthly_list(KG_train_time_range_list)
print('KG_train_time_monthly_range_list:', KG_train_time_monthly_range_list)

KG_validation_time_monthly_range_list = divid_range_list_to_monthly_list(KG_validation_time_range_list)
print('KG_validation_time_monthly_range_list:', KG_validation_time_monthly_range_list)

KG_test_time_monthly_range_list = divid_range_list_to_monthly_list(KG_test_time_range_list)
print('KG_test_time_monthly_range_list:', KG_test_time_monthly_range_list)

# 全部要计算的时间区间
all_aim_time_monthly_range_list = (KG_train_time_monthly_range_list + KG_validation_time_monthly_range_list + 
                                   KG_test_time_monthly_range_list)


# # 预处理数据，并存储或读取已存在的数据

# In[9]:


from kg_lib.Get_SeHGNN_Required_Data import Get_SeHGNN_Required_Pandas_Data

All_Feature_Type_Name_list = []
Node_Type_to_Feature_len_dict = {}
Feature_Node_Type_dict = {}

time_range_to_Processed_SeHGNN_Data_dict = {}

for tmp_aim_time_monthly_range in all_aim_time_monthly_range_list:
    time_range_str = ('Time_Range:' + str(tmp_aim_time_monthly_range))
    Processed_SeHGNN_Data_dict = Get_SeHGNN_Required_Pandas_Data(data_source_description_str, time_range_str,
                                                                 aim_node_type = Label_Data_Config_dict['Node_Type'],
                                                                 Meta_path_drop_list = [], 
                                                                 Feature_Type_list = ['Norm'])
    
    if len(All_Feature_Type_Name_list) == 0:
        All_Feature_Type_Name_list = list(Processed_SeHGNN_Data_dict['Feature_Dict'].keys())
    if len(Node_Type_to_Feature_len_dict.keys()) == 0:
        for tmp_Feature_Type in Processed_SeHGNN_Data_dict['Feature_Dict']:
            tmp_node_type = Processed_SeHGNN_Data_dict['Feature_Node_Type_Dict'][tmp_Feature_Type]
            Node_Type_to_Feature_len_dict[tmp_node_type] = Processed_SeHGNN_Data_dict['Feature_Dict'][tmp_Feature_Type].shape[1]
    if len(Feature_Node_Type_dict.keys()) == 0:
        Feature_Node_Type_dict = Processed_SeHGNN_Data_dict['Feature_Node_Type_Dict'].copy()
        
    time_range_to_Processed_SeHGNN_Data_dict[time_range_str] = Processed_SeHGNN_Data_dict


# # 读取预处理后的数据
# 读取特征，标签向量，获取各类型节点对应的特征长度

# In[10]:


def Merge_SeHGNN_Data(aim_time_monthly_range_list):
    Return_Data_Dict = {}
    
    # 先合并标签并记录各元路径的各列的类型
    tmp_all_label_list = []
    tmp_aim_node_feature_list = []
    for tmp_aim_time_monthly_range in aim_time_monthly_range_list:
        time_range_str = ('Time_Range:' + str(tmp_aim_time_monthly_range))
        Processed_SeHGNN_Data_dict = time_range_to_Processed_SeHGNN_Data_dict[time_range_str]
        
        tmp_all_label_list.append(Processed_SeHGNN_Data_dict['Label'])
        
    Return_Data_Dict['Label'] = pd.concat(tmp_all_label_list).values
    
    Return_Data_Dict['Feature_Dict'] = {}
    for tmp_feature_type in All_Feature_Type_Name_list:
        tmp_time_feature_list = []
        for tmp_aim_time_monthly_range in aim_time_monthly_range_list:
            time_range_str = ('Time_Range:' + str(tmp_aim_time_monthly_range))
            Processed_SeHGNN_Data_dict = time_range_to_Processed_SeHGNN_Data_dict[time_range_str]

            tmp_time_feature_list.append(Processed_SeHGNN_Data_dict['Feature_Dict'][tmp_feature_type])

        tmp_time_feature_pd = pd.concat(tmp_time_feature_list)
            
        Return_Data_Dict['Feature_Dict'][tmp_feature_type] = tmp_time_feature_pd.values
            
    return Return_Data_Dict


# In[11]:


Train_Data_Dict = Merge_SeHGNN_Data(KG_train_time_monthly_range_list)
Validation_Data_Dict = Merge_SeHGNN_Data(KG_validation_time_monthly_range_list)
Test_Data_Dict = Merge_SeHGNN_Data(KG_test_time_monthly_range_list)


# # 根据需求随机采样

# In[12]:


def sample_random_index_with_portion(Source_Data_Dict, sample_size, positive_percent):
    
    tmp_pos_sample_size = math.ceil(sample_size * positive_percent)
    tmp_neg_sample_size = (sample_size - tmp_pos_sample_size)

    # 获取正样本的序号
    tmp_pos_sample_index_np = np.argwhere(Source_Data_Dict['Label'] == 1).T[0]

    # 随机选取指定数目的正样本的序号
    tmp_sub_pos_sample_index_np = np.random.choice(tmp_pos_sample_index_np, size = tmp_pos_sample_size, replace = False)

    # 获取负样本的序号
    tmp_neg_sample_index_np = np.argwhere(Source_Data_Dict['Label'] == 0).T[0]

    # 随机选取指定数目的负样本的序号
    tmp_sub_neg_sample_index_np = np.random.choice(tmp_neg_sample_index_np, size = tmp_neg_sample_size, replace = False)

    # 合并两组序号
    tmp_sampled_label_index = np.concatenate((tmp_sub_pos_sample_index_np, tmp_sub_neg_sample_index_np))
    
    return tmp_sampled_label_index


# In[13]:


def sample_for_SeHGNN(Source_Data_Dict, tmp_sampled_index_np):
    Sampled_Data_Dict = {}
    
    # 提取对应标签
    Sampled_Data_Dict['Label'] = Source_Data_Dict['Label'][tmp_sampled_index_np]

    # 提取各类型对应特征
    Sampled_Data_Dict['Feature_Dict'] = {}
    for tmp_feature_type in All_Feature_Type_Name_list:
        tmp_sampled_feature = Source_Data_Dict['Feature_Dict'][tmp_feature_type][tmp_sampled_index_np]
        Sampled_Data_Dict['Feature_Dict'][tmp_feature_type] = tmp_sampled_feature
        
    # 转Tensor
    # 先转标签
    Sampled_Data_Dict['Label'] = torch.FloatTensor(Sampled_Data_Dict['Label'])
    for tmp_feature_type in All_Feature_Type_Name_list:
        Sampled_Data_Dict['Feature_Dict'][tmp_feature_type] = torch.FloatTensor(Sampled_Data_Dict['Feature_Dict'][tmp_feature_type])
            
    if args_cuda:
        Sampled_Data_Dict['Label'] = Sampled_Data_Dict['Label'].cuda()
        for tmp_feature_type in All_Feature_Type_Name_list:
            Sampled_Data_Dict['Feature_Dict'][tmp_feature_type] = Sampled_Data_Dict['Feature_Dict'][tmp_feature_type].cuda()
            
            
    return Sampled_Data_Dict


# In[14]:


# # # 计算消耗内存
# Train_Data_Dict = Merge_SeHGNN_Data(KG_train_time_monthly_range_list)
# sub_data_dict = sample_for_SeHGNN(Train_Data_Dict, np.arange(0, Train_Data_Dict['Label'].shape[0]))

# tmp_feat_mem = 0
# for tmp_feature_type in sub_data_dict['Feature_Dict']:
#     print(tmp_feature_type, sub_data_dict['Feature_Dict'][tmp_feature_type].shape)
#     tmp_feat_mem = tmp_feat_mem + sub_data_dict['Feature_Dict'][tmp_feature_type].element_size() * sub_data_dict['Feature_Dict'][tmp_feature_type].nelement()
    
# print('Feature Memory:', tmp_feat_mem/(1024**2))

# # 计算消耗时间
# curr_time = datetime.now()

# for tmp_i in range(3):
#     sampled_label_index = sample_random_index_with_portion(Train_Data_Dict, 4096, 0.5)
#     Sampled_Data_Dict = sample_for_SeHGNN(Train_Data_Dict, sampled_label_index)
    
# curr_time2 = datetime.now()
# print(curr_time2-curr_time) 


# # 模型

# In[15]:


# # 残差网络，用于转换特征，以及输出最终预测结果
# class DNN(nn.Module):
#     def __init__(self, input_size, output_size, dropout = 0.5):
#         super().__init__()
#         self.dense = nn.Linear(input_size, output_size)
#         self.dropout = nn.Dropout(dropout)
#         self.activation = nn.Tanh()
#         self.LayerNorm = nn.LayerNorm(output_size)

#     def forward(self, hidden_states):
#         hidden_states = self.dense(hidden_states)
#         hidden_states = self.dropout(hidden_states)
#         hidden_states = self.activation(hidden_states)
#         hidden_states = self.LayerNorm(hidden_states)
#         return hidden_states

# class Res_DNN_layer(nn.Module):
#     def __init__(self, hidden_size, dropout, num_DNN):
#         super().__init__()
#         self.multi_DNN = nn.ModuleList([DNN(hidden_size, hidden_size, dropout) for _ in range(num_DNN)])
#         self.activation = nn.Tanh()
#         self.LayerNorm = nn.LayerNorm(hidden_size)

#     def forward(self, hidden_states):
        
#         hidden_states_shortcut = hidden_states
#         for i,layer_module in enumerate(self.multi_DNN):
#             hidden_states = layer_module(hidden_states)
#         hidden_states = hidden_states_shortcut + hidden_states
#         hidden_states = self.activation(hidden_states)
#         hidden_states = self.LayerNorm(hidden_states)
        
#         return hidden_states

# class Res_DNN(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, dropout, num_Res_DNN, num_DNN):
#         super().__init__()
#         # 先将数据降维
#         self.prepare = nn.Linear(input_size, hidden_size) 
        
#         # 再导入两轮3层Res_DNN
#         self.multi_Res_DNN = nn.ModuleList([Res_DNN_layer(hidden_size, dropout, num_DNN) for _ in range(num_Res_DNN)])
        
#         # 输出层，简单的一个线性层，从hidden_size映射到num_labels
#         self.classifier = nn.Linear(hidden_size, output_size) 
        
#     def forward(self, input_ids):
#         hidden_states = self.prepare(input_ids)
        
#         for i,layer_module in enumerate(self.multi_Res_DNN):
#             hidden_states = layer_module(hidden_states)
        
#         hidden_states = self.classifier(hidden_states)
    
#         return hidden_states


# In[16]:


# from torch.nn import TransformerEncoder, TransformerEncoderLayer
# device = torch.device('cuda' if args_cuda else 'cpu')

# class TransformerLayerNorm(nn.Module):
#     def __init__(self, hidden_size, eps=1e-12):
#         """Construct a layernorm module in the TF style (epsilon inside the square root).
#         """
#         super(TransformerLayerNorm, self).__init__()
#         self.weight = nn.Parameter(torch.ones(hidden_size))
#         self.bias = nn.Parameter(torch.zeros(hidden_size))
#         self.variance_epsilon = eps

#     def forward(self, x):
#         #归一化
#         u = x.mean(-1, keepdim=True)                         #找均值，-1表示找倒数第一维的均值
#         s = (x - u).pow(2).mean(-1, keepdim=True)            #计算方差
#         x = (x - u) / torch.sqrt(s + self.variance_epsilon)  #除以标准差，将整体拉回到【-1,1】区间
#         return self.weight * x + self.bias

# class TransformerEmbeddings(nn.Module):
#     """Construct the embeddings from word, position and token_type embeddings.
#     """
#     def __init__(self, nfeat, seq_length, dropout):
#         super(TransformerEmbeddings, self).__init__()
#         self.position_embeddings = nn.Embedding(seq_length, nfeat)
#         self.seq_length = seq_length
#         self.LayerNorm = TransformerLayerNorm(nfeat, eps=1e-12)
#         self.dropout = nn.Dropout(dropout)

#     def forward(self, features_embeddings):
#         position_ids = torch.arange(self.seq_length, dtype=torch.long, device = device)
        
#         position_embeddings = self.position_embeddings(position_ids)
        
#         position_embeddings = position_embeddings.unsqueeze(1).expand(features_embeddings.size(0),
#                                                                        features_embeddings.size(1),
#                                                                        features_embeddings.size(2))

#         embeddings = features_embeddings + position_embeddings
#         embeddings = self.LayerNorm(embeddings)
#         embeddings = self.dropout(embeddings)
#         return embeddings

# class TransformerModel(nn.Module):
#     def __init__(self, nfeat, nhead, nhid, nlayers, nout, seq_length, dropout=0.5):
#         super(TransformerModel, self).__init__()
        
# #         print(nfeat,nhead,nhid)
        
#         self.get_embed = TransformerEmbeddings(nfeat=nfeat, seq_length=seq_length, dropout = dropout)
        
#         encoder_layers = TransformerEncoderLayer(nfeat, nhead, nhid, dropout)
#         self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        
#         self.decoder = nn.Linear(nfeat, nout)

#     # h = features_embeddings
#     def forward(self, h):
#         h = self.get_embed(h)
# #         print('h_embed_size:', h.size())
        
#         h = self.transformer_encoder(h)
#         h = self.decoder(h)
#         return h


# In[17]:


# class SeHGNN(nn.Module):
#     def __init__(self, node_feature_hid_len, metapath_level_nhid, metapath_level_nhead, 
#                  semantic_level_nhid, semantic_level_nhead, dropout=0.5):
#         super().__init__()
        
#         # 随机dropout函数，防止过拟合
#         self.dropout = nn.Dropout(dropout)
        
#         # 对原始特征进行映射
#         self.node_feature_transform_dict = {}
#         for tmp_node_type in Node_Type_to_Feature_len_dict:
#             tmp = nn.Linear(Node_Type_to_Feature_len_dict[tmp_node_type], node_feature_hid_len)
#             self.node_feature_transform_dict[tmp_node_type] = tmp
#             self.add_module('Node_feature_transform_{}'.format(tmp_node_type), tmp)
        
#         # 各元路径总体的transformer
#         self.semantic_level_transformer = TransformerModel(node_feature_hid_len, semantic_level_nhead, semantic_level_nhid,
#                                                            1, semantic_level_nhid, len(All_Feature_Type_Name_list), dropout)
        
#         # 最后的输出函数
#         self.output_dense = Res_DNN(semantic_level_nhid, semantic_level_nhid, 1, dropout, 2, 3)
        
#         self.activation = nn.Sigmoid()
        
#     def forward(self, input_feature_dict):
#         # 获取各关系对应的特征
#         metapath_h_feature_list = []
        
#         # 对原始特征进行转换
#         for tmp_feature_type in All_Feature_Type_Name_list:
#             tmp_node_type = Feature_Node_Type_dict[tmp_feature_type]
#             tmp_transferred_h = self.node_feature_transform_dict[tmp_node_type](input_feature_dict['Feature_Dict'][tmp_feature_type])
            
#             metapath_h_feature_list.append(tmp_transferred_h)
        
#         ###################################################################################################################
#         # 合并各元路径的结果
#         tmp_metapath_h_feature_stack = torch.stack(metapath_h_feature_list, 1)
#         tmp_metapath_h_feature_stack = self.dropout(tmp_metapath_h_feature_stack)
            
#         # 通过semantic_level transformer
#         tmp_metapath_h_feature_stack = tmp_metapath_h_feature_stack.permute(1, 0, 2)
#         tmp_metapath_h_feature_stack = self.semantic_level_transformer(tmp_metapath_h_feature_stack)
#         tmp_metapath_h_feature_stack = tmp_metapath_h_feature_stack[-1,:,:].squeeze()
            
#         # 输出最终结果
#         h_output = self.output_dense(tmp_metapath_h_feature_stack)
#         h_output = h_output.squeeze()
#         h_output = self.activation(h_output)
        
#         return h_output


# In[18]:


from kg_model.SeHGNN import SeHGNN 

# 建立模型
model = SeHGNN(Node_Type_to_Feature_len_dict, All_Feature_Type_Name_list, Feature_Node_Type_dict, 
               Model_Config_dict['node_feature_hid_len'], Model_Config_dict['metapath_level_nhid'],
               Model_Config_dict['metapath_level_nhead'], Model_Config_dict['metapath_level_nlayers'], 
               Model_Config_dict['num_Res_DNN'], Model_Config_dict['each_Res_DNN_num'], dropout = Model_Config_dict['dropout'])
if args_cuda:
    model.cuda()
# print(model)

# 优化器
optimizer = optim.Adam(model.parameters(), lr = Model_Config_dict['learning_rate'], weight_decay = Model_Config_dict['weight_decay'])

# 损失函数
BCE_loss = torch.nn.BCELoss()


# # 评价函数

# In[19]:


def top_k_accuracy_score(y_true, y_score, k):
    sorted_pred = np.argsort(y_score)
    sorted_pred = sorted_pred[::-1]
    sorted_pred = sorted_pred[:k]

    hits = y_true[sorted_pred]
    
    return np.sum(hits)/k


# In[20]:


import sklearn
from sklearn.metrics import roc_auc_score, average_precision_score

def evaluate(model, source_Data_Dict, need_transfer = True):
    model.eval()
    with torch.no_grad():    
        if need_transfer:
            # 分割成各个小数据
            h_output_list = []
            for sample_start in tqdm(range(0, source_Data_Dict['Label'].shape[0], Model_Config_dict['eval_sample_size'])):
                sample_end = sample_start + Model_Config_dict['eval_sample_size']
                if sample_end > source_Data_Dict['Label'].shape[0]:
                    sample_end = source_Data_Dict['Label'].shape[0]
            
                Sampled_Data_Dict = sample_for_SeHGNN(source_Data_Dict, 
                                                               np.arange(sample_start, sample_end))

                h_output = model(Sampled_Data_Dict)
                
                h_output_list.append(h_output)
                
            h_output = torch.cat(h_output_list)
            
            source_data_label = torch.FloatTensor(source_Data_Dict['Label'])
            if args_cuda:
                source_data_label = source_data_label.cuda()
        else:
            h_output = model(source_Data_Dict)

            source_data_label = source_Data_Dict['Label']
    
    loss = BCE_loss(h_output, source_data_label)
    
#     # 计算roc-auc值和AVG_Pre值
#     if args_cuda:
#         roc_auc = roc_auc_score(source_data_label.data.cpu().numpy(), h_output.data.cpu().numpy())
#         average_precision = average_precision_score(source_data_label.data.cpu().numpy(), h_output.data.cpu().numpy())
#     else:
#         roc_auc = roc_auc_score(source_data_label.data.numpy(), h_output.data.numpy())
#         average_precision = average_precision_score(source_data_label.data.numpy(), h_output.data.numpy())
    
    # 计算roc-auc值和AVG_Pre值
    if args_cuda:
        source_data_label_np = source_data_label.data.cpu().numpy()
        h_output_squeeze_np = h_output.data.cpu().numpy()
    else:
        source_data_label_np = source_data_label.data.numpy()
        h_output_squeeze_np = h_output.data.numpy()
    
    fpr, tpr, thresholds = roc_curve(source_data_label_np, h_output_squeeze_np)
    roc_auc = auc(fpr, tpr)
    
    precision, recall, _ = precision_recall_curve(source_data_label_np, h_output_squeeze_np)
    average_precision = auc(recall, precision)
    
    top_k_acc_dict = {}
    for tmp_aim_k in [100, 500, 1000, 5000, 10000]:
        top_k_acc_dict[tmp_aim_k] = top_k_accuracy_score(source_data_label_np, h_output_squeeze_np, k = tmp_aim_k)
    
    return loss.item(), roc_auc, average_precision, top_k_acc_dict


# # 训练模型

# In[21]:


from kg_lib.utils import mkdir

localtime = time.strftime("%m-%d-%H:%M", time.localtime())

temp_train_list_name = ('Train_' + KG_train_time_range_list[0].strftime('%Y_%m_%d') + '-' 
                        + KG_train_time_range_list[1].strftime('%Y_%m_%d') + '/')

# 模型参数的输出文件夹
tmp_model_parameter_output_dir = '../Model_Parameter/SeHGNN/' + localtime + '_' + temp_train_list_name + '/'
mkdir('../Model_Parameter')
mkdir('../Model_Parameter/SeHGNN/')
mkdir(tmp_model_parameter_output_dir)

# 各评价指标的变化情况
metric_list_dict = {}
metric_list_dict['train'] = {'loss':[], 'roc_auc':[], 'avg_precision':[]}
metric_list_dict['val'] = {'loss':[], 'roc_auc':[], 'avg_precision':[]}
metric_list_dict['test'] = {'loss':[], 'roc_auc':[], 'avg_precision':[]}

# 最优roc_auc
best_roc_auc = 0


# In[22]:


from tqdm import tqdm

for epoch in range(Model_Config_dict['train_epoch']):
    # 多少轮查看一次效果
    pbar = tqdm(range(Model_Config_dict['sample_num_for_eval']))
    for sample_index in pbar:
        # 先采样
        sampled_label_index = sample_random_index_with_portion(Train_Data_Dict, 
                                                               Model_Config_dict['train_sample_size'], 
                                                               Model_Config_dict['train_pos_sample_percent'])
        Sampled_Data_Dict = sample_for_SeHGNN(Train_Data_Dict, sampled_label_index)
        
        # 再训练模型
        model.train()

        h_output = model(Sampled_Data_Dict)

        loss = BCE_loss(h_output, Sampled_Data_Dict['Label'])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 查看其他指标
        if args_cuda:
            roc_auc = roc_auc_score(Sampled_Data_Dict['Label'].data.cpu().numpy(), h_output.data.cpu().numpy())
            average_precision = average_precision_score(Sampled_Data_Dict['Label'].data.cpu().numpy(), h_output.data.cpu().numpy())
        else:
            roc_auc = roc_auc_score(Sampled_Data_Dict['Label'].data.numpy(), h_output.data.numpy())
            average_precision = average_precision_score(Sampled_Data_Dict['Label'].data.numpy(), h_output.data.numpy())
        
        metric_list_dict['train']['loss'].append(loss.item())
        metric_list_dict['train']['roc_auc'].append(roc_auc)
        metric_list_dict['train']['avg_precision'].append(average_precision)
        
        pbar.set_postfix({'loss':loss.item(), 'roc_auc': roc_auc, 'avg_precision': average_precision})
    
    # 查看效果
    train_loss, train_roc_auc, train_average_precision, train_top_k_acc_dict = evaluate(model, Train_Data_Dict)
    val_loss, val_roc_auc, val_average_precision, val_top_k_acc_dict = evaluate(model, Validation_Data_Dict)
    test_loss, test_roc_auc, test_average_precision, test_top_k_acc_dict = evaluate(model, Test_Data_Dict)
    
    print('Epoch:', epoch)
    
    metric_list_dict['val']['loss'].append(val_loss)
    metric_list_dict['val']['roc_auc'].append(val_roc_auc)
    metric_list_dict['val']['avg_precision'].append(val_average_precision)
    print('Validation - loss:', val_loss, 'roc_auc:',val_roc_auc, 'avg_precision:',val_average_precision, val_top_k_acc_dict)
    
    metric_list_dict['test']['loss'].append(test_loss)
    metric_list_dict['test']['roc_auc'].append(test_roc_auc)
    metric_list_dict['test']['avg_precision'].append(test_average_precision)
    print('Test - loss:', test_loss, 'roc_auc:', test_roc_auc, 'avg_precision:', test_average_precision, test_top_k_acc_dict)
    
    # 达到最优效果时存储一次模型
    if val_roc_auc > best_roc_auc:
        best_roc_auc = val_roc_auc
        torch.save(model.state_dict(), tmp_model_parameter_output_dir + 'model_parameter' + '_best_roc_auc_' + ("%.4f" % best_roc_auc) + '.pt')


# # 打印趋势

# In[ ]:


plt.plot(range(len(metric_list_dict['train']['loss'])), metric_list_dict['train']['loss'])
plt.plot(range(len(metric_list_dict['val']['loss'])), metric_list_dict['val']['loss'])
plt.plot(range(len(metric_list_dict['test']['loss'])), metric_list_dict['test']['loss'])
plt.show()


# In[ ]:


plt.plot(range(len(metric_list_dict['val']['roc_auc'])), metric_list_dict['val']['roc_auc'])
plt.plot(range(len(metric_list_dict['test']['roc_auc'])), metric_list_dict['test']['roc_auc'])
plt.show()


# In[ ]:


plt.plot(range(len(metric_list_dict['val']['avg_precision'])), metric_list_dict['val']['avg_precision'])
plt.plot(range(len(metric_list_dict['test']['avg_precision'])), metric_list_dict['test']['avg_precision'])
plt.show()


# In[ ]:




