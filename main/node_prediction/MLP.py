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


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
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
Label_Data_Config_file = './kg_config/Target_Node_Dataset_Config/Sign_Label_Train_Target_Data_Config_2023_01_01.json'
Label_Data_Config_dict = read_json_config_file(Label_Data_Config_file)

DNN_Config_dict = {}
DNN_Config_dict['train_sample_size'] = 5000
DNN_Config_dict['eval_sample_size'] = 10000
DNN_Config_dict['train_pos_sample_percent'] = 0.1
DNN_Config_dict['train_epoch'] = 1000
DNN_Config_dict['sample_num_for_eval'] = 100
DNN_Config_dict['learning_rate'] = 0.0001
DNN_Config_dict['weight_decay'] = 0

DNN_Config_dict['DNN_hid_len'] = 128
DNN_Config_dict['num_Res_DNN'] = 2
DNN_Config_dict['each_Res_DNN_num'] = 3
DNN_Config_dict['dropout'] = 0


# In[7]:


# 数据来源描述
Output_Columns_Type = "Head_And_Tail"
# Output_Columns_Type = "ALL_Nodes"
# Output_Columns_Type = "ALL_Nodes_And_Edges"

data_source_description_str = '01_10-9至12月训练数据-' + Output_Columns_Type + '格式'
print(data_source_description_str)

# ML文件结果描述
aim_ML_file_store_name = '01_12-当前全部元路径特征测试'


# In[8]:


from kg_lib.utils import divid_range_list_to_monthly_list

# 目标时间及月份(左闭右开)
KG_train_time_range_list = [datetime(2022, 9, 1), datetime(2022, 11, 10)]
KG_validation_time_range_list = [datetime(2022, 11, 10), datetime(2022, 12, 1)]    # 以哪个日期之后的数据作为验证集
KG_test_time_range_list = [datetime(2022, 12, 1), datetime(2022, 12, 20)]

KG_train_time_monthly_range_list = divid_range_list_to_monthly_list(KG_train_time_range_list)
print('KG_train_time_monthly_range_list:', KG_train_time_monthly_range_list)

KG_validation_time_monthly_range_list = divid_range_list_to_monthly_list(KG_validation_time_range_list)
print('KG_validation_time_monthly_range_list:', KG_validation_time_monthly_range_list)

KG_test_time_monthly_range_list = divid_range_list_to_monthly_list(KG_test_time_range_list)
print('KG_test_time_monthly_range_list:', KG_test_time_monthly_range_list)

# 全部要计算的时间区间
all_aim_time_monthly_range_list = (KG_train_time_monthly_range_list + KG_validation_time_monthly_range_list + 
                                   KG_test_time_monthly_range_list)


# # 预处理ML格式的数据

# In[9]:


from kg_lib.Get_ML_Required_Data import get_ML_required_pandas_data

# 预处理各元路径结果
time_range_to_Processed_ML_Data_dict = {}
for tmp_aim_time_monthly_range in all_aim_time_monthly_range_list:
    time_range_str = ('Time_Range:' + str(tmp_aim_time_monthly_range))
    Processed_ML_Data_dict = get_ML_required_pandas_data(data_source_description_str, time_range_str, aim_ML_file_store_name, 
                                                         aim_node_type = Label_Data_Config_dict['Node_Type'], Meta_path_drop_list = [])
    
    time_range_to_Processed_ML_Data_dict[time_range_str] = Processed_ML_Data_dict
    
    print('结果列数:', time_range_to_Processed_ML_Data_dict[time_range_str]['Feature'].shape)


# # 合并所需时间段的全部数据

# In[10]:


def Merge_ML_Data(aim_time_monthly_range_list):
    Return_Data_Dict = {}
    
    tmp_all_label_list = []
    tmp_all_feature_list = []
    for tmp_aim_time_monthly_range in aim_time_monthly_range_list:
        time_range_str = ('Time_Range:' + str(tmp_aim_time_monthly_range))
        tmp_all_label_list.append(time_range_to_Processed_ML_Data_dict[time_range_str]['Label'])
        tmp_all_feature_list.append(time_range_to_Processed_ML_Data_dict[time_range_str]['Feature'])
        
    # 先合并标签
    Return_Data_Dict['Label'] = pd.concat(tmp_all_label_list).values
    
    # 再合并特征
    tmp_all_feature_pd = pd.concat(tmp_all_feature_list)
    Return_Data_Dict['Feature'] = tmp_all_feature_pd.values
    
    # 保留特征列名
    Return_Data_Dict['Columns'] = list(tmp_all_feature_pd.columns)
    
    return Return_Data_Dict


# In[11]:


Train_Data_Dict = Merge_ML_Data(KG_train_time_monthly_range_list)
Validation_Data_Dict = Merge_ML_Data(KG_validation_time_monthly_range_list)
Test_Data_Dict = Merge_ML_Data(KG_test_time_monthly_range_list)

print("训练集维度：", Train_Data_Dict['Feature'].shape) 
print("验证集维度：", Validation_Data_Dict['Feature'].shape) 
print("测试集维度：", Test_Data_Dict['Feature'].shape) 


# # 模型

# In[12]:


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


# In[13]:


class Pure_Res_DNN_Classification(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, num_Res_DNN, num_DNN):
        super().__init__()

        self.Res_DNN = Res_DNN(input_size, hidden_size, 1, dropout, num_Res_DNN, num_DNN) 
        
        self.activation = nn.Sigmoid()
        
    def forward(self, feature):
        return self.activation(self.Res_DNN(feature).squeeze())


# In[14]:


# 建立模型
DNN_model = Pure_Res_DNN_Classification(Train_Data_Dict['Feature'].shape[1], DNN_Config_dict['DNN_hid_len'], DNN_Config_dict['dropout'], 
                                        DNN_Config_dict['num_Res_DNN'], DNN_Config_dict['each_Res_DNN_num'])

if args_cuda:
    DNN_model.cuda()

# 优化器
DNN_optimizer = optim.Adam(DNN_model.parameters(), lr = DNN_Config_dict['learning_rate'], weight_decay = DNN_Config_dict['weight_decay'])

# 损失函数
BCE_loss = torch.nn.BCELoss()


# # 采样函数

# In[15]:


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


# In[16]:


def sample_for_DNN(Source_Data_Dict, tmp_sampled_index_np):
    Sampled_Data_Dict = {}
    
    # 提取对应标签
    Sampled_Data_Dict['Label'] = Source_Data_Dict['Label'][tmp_sampled_index_np]

    # 提取对应特征
    Sampled_Data_Dict['Feature'] = Source_Data_Dict['Feature'][tmp_sampled_index_np]
        
    # 转Tensor
    # 先转标签
    Sampled_Data_Dict['Label'] = torch.FloatTensor(Sampled_Data_Dict['Label'])
    Sampled_Data_Dict['Feature'] = torch.FloatTensor(Sampled_Data_Dict['Feature'])
    if args_cuda:
        Sampled_Data_Dict['Label'] = Sampled_Data_Dict['Label'].cuda()
        Sampled_Data_Dict['Feature'] = Sampled_Data_Dict['Feature'].cuda()
    
    return Sampled_Data_Dict


# # 训练函数

# In[17]:


def evaluate(model, source_Data_Dict, need_transfer = True):
    model.eval()
    with torch.no_grad():    
        if need_transfer:
            # 分割成各个小数据
            h_output_squeeze_list = []
            for sample_start in tqdm(range(0, source_Data_Dict['Label'].shape[0], DNN_Config_dict['eval_sample_size'])):
                sample_end = sample_start + DNN_Config_dict['eval_sample_size']
                if sample_end > source_Data_Dict['Label'].shape[0]:
                    sample_end = source_Data_Dict['Label'].shape[0]
            
                Sampled_Data_Dict = sample_for_DNN(source_Data_Dict, np.arange(sample_start,sample_end))

                h_output = model(Sampled_Data_Dict['Feature'])
                h_output_squeeze = torch.squeeze(h_output)
                
                h_output_squeeze_list.append(h_output_squeeze)
                
            h_output_squeeze = torch.cat(h_output_squeeze_list)
            
            source_data_label = torch.FloatTensor(source_Data_Dict['Label'])
            if args_cuda:
                source_data_label = source_data_label.cuda()
        else:
            h_output = model(source_Data_Dict)
            h_output_squeeze = torch.squeeze(h_output)
            
            source_data_label = source_Data_Dict['Label']
    
    loss = BCE_loss(h_output_squeeze, source_data_label)
    
    # 计算roc-auc值和AVG_Pre值
    if args_cuda:
        roc_auc = roc_auc_score(source_data_label.data.cpu().numpy(), h_output_squeeze.data.cpu().numpy())
        average_precision = average_precision_score(source_data_label.data.cpu().numpy(), h_output_squeeze.data.cpu().numpy())
    else:
        roc_auc = roc_auc_score(source_data_label.data.numpy(), h_output_squeeze.data.numpy())
        average_precision = average_precision_score(source_data_label.data.numpy(), h_output_squeeze.data.numpy())
        
    return loss.item(), roc_auc, average_precision


# In[18]:


for epoch in range(DNN_Config_dict['train_epoch']):
    # 多少轮查看一次效果
    pbar = tqdm(range(DNN_Config_dict['sample_num_for_eval']))
    for sample_index in pbar:
        # 先采样
        source_Data_Dict = Train_Data_Dict
        sampled_label_index = sample_random_index_with_portion(source_Data_Dict, DNN_Config_dict['train_sample_size'], 
                                                               DNN_Config_dict['train_pos_sample_percent'])
        Sampled_Data_Dict = sample_for_DNN(source_Data_Dict, sampled_label_index)

        # 再训练模型
        DNN_model.train()

        h_output = DNN_model(Sampled_Data_Dict['Feature'])
        h_output_squeeze = torch.squeeze(h_output)
        
        loss = BCE_loss(h_output_squeeze, Sampled_Data_Dict['Label'])

        DNN_optimizer.zero_grad()
        loss.backward()
        DNN_optimizer.step()
        
        # 查看其他指标
        if args_cuda:
            roc_auc = roc_auc_score(Sampled_Data_Dict['Label'].data.cpu().numpy(), h_output_squeeze.data.cpu().numpy())
            average_precision = average_precision_score(Sampled_Data_Dict['Label'].data.cpu().numpy(), h_output_squeeze.data.cpu().numpy())
        else:
            roc_auc = roc_auc_score(Sampled_Data_Dict['Label'].data.numpy(), h_output_squeeze.data.numpy())
            average_precision = average_precision_score(Sampled_Data_Dict['Label'].data.numpy(), h_output_squeeze.data.numpy())
        
        pbar.set_postfix({'loss':loss.item(), 'roc_auc': roc_auc, 'avg_precision': average_precision})
    
    # 查看效果
    val_loss, val_roc_auc, val_average_precision = evaluate(DNN_model, Validation_Data_Dict)
    test_loss, test_roc_auc, test_average_precision = evaluate(DNN_model, Test_Data_Dict)
    
    print('Epoch:', epoch)
    print('Validation - loss:', val_loss, 'roc_auc:',val_roc_auc, 'avg_precision:',val_average_precision)
    print('Test - loss:', test_loss, 'roc_auc:', test_roc_auc, 'avg_precision:', test_average_precision)


# In[ ]:




