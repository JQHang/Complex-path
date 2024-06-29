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
import json
from datetime import datetime
import sys
sys.path.append("..")

import random
from random import sample
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.optim as optim
import math

import matplotlib.pyplot as plt


# In[2]:


#显示所有列
pd.set_option('display.max_columns', None)
#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)


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


from kg_lib.utils import divid_range_list_to_monthly_list

# 目标时间及月份(左闭右开)
KG_train_time_range_list = [datetime(2022, 11, 1), datetime(2022, 12, 1)]
KG_validation_time_range_list = [datetime(2022, 12, 1), datetime(2022, 12, 10)]
KG_test_time_range_list = [datetime(2022, 12, 10), datetime(2022, 12, 20)]

KG_train_time_monthly_range_list = divid_range_list_to_monthly_list(KG_train_time_range_list)
print('KG_train_time_monthly_range_list:', KG_train_time_monthly_range_list)

KG_validation_time_monthly_range_list = divid_range_list_to_monthly_list(KG_validation_time_range_list)
print('KG_validation_time_monthly_range_list:', KG_validation_time_monthly_range_list)

KG_test_time_monthly_range_list = divid_range_list_to_monthly_list(KG_test_time_range_list)
print('KG_test_time_monthly_range_list:', KG_test_time_monthly_range_list)

# 要计算的时间区间
all_aim_time_monthly_range_list = (KG_train_time_monthly_range_list + KG_validation_time_monthly_range_list + 
                                   KG_test_time_monthly_range_list)


# In[7]:


from kg_lib.utils import read_json_config_file

Target_Node_Type = 'Mobile_Node'

# 数据来源描述
# Output_Columns_Type = "Head_And_Tail"
Output_Columns_Type = "ALL_Nodes"
# Output_Columns_Type = "ALL_Nodes_And_Edges"

Feature_Month_Range = 1

# data_source_description_str = '01_23-签约标签9至12月训练数据-' + Output_Columns_Type + '格式'
# data_source_description_str = '06_13-签约标签4月复杂度测试数据-过去' + str(Feature_Month_Range) + '个月的特征-' + Output_Columns_Type + '格式'
data_source_description_str = '07_13-签约标签11至12月数据-过去' + str(Feature_Month_Range) + '个月的特征-' + Output_Columns_Type + '格式'
print(data_source_description_str)


# In[8]:


Model_Config_dict = {}
Model_Config_dict['train_sample_size'] = 2048
Model_Config_dict['eval_sample_size'] = 100000
Model_Config_dict['train_pos_sample_percent'] = 0.1
Model_Config_dict['train_epoch'] = 1000
Model_Config_dict['early_stop'] = 20
Model_Config_dict['sample_num_for_eval'] = 100
# Model_Config_dict['learning_rate'] = 0.0001
Model_Config_dict['learning_rate'] = 0.0001
# Model_Config_dict['weight_decay'] = 1e-4
Model_Config_dict['weight_decay'] = 0

Model_Config_dict['node_feature_hid_len'] = 128
Model_Config_dict['metapath_level_nhid'] = 128
Model_Config_dict['num_Res_DNN'] = 1
Model_Config_dict['each_Res_DNN_num'] = 2
# Model_Config_dict['dropout'] = 0.2
Model_Config_dict['dropout'] = 0.1

print(Model_Config_dict)


# # 预处理数据，并存储或读取已存在的数据（从而方便调节参数）

# In[9]:


from kg_lib.Get_MAGNN_Required_Data import Get_MAGNN_Required_Data_and_Store

# 各时间区间对应的数据
time_range_to_Result_Data_dict = {}

# 各类型节点对应的特征长度
node_type_to_feature_len_dict = {}

# 涉及到的全部元路径
all_meta_path_list = []

for tmp_aim_time_monthly_range in all_aim_time_monthly_range_list:
    time_range_str = ('Time_Range:' + str(tmp_aim_time_monthly_range))
    Result_Data_dict = Get_MAGNN_Required_Data_and_Store(data_source_description_str, time_range_str, aim_node_type = Target_Node_Type, 
                                                         Meta_path_drop_list = [])
    
    if len(node_type_to_feature_len_dict.keys()) == 0:
        for tmp_node_type in Result_Data_dict['Feature']:
            node_type_to_feature_len_dict[tmp_node_type] = Result_Data_dict['Feature'][tmp_node_type].shape[1]
        
        all_meta_path_list = list(Result_Data_dict['Adj'].keys())
    
    time_range_to_Result_Data_dict[time_range_str] = Result_Data_dict


# # 根据样本数目，按比例随机取样

# In[10]:


def sample_random_index_with_portion(Result_Data_dict, sample_size = 3000, positive_percent = 0.2):
    tmp_pos_sample_size = math.ceil(sample_size * positive_percent)
    tmp_neg_sample_size = (sample_size - tmp_pos_sample_size)

    # 获取正样本的序号
    tmp_pos_sample_index_np = np.argwhere(Result_Data_dict['Label'] == 1).T[0]

    # 随机选取指定数目的正样本的序号
    tmp_sub_pos_sample_index_np = np.random.choice(tmp_pos_sample_index_np, size = tmp_pos_sample_size, replace = False)

    # 获取负样本的序号
    tmp_neg_sample_index_np = np.argwhere(Result_Data_dict['Label'] == 0).T[0]

    # 随机选取指定数目的负样本的序号
    tmp_sub_neg_sample_index_np = np.random.choice(tmp_neg_sample_index_np, size = tmp_neg_sample_size, replace = False)

    # 合并两组序号
    tmp_sampled_label_index = np.concatenate((tmp_sub_pos_sample_index_np, tmp_sub_neg_sample_index_np))
    
    return tmp_sampled_label_index

# sampled_label_index = sample_random_index_with_portion(time_range_to_Result_Data_dict[time_range_str])


# # 根据取样结果，获取子图，并转Tensor

# In[13]:


def sample_sub_graph_with_label_index(Result_Data_dict, tmp_head_node_type, tmp_sampled_label_index):
    sub_graph_data_dict = {}
    sub_graph_data_dict['Feature'] = {}
    sub_graph_data_dict['Feature_Node_Type'] = {}
    sub_graph_data_dict['Adj'] = {}
    
    # 取出对应的标签、转为tensor，并看情况放入cuda
    sub_graph_data_dict['Label'] = torch.FloatTensor(Result_Data_dict['Label'][tmp_sampled_label_index])
    if args_cuda:
        sub_graph_data_dict['Label'] = sub_graph_data_dict['Label'].cuda()
        
    # 取出对应的index号
    tmp_sampled_index = Result_Data_dict['Label_Index'][tmp_sampled_label_index].astype(int)
    
    # 获取目标点原始特征
    tmp_head_node_feature = Result_Data_dict['Feature'][tmp_head_node_type][tmp_sampled_index]
    sub_graph_data_dict['Feature']['src_feat'] = torch.FloatTensor(tmp_head_node_feature)
    sub_graph_data_dict['Feature_Node_Type']['src_feat'] = tmp_head_node_type
    if args_cuda:
        sub_graph_data_dict['Feature']['src_feat'] = sub_graph_data_dict['Feature']['src_feat'].cuda()
        
    # 对各元路径对应的邻接表分批处理
    for tmp_meta_path_name in Result_Data_dict['Adj']:
        # print('处理元路径:', tmp_meta_path_name)
        
        # 只保留起始点被采样了的相关边
        tmp_sampled_adj_mask = np.isin(Result_Data_dict['Adj'][tmp_meta_path_name][0], tmp_sampled_index)
        tmp_sampled_adj = Result_Data_dict['Adj'][tmp_meta_path_name][:, tmp_sampled_adj_mask]
        
        # 如果总边数为0
        if tmp_sampled_adj.shape[1] == 0:
            # 保存空的邻接表（做运算时就会特征自动全给0）
            sub_graph_data_dict['Adj'][tmp_meta_path_name] = torch.LongTensor(np.array([[] for x in range(tmp_sampled_adj.shape[0])]))
            
            if args_cuda:
                sub_graph_data_dict['Adj'][tmp_meta_path_name] = sub_graph_data_dict['Adj'][tmp_meta_path_name].cuda()
                
        else:
            sub_graph_data_dict['Adj'][tmp_meta_path_name] = []
            sub_graph_data_dict['Feature'][tmp_meta_path_name] = {}
            sub_graph_data_dict['Feature_Node_Type'][tmp_meta_path_name] = [Target_Node_Type]
            
            # 将起始点序号转化为其在全部采样点中的序号
            tmp_index_trans_dict = dict(zip(tmp_sampled_index, range(len(tmp_sampled_index))))
            tmp_head_new_index = np.vectorize(tmp_index_trans_dict.get)(tmp_sampled_adj[0])
            
            sub_graph_data_dict['Adj'][tmp_meta_path_name].append(torch.LongTensor(tmp_head_new_index))
            if args_cuda:
                sub_graph_data_dict['Adj'][tmp_meta_path_name][0] = sub_graph_data_dict['Adj'][tmp_meta_path_name][0].cuda()
                
            # 获取路径上涉及的全部点对应的特征
            for tmp_path_i in range(1, tmp_sampled_adj.shape[0]):
                tmp_tail_node_index = np.unique(tmp_sampled_adj[tmp_path_i])
                tmp_tail_node_index.sort()

                # 获取终止点对应特征(该关系涉及的全部点)
                tmp_tail_node_type = Result_Data_dict['Adj_Node_Type'][tmp_meta_path_name][tmp_path_i]
                tmp_tail_node_feature = Result_Data_dict['Feature'][tmp_tail_node_type][tmp_tail_node_index]

                sub_graph_data_dict['Feature'][tmp_meta_path_name][tmp_path_i] = torch.FloatTensor(tmp_tail_node_feature)
                sub_graph_data_dict['Feature_Node_Type'][tmp_meta_path_name].append(tmp_tail_node_type)

                # 将终止点序号转化为其在全部终止点中的序号
                tmp_index_trans_dict = dict(zip(tmp_tail_node_index, range(len(tmp_tail_node_index))))
                tmp_tail_new_index = np.vectorize(tmp_index_trans_dict.get)(tmp_sampled_adj[tmp_path_i])
                
                sub_graph_data_dict['Adj'][tmp_meta_path_name].append(torch.LongTensor(tmp_tail_new_index))

                if args_cuda:
                    sub_graph_data_dict['Feature'][tmp_meta_path_name][tmp_path_i] = sub_graph_data_dict['Feature'][tmp_meta_path_name][tmp_path_i].cuda()
                    sub_graph_data_dict['Adj'][tmp_meta_path_name][tmp_path_i] = sub_graph_data_dict['Adj'][tmp_meta_path_name][tmp_path_i].cuda()
                    
    return sub_graph_data_dict

# sampled_label_index = sample_random_index_with_portion(time_range_to_Result_Data_dict[time_range_str])
# sub_graph_data_dict = sample_sub_graph_with_label_index(Result_Data_dict, Target_Node_Type, sampled_label_index)


# In[17]:


# from tqdm import tqdm

# # 计算消耗内存
# sub_graph_data_dict = sample_sub_graph_with_label_index(Processed_HAN_Data_dict, Label_Data_Config_dict['Node_Type'], 
#                                                         np.arange(0, Processed_HAN_Data_dict['Label'].shape[0]))

# tmp_feat_mem = 0
# for tmp_path in sub_graph_data_dict['Feature']:
#     tmp_feat_mem = tmp_feat_mem + sub_graph_data_dict['Feature'][tmp_path].element_size() * sub_graph_data_dict['Feature'][tmp_path].nelement()

# tmp_adj_mem = 0
# for tmp_path in sub_graph_data_dict['Adj']:
#     tmp_adj_mem = tmp_adj_mem + sub_graph_data_dict['Adj'][tmp_path].element_size() * sub_graph_data_dict['Adj'][tmp_path].nelement()

# print('Feature Memory:', tmp_feat_mem/(1024**2))
# print('Adj Memory:', tmp_adj_mem/(1024**2))
# print('All Memory:', (tmp_feat_mem + tmp_adj_mem)/(1024**2))

# tmp_time_list = []

# for tmp in tqdm(range(20)):
#     # 计算消耗时间
#     curr_time = datetime.now()

#     for tmp_i in range(3):
#         sampled_label_index = sample_random_index_with_portion(Processed_HAN_Data_dict, sample_size = 4096, positive_percent = 0.5)
#         sub_graph_data_dict = sample_sub_graph_with_label_index(Processed_HAN_Data_dict, Label_Data_Config_dict['Node_Type'], 
#                                                                 sampled_label_index)

#     curr_time2 = datetime.now()
    
# #     print(curr_time2-curr_time)

#     tmp_time_list.append((curr_time2-curr_time).total_seconds())
    
# print(np.mean(tmp_time_list), np.var(tmp_time_list))


# # 模型

# In[58]:


from importlib import reload 
import kg_model
reload(kg_model.MAGNN)
from kg_model.MAGNN import MAGNN

# 建立模型
model = MAGNN(node_type_to_feature_len_dict, all_meta_path_list, node_feature_hid_len = Model_Config_dict['node_feature_hid_len'], 
            metapath_level_nhid = Model_Config_dict['metapath_level_nhid'], num_Res_DNN = Model_Config_dict['num_Res_DNN'], 
            each_Res_DNN_num = Model_Config_dict['each_Res_DNN_num'], dropout = Model_Config_dict['dropout'])

if args_cuda:
    model.cuda()
# print(model)

# 优化器
optimizer = optim.Adam(model.parameters(), lr = Model_Config_dict['learning_rate'], weight_decay = Model_Config_dict['weight_decay'])

# 损失函数
BCE_loss = torch.nn.BCELoss()


# ## 评价函数

# In[59]:


def top_k_accuracy_score(y_true, y_score, k):
    sorted_pred = np.argsort(y_score)
    sorted_pred = sorted_pred[::-1]
    sorted_pred = sorted_pred[:k]

    hits = y_true[sorted_pred]
    
    return np.sum(hits)/k


# In[60]:


import sklearn
from sklearn.metrics import roc_auc_score, average_precision_score

def evaluate(model, source_Data_Dict, need_transfer = True):
    model.eval()
    with torch.no_grad():    
        if need_transfer:
            # 分割成各个小数据
            h_output_squeeze_list = []
            for sample_start in tqdm(range(0, source_Data_Dict['Label'].shape[0], Model_Config_dict['eval_sample_size'])):
                sample_end = sample_start + Model_Config_dict['eval_sample_size']
                if sample_end > source_Data_Dict['Label'].shape[0]:
                    sample_end = source_Data_Dict['Label'].shape[0]
            
                Sampled_Data_Dict = sample_sub_graph_with_label_index(source_Data_Dict, Target_Node_Type, 
                                                                      np.arange(sample_start,sample_end))

                h_output = model(Sampled_Data_Dict)
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


# In[64]:


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
            source_Data_Dict = time_range_to_Result_Data_dict[time_range_str]
            for sample_start in tqdm(range(0, source_Data_Dict['Label'].shape[0], Model_Config_dict['eval_sample_size'])):
                sample_end = sample_start + Model_Config_dict['eval_sample_size']
                if sample_end > source_Data_Dict['Label'].shape[0]:
                    sample_end = source_Data_Dict['Label'].shape[0]
            
                Sampled_Data_Dict = sample_sub_graph_with_label_index(source_Data_Dict, Target_Node_Type, 
                                                                      np.arange(sample_start,sample_end))

                h_output = model(Sampled_Data_Dict)
                h_output_squeeze = torch.squeeze(h_output)
                
                h_output_squeeze_list.append(h_output_squeeze)
            
            source_data_label = torch.FloatTensor(source_Data_Dict['Label'])
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


# ## 训练模型

# In[62]:


from kg_lib.utils import mkdir

localtime = time.strftime("%m-%d-%H:%M", time.localtime())

temp_train_list_name = ('Train_' + KG_train_time_range_list[0].strftime('%Y_%m_%d') + '-' 
                        + KG_train_time_range_list[1].strftime('%Y_%m_%d') + '/')

# 模型参数的输出文件夹
tmp_model_parameter_output_dir = '../../Model_Parameter/MAGNN/' + localtime + '_' + temp_train_list_name + '/'
mkdir('../../Model_Parameter')
mkdir('../../Model_Parameter/MAGNN/')
mkdir(tmp_model_parameter_output_dir)

# 各评价指标的变化情况
metric_list_dict = {}
metric_list_dict['train'] = {'loss':[], 'roc_auc':[], 'avg_precision':[]}
metric_list_dict['val'] = {'loss':[], 'roc_auc':[], 'avg_precision':[]}
metric_list_dict['test'] = {'loss':[], 'roc_auc':[], 'avg_precision':[]}

# 最优roc_auc
best_roc_auc = 0

# 累计未优化次数
early_stop_count = 0


# In[65]:


from tqdm import tqdm

for epoch in range(Model_Config_dict['train_epoch']):
    # 多少轮查看一次效果
    pbar = tqdm(range(Model_Config_dict['sample_num_for_eval']))
    for sample_index in pbar:
        # 先采样
        time_range_str = ('Time_Range:' + str(KG_train_time_monthly_range_list[sample_index%len(KG_train_time_monthly_range_list)]))
        sampled_label_index = sample_random_index_with_portion(time_range_to_Result_Data_dict[time_range_str],
                                                               Model_Config_dict['train_sample_size'],
                                                               Model_Config_dict['train_pos_sample_percent'])
        Sampled_Data_Dict = sample_sub_graph_with_label_index(time_range_to_Result_Data_dict[time_range_str], 
                                                              Target_Node_Type, 
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
            roc_auc = roc_auc_score(Sampled_Data_Dict['Label'].data.cpu().numpy(), h_output_squeeze.data.cpu().numpy())
            average_precision = average_precision_score(Sampled_Data_Dict['Label'].data.cpu().numpy(), h_output_squeeze.data.cpu().numpy())
        else:
            roc_auc = roc_auc_score(Sampled_Data_Dict['Label'].data.numpy(), h_output_squeeze.data.numpy())
            average_precision = average_precision_score(Sampled_Data_Dict['Label'].data.numpy(), h_output_squeeze.data.numpy())
        
        metric_list_dict['train']['loss'].append(loss.item())
        metric_list_dict['train']['roc_auc'].append(roc_auc)
        metric_list_dict['train']['avg_precision'].append(average_precision)
        
        pbar.set_postfix({'loss':loss.item(), 'roc_auc': roc_auc, 'avg_precision': average_precision})
    
    # 查看效果
    train_loss, train_roc_auc, train_average_precision, train_top_k_acc_dict = evaluate_multi_time(model, KG_train_time_monthly_range_list)
    val_loss, val_roc_auc, val_average_precision, val_top_k_acc_dict = evaluate_multi_time(model, KG_validation_time_monthly_range_list)
    test_loss, test_roc_auc, test_average_precision, test_top_k_acc_dict = evaluate_multi_time(model, KG_test_time_monthly_range_list)
    
    print('Epoch:', epoch)
    
    metric_list_dict['train']['loss'].append(train_loss)
    metric_list_dict['train']['roc_auc'].append(train_roc_auc)
    metric_list_dict['train']['avg_precision'].append(train_average_precision)
    print('Train - loss:', train_loss, 'roc_auc:',train_roc_auc, 'avg_precision:',train_average_precision, train_top_k_acc_dict)
    
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
        early_stop_count = 0
        best_roc_auc = val_roc_auc
        
        torch.save(model.state_dict(), tmp_model_parameter_output_dir + 'model_parameter' + '_best_roc_auc_' + ("%.4f" % best_roc_auc) + '.pt')
    else:
        early_stop_count = early_stop_count + 1
        print("Early Stop Count:", early_stop_count)
        
        if early_stop_count >= Model_Config_dict['early_stop']:
            break


# # 打印训练趋势和评价指标

# In[21]:


plt.plot(range(len(metric_list_dict['val']['loss'])), metric_list_dict['val']['loss'])
plt.plot(range(len(metric_list_dict['test']['loss'])), metric_list_dict['test']['loss'])
plt.show()


# In[22]:


plt.plot(range(len(metric_list_dict['val']['roc_auc'])), metric_list_dict['val']['roc_auc'])
plt.plot(range(len(metric_list_dict['test']['roc_auc'])), metric_list_dict['test']['roc_auc'])
plt.show()


# In[23]:


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


val_loss, val_roc_auc, val_average_precision, val_top_k_acc_dict = evaluate_multi_time(model, KG_validation_time_monthly_range_list, print_figure = True)
print(val_roc_auc, val_average_precision, val_top_k_acc_dict)

test_loss, test_roc_auc, test_average_precision, test_top_k_acc_dict = evaluate_multi_time(model, KG_test_time_monthly_range_list, print_figure = True)
print(test_roc_auc, test_average_precision, test_top_k_acc_dict)


# In[ ]:




