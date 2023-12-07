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

import sklearn
from sklearn.metrics import roc_auc_score, average_precision_score, mean_absolute_error, mean_squared_error
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import precision_recall_curve, roc_curve
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


# In[6]:


# spark相关配置
from pyspark.sql import SparkSession
# import pyspark.sql.functions as F
from pyspark.sql.types import *
from pyspark import StorageLevel
from pyspark.sql import HiveContext,SparkSession
from pyspark.sql.window import Window
from pyspark.sql.functions import col, row_number

from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql import Row
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType

# 创建spark_session
os.environ['PYSPARK_PYTHON']="/usr/local/anaconda3/bin/python"

def Start_Spark():
    Spark_Session = SparkSession.builder                     .appName("kg_spark_for_model")                     .enableHiveSupport()                     .config("spark.sql.shuffle.partitions", "2000")                     .config("spark.sql.broadcastTimeout","3600")                    .config("spark.driver.memory", "200g")                     .config("spark.executor.memory", "40g")                    .config("spark.yarn.appMasterEnv.yarn.nodemanager.container-executor.class", "DockerLinuxContainer")                    .config("spark.executorEnv.yarn.nodemanager.container-executor.class", "DockerLinuxContainer")                    .config("spark.yarn.appMasterEnv.yarn.nodemanager.docker-container-executor.image-name", 
                            "bdp-docker.jd.com:5000/wise_mart_bag:latest")\
                    .config("spark.executorEnv.yarn.nodemanager.docker-container-executor.image-name", 
                            "bdp-docker.jd.com:5000/wise_mart_bag:latest")\
                    .config("spark.sql.crossJoin.enabled", "true")\
                    .config("spark.driver.maxResultSize", "40g")\
                    .config("spark.sql.autoBroadcastJoinThreshold","-1")\
                    .getOrCreate()
    
    return Spark_Session


# # 读取参数

# In[7]:


from kg_lib.utils import mkdir
from kg_lib.utils import read_json_config_file

# 目标节点类型
Aim_Node_Type = 'Mobile_Node'

# 数据保存的位置
Feature_Data_From_Online_Store_dir = '../../Data/07_05-签约标签11至1月ComplexPath全特征训练数据/'
# Feature_Data_From_Online_Store_dir = '../../Data/07_05-热线索标签11至1月ComplexPath全特征训练数据/'
mkdir('../../Data/')
mkdir(Feature_Data_From_Online_Store_dir)

Feature_Month_Range = 1

Model_Config_dict = {}

Model_Config_dict['Data_Regenerate'] = False

Model_Config_dict['Feature_Preprocess_Type_List'] = ['Norm', 'Std']

# Model_Config_dict['Meta_path_drop_list'] = ['user_pin_related_to_mobile_by_set', 'mobile_company_site', 'mobile_receive_from_site']
# Model_Config_dict['Meta_path_drop_list'] = ['user_pin_related_to_mobile_by_set', 'mobile_company_site', 'mobile_receive_from_site', 
#                                             'company_related_to_industry']
Model_Config_dict['Meta_path_drop_list'] = {}

# Model_Config_dict['Meta_path_Column_drop_dict'] = {'company_related_to_industry': [6]}
# Model_Config_dict['Meta_path_Column_drop_dict'] = {'user_pin_related_to_mobile_by_express': [0], 'mobile_send_from_site': [0, 2], 
#                                                    'company_related_to_industry': [0, 2, 4, 5, 6]}
Model_Config_dict['Meta_path_Column_drop_dict'] = {}

Model_Config_dict['train_sample_size'] = 4096
Model_Config_dict['eval_sample_size'] = 100000
Model_Config_dict['train_pos_sample_percent'] = 0.1
Model_Config_dict['train_epoch'] = 10000
Model_Config_dict['sample_num_for_eval'] = 40
Model_Config_dict['early_stop'] = 1000
Model_Config_dict['learning_rate'] = 0.00005
Model_Config_dict['weight_decay'] = 0.00005

Model_Config_dict['node_feature_hid_len'] = 512
Model_Config_dict['node_feature_out_len'] = 512
Model_Config_dict['metapath_level_nhid'] = 512
Model_Config_dict['metapath_level_nout'] = 512
Model_Config_dict['metapath_level_nhead'] = 1
Model_Config_dict['metapath_level_nlayers'] = 1
Model_Config_dict['semantic_level_nhid'] = 128
Model_Config_dict['semantic_level_nout'] = 128
Model_Config_dict['semantic_level_nhead'] = 1
Model_Config_dict['semantic_level_nlayers'] = 1
Model_Config_dict['num_Res_DNN'] = 1
Model_Config_dict['each_Res_DNN_num'] = 2
Model_Config_dict['dropout'] = 0.5

print(Model_Config_dict)

Label_Data_Config_file = './kg_config/Target_Node_Dataset_Config/Sign_Label_Train_Target_Data_Config_2023_05_20.json'
# Label_Data_Config_file = './kg_config/Target_Node_Dataset_Config/Hot_Clue_Label_Train_Target_Data_Config_VLDB.json'
Feature_Data_From_Online_Config_file = './kg_config/Feature_Data_From_Online_Config/Complex_Path/Feature_Data_From_Online_Config_2023_05_26_Complex.json'

Label_Data_Config_dict = read_json_config_file(Label_Data_Config_file)
Feature_Data_From_Online_Config_dict = read_json_config_file(Feature_Data_From_Online_Config_file)


# In[8]:


from kg_lib.utils import divid_range_list_to_monthly_list, divid_range_list_to_monthly_first_day_list

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

# 获取全部涉及到的时间区间
# all_aim_time_range = [KG_train_time_range_list[0].strftime("%Y-%m-%d"), KG_test_time_range_list[1].strftime("%Y-%m-%d")]
all_aim_time_range = ['2022-11-01', '2023-02-01']
print('all_aim_time_range:', all_aim_time_range)


# # 预处理数据，存储或读取已存在的数据，并合并结果

# In[ ]:


from kg_lib.Get_ComplexPath_Required_Data import Get_ComplexPath_Data_From_Online
from kg_lib.Target_Node_Dataloader import get_aim_UID_with_label_rdd, Extend_feature_time_for_aim_UID
from py4j.protocol import Py4JJavaError

spark_restart_count = 0

start_time = datetime.now()
print(start_time)

while True:
    try:
        # 启动spark
        Spark_Session = Start_Spark()

        time_range_str = ('Time_Range:' + str(all_aim_time_range))

        # 根据目标时间创建对应文件夹
        Feature_Data_From_Online_Time_Store_dir = Feature_Data_From_Online_Store_dir + time_range_str + '/'
        mkdir(Feature_Data_From_Online_Time_Store_dir)
        print('目标时间数据存储文件夹', Feature_Data_From_Online_Time_Store_dir)

        # 获取整个目标时间对应的目标点
        tmp_aim_UID_info_dict = get_aim_UID_with_label_rdd(Spark_Session, Label_Data_Config_dict, all_aim_time_range[0], 
                                                           all_aim_time_range[1], Feature_Data_From_Online_Time_Store_dir, True)

        # 基于Feature_Month_Range扩展目标点对应特征时间
        tmp_aim_UID_info_dict = Extend_feature_time_for_aim_UID(Spark_Session, tmp_aim_UID_info_dict, Feature_Month_Range)

        # 获取目标特征
        Result_Data_dict = Get_ComplexPath_Data_From_Online(Spark_Session, tmp_aim_UID_info_dict, Feature_Data_From_Online_Config_dict,
                                                           Feature_Data_From_Online_Time_Store_dir, 
                                                           Model_Config_dict['Data_Regenerate'], Model_Config_dict['Meta_path_drop_list'],
                                                           Model_Config_dict['Meta_path_Column_drop_dict'],
                                                           Model_Config_dict['Feature_Preprocess_Type_List'])

        Meta_Path_to_Complex_Path_dict = {}
        for tmp_meta_path_name in Result_Data_dict['Meta_Path_Feature'].keys():
            Meta_Path_to_Complex_Path_dict[tmp_meta_path_name] = []
            for tmp_complex_path_name in Result_Data_dict['Meta_Path_Feature'][tmp_meta_path_name]:
                Meta_Path_to_Complex_Path_dict[tmp_meta_path_name].append(tmp_complex_path_name)
                
        All_Meta_Path_Name_list = list(Result_Data_dict['Meta_Path_Feature'].keys())
        Node_Type_to_Feature_len_dict = Result_Data_dict['Node_Type_to_Feature_len'].copy()
        Meta_Path_Column_Type_dict = Result_Data_dict['Meta_Path_Column_Type'].copy()
        
        break
        
    except Py4JJavaError:
        print('*******************************************************************************')
        
        Spark_Session.stop()
        Spark_Session._instantiatedContext = None
        

        interrupt_time = datetime.now()
        print('pyspark异常中断，时间:', interrupt_time)
        
        # 如果不是0-9点间的中断且重启次数小于20，则进行重启
        if interrupt_time.hour < 9:
            print('0-9点无法运算，故开始等待')

            time_sleep = (8 - interrupt_time.hour)*3600 + (60 - interrupt_time.minute) * 60
            
            print('休眠时间', time_sleep)
            
            time.sleep(time_sleep)
            
            print('已到早上9点，重启运算')
            
        elif spark_restart_count > 20:
            print('重启超过20次，故终止')

            break
            
        else:
            spark_restart_count = spark_restart_count + 1
            print('重启Spark并重新开始运算')
            
        print('*******************************************************************************')
        
end_time = datetime.now()
print(end_time)
print('花费时间:', end_time - start_time)


# # 取出各时间段对应的数据

# In[ ]:


def Get_Required_Complex_Path_Transformer_Data_by_Time(tmp_aim_UID_info_dict, Result_Data_dict, aim_time_monthly_range_list):
    Required_Time_Data_dict = {}
    
    # 获取目标时间内的点对应的index
    tmp_aim_node_pd = tmp_aim_UID_info_dict['Data_pd']
    tmp_required_time_node_index = ((tmp_aim_node_pd['Source_Time'] >= aim_time_monthly_range_list[0].strftime("%Y-%m-%d")) &
                                    (tmp_aim_node_pd['Source_Time'] < aim_time_monthly_range_list[1].strftime("%Y-%m-%d")))
    
    # 获取对应点的标签
    Required_Time_Data_dict['Label'] = torch.FloatTensor(Result_Data_dict['Label'][tmp_required_time_node_index].values)
    
    # 获取目标点本身对应特征
    Required_Time_Data_dict['Start_Node_Feature'] = torch.FloatTensor(Result_Data_dict['Start_Node_Feature_List'][0][tmp_required_time_node_index].values)
    
    # 获取各元路径的各列对应的特征
    Required_Time_Data_dict['Complex_Path_Feature'] = {}
    for tmp_meta_path_name in Result_Data_dict['Meta_Path_Feature']:
        Required_Time_Data_dict['Complex_Path_Feature'][tmp_meta_path_name] = {}
        for tmp_complex_path_name in Result_Data_dict['Meta_Path_Feature'][tmp_meta_path_name]:
            Required_Time_Data_dict['Complex_Path_Feature'][tmp_meta_path_name][tmp_complex_path_name] = {}
            for tmp_index in Meta_Path_Column_Type_dict[tmp_meta_path_name].keys():
                tmp_meta_path_column_feature = Result_Data_dict['Meta_Path_Feature'][tmp_meta_path_name][tmp_complex_path_name][tmp_index][tmp_required_time_node_index].values

                Required_Time_Data_dict['Complex_Path_Feature'][tmp_meta_path_name][tmp_complex_path_name][tmp_index] = torch.FloatTensor(tmp_meta_path_column_feature)
            
    # 查看正负样本总数
    All_Label_np = Required_Time_Data_dict['Label'].data.numpy().astype(int)
    print('Positive Sample Count:', np.sum(All_Label_np == 1))
    print('Negative Sample Count:', np.sum(All_Label_np == 0))
    
    # 记录正负样本对应序号
    Required_Time_Data_dict['Pos_Label_loc'] = np.argwhere(All_Label_np == 1).T[0]
    Required_Time_Data_dict['Neg_Label_loc'] = np.argwhere(All_Label_np == 0).T[0]
    
    return Required_Time_Data_dict


# In[ ]:


Train_Data_Dict = Get_Required_Complex_Path_Transformer_Data_by_Time(tmp_aim_UID_info_dict, Result_Data_dict, KG_train_time_range_list)
Valid_Data_Dict = Get_Required_Complex_Path_Transformer_Data_by_Time(tmp_aim_UID_info_dict, Result_Data_dict, KG_validation_time_range_list)
Test_Data_Dict = Get_Required_Complex_Path_Transformer_Data_by_Time(tmp_aim_UID_info_dict, Result_Data_dict, KG_test_time_range_list)


# # 根据需求随机采样

# In[ ]:


def sample_random_index_with_portion(Source_Data_Dict, sample_size, positive_percent):
    tmp_pos_sample_size = math.ceil(sample_size * positive_percent)
    tmp_neg_sample_size = (sample_size - tmp_pos_sample_size)
    
    # 随机选取指定数目的正样本的序号
    tmp_sub_pos_sample_index_np = np.random.choice(Source_Data_Dict['Pos_Label_loc'], size = tmp_pos_sample_size)
#     tmp_sub_pos_sample_index_np = np.random.choice(Source_Data_Dict['Pos_Label_loc'], size = tmp_pos_sample_size, replace = False)
    
    # 随机选取指定数目的负样本的序号
    tmp_sub_neg_sample_index_np = np.random.choice(Source_Data_Dict['Neg_Label_loc'], size = tmp_neg_sample_size)
#     tmp_sub_neg_sample_index_np = np.random.choice(Source_Data_Dict['Neg_Label_loc'], size = tmp_neg_sample_size, replace = False)

    # 合并两组序号
    tmp_sampled_label_index = np.concatenate((tmp_sub_pos_sample_index_np, tmp_sub_neg_sample_index_np))
    
    return tmp_sampled_label_index


# In[ ]:


def sample_for_Complex_Path_Transformer(Source_Data_Dict, tmp_sampled_index_np):
    Sampled_Data_Dict = {}
    
    # 提取对应标签
    Sampled_Data_Dict['Label'] = Source_Data_Dict['Label'][tmp_sampled_index_np]

    # 提取目标节点对应特征
    Sampled_Data_Dict['Start_Node_Feature'] = Source_Data_Dict['Start_Node_Feature'][tmp_sampled_index_np, :]
    
    # 提取元路径对应特征
    Sampled_Data_Dict['Complex_Path_Feature'] = {}
    for tmp_meta_path_name in Source_Data_Dict['Complex_Path_Feature']:
        Sampled_Data_Dict['Complex_Path_Feature'][tmp_meta_path_name] = {}
        for tmp_complex_path_name in Source_Data_Dict['Complex_Path_Feature'][tmp_meta_path_name]:
            Sampled_Data_Dict['Complex_Path_Feature'][tmp_meta_path_name][tmp_complex_path_name] = {}
            for tmp_index in Source_Data_Dict['Complex_Path_Feature'][tmp_meta_path_name][tmp_complex_path_name]:
                tmp_sampled_tensor = Source_Data_Dict['Complex_Path_Feature'][tmp_meta_path_name][tmp_complex_path_name][tmp_index][tmp_sampled_index_np, :]
                Sampled_Data_Dict['Complex_Path_Feature'][tmp_meta_path_name][tmp_complex_path_name][tmp_index] = tmp_sampled_tensor
            
    # 放入cuda
    if args_cuda:
        Sampled_Data_Dict['Label'] = Sampled_Data_Dict['Label'].cuda()
        Sampled_Data_Dict['Start_Node_Feature'] = Sampled_Data_Dict['Start_Node_Feature'].cuda()
        for tmp_meta_path_name in Sampled_Data_Dict['Complex_Path_Feature']:
            for tmp_complex_path_name in Sampled_Data_Dict['Complex_Path_Feature'][tmp_meta_path_name]:
                for tmp_index in Sampled_Data_Dict['Complex_Path_Feature'][tmp_meta_path_name][tmp_complex_path_name]:
                    tmp_sampled_tensor_cuda = Sampled_Data_Dict['Complex_Path_Feature'][tmp_meta_path_name][tmp_complex_path_name][tmp_index].cuda()
                    Sampled_Data_Dict['Complex_Path_Feature'][tmp_meta_path_name][tmp_complex_path_name][tmp_index] = tmp_sampled_tensor_cuda
            
            
    return Sampled_Data_Dict


# # 模型

# In[ ]:


from kg_model.Complex_Path_Transformer import Complex_Path_Transformer
from torch.optim.lr_scheduler import LambdaLR, StepLR

# 建立模型
model = Complex_Path_Transformer(Node_Type_to_Feature_len_dict, Meta_Path_to_Complex_Path_dict, Meta_Path_Column_Type_dict,
                                  node_feature_hid_len = Model_Config_dict['node_feature_hid_len'], 
                                  metapath_level_nhid = Model_Config_dict['metapath_level_nhid'], 
                                  metapath_level_nhead = Model_Config_dict['metapath_level_nhead'],
                                  metapath_level_nlayers = Model_Config_dict['metapath_level_nlayers'],
                                  semantic_level_nhid = Model_Config_dict['semantic_level_nhid'], 
                                  semantic_level_nhead = Model_Config_dict['semantic_level_nhead'],
                                  semantic_level_nlayers = Model_Config_dict['semantic_level_nlayers'],
                                  num_Res_DNN = Model_Config_dict['num_Res_DNN'],
                                  each_Res_DNN_num = Model_Config_dict['each_Res_DNN_num'],
                                  dropout = Model_Config_dict['dropout'])
if args_cuda:
    model.cuda()
# print(model)

# 优化器
optimizer = optim.Adam(model.parameters(), lr = Model_Config_dict['learning_rate'], weight_decay = Model_Config_dict['weight_decay'])

# 动态调节学习率
# scheduler = StepLR(optimizer, step_size = 25, gamma = 0.5)

# def get_linear_schedule(optimizer, base_lr, num_training_steps, last_epoch=-1):
#     def lr_lambda(current_step):
#         return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps)))
#     return LambdaLR(optimizer, lr_lambda, last_epoch)

# scheduler = get_linear_schedule(optimizer, Model_Config_dict['learning_rate'], Model_Config_dict['train_epoch'])


# In[ ]:


# # 实际标签为负，但被预测为正，影响很大
# # 实际标签为正，但被预测为负，影响没那么大
# # 标签为1的概率小就小了，标签为0的样本，预测概率越大，损失值权重越大
# class FocalLoss(nn.Module):
#     def __init__(self, gamma=2):
#         super(FocalLoss, self).__init__()
        
#         self.gamma = gamma

#     def forward(self, preds, labels):
#         # 计算每一个样本的损失值
#         BCE_loss = torch.nn.functional.binary_cross_entropy(preds, labels, reduction='none')
        
# #         return torch.mean(BCE_loss)
        
#         # prevents nans when probability 0
#         pt = torch.exp(-BCE_loss)
        
# #         print(BCE_loss)
# #         print(1-pt)
        
#         # 只针对标签为0的
#         mask_0 = 1 - labels
        
# #         print((1-pt)**self.gamma*mask_0 + 0.1)
# #         print(((1.5 - pt)**self.gamma)*mask_0 + 1 * labels)
    
#         # focal_loss
# #         F_loss = (1-pt)**self.gamma * BCE_loss
# #         F_loss = ((1-pt)**self.gamma*mask_0 + 0.1) * BCE_loss
#         F_loss = (((1 + 0.5 - pt)**self.gamma)*mask_0 + 1 * labels) * BCE_loss

#         return torch.mean(F_loss)

# 损失函数
BCE_loss = torch.nn.BCELoss()
# BCE_loss = FocalLoss()


# # 评价函数

# In[ ]:


def top_k_accuracy_score(y_true, y_score, k):
    sorted_pred = np.argsort(y_score)
    sorted_pred = sorted_pred[::-1]
    sorted_pred = sorted_pred[:k]

    hits = y_true[sorted_pred]
    
    return np.sum(hits)/k


# In[ ]:


import sklearn
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.metrics import precision_recall_curve, roc_curve
from sklearn.metrics import auc

def evaluate(model, source_Data_Dict, need_transfer = True, print_figure = False):
    model.eval()
    with torch.no_grad():    
        if need_transfer:
            # 分割成各个小数据
            h_output_list = []
            for sample_start in tqdm(range(0, source_Data_Dict['Label'].shape[0], Model_Config_dict['eval_sample_size'])):
                sample_end = sample_start + Model_Config_dict['eval_sample_size']
                if sample_end > source_Data_Dict['Label'].shape[0]:
                    sample_end = source_Data_Dict['Label'].shape[0]
            
                Sampled_Data_Dict = sample_for_Complex_Path_Transformer(source_Data_Dict, np.arange(sample_start, sample_end))

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
    for tmp_aim_k in [100, 500, 1000, 2000, 5000, 10000, 20000]:
        top_k_acc_dict[tmp_aim_k] = top_k_accuracy_score(source_data_label_np, h_output_squeeze_np, k = tmp_aim_k)
    
    return loss.item(), roc_auc, average_precision, top_k_acc_dict


# # 训练模型

# In[ ]:


from kg_lib.utils import mkdir

localtime = time.strftime("%m-%d-%H:%M", time.localtime())

temp_train_list_name = ('Train_' + KG_train_time_range_list[0].strftime('%Y_%m_%d') + '-' 
                        + KG_train_time_range_list[1].strftime('%Y_%m_%d'))

# 模型参数的输出文件夹
tmp_model_parameter_output_dir = '../../Model_Parameter/Meta_Path_Transformer/' + localtime + '_' + temp_train_list_name + '/'
mkdir('../../Model_Parameter')
mkdir('../../Model_Parameter/Meta_Path_Transformer/')
mkdir(tmp_model_parameter_output_dir)

# 保存基本信息
f = open(tmp_model_parameter_output_dir + 'Node_Type_to_Feature_len_dict.json', 'w')
f.write(json.dumps(Node_Type_to_Feature_len_dict, sort_keys = False, indent = 4))
f.close()

np.save(tmp_model_parameter_output_dir + 'All_Meta_Path_Name_list.npy', np.array(All_Meta_Path_Name_list))

f = open(tmp_model_parameter_output_dir + 'Meta_Path_Column_Type_dict.json', 'w')
f.write(json.dumps(Meta_Path_Column_Type_dict, sort_keys = False, indent = 4))
f.close()

# 各评价指标的变化情况
metric_list_dict = {}
metric_list_dict['sample'] = {'loss':[], 'roc_auc':[], 'avg_precision':[], 'top_k_acc_dict':[]}

metric_list_dict['train'] = {'loss':[], 'roc_auc':[], 'avg_precision':[], 'top_k_acc_dict':[]}
metric_list_dict['val'] = {'loss':[], 'roc_auc':[], 'avg_precision':[], 'top_k_acc_dict':[]}
metric_list_dict['test'] = {'loss':[], 'roc_auc':[], 'avg_precision':[], 'top_k_acc_dict':[]}

# 最优roc_auc
best_roc_auc = 0

# 累计未优化次数
early_stop_count = 0


# In[ ]:


from tqdm import tqdm

for epoch in range(Model_Config_dict['train_epoch']):
    # 多少轮查看一次效果
    pbar = tqdm(range(Model_Config_dict['sample_num_for_eval']))
    for sample_index in pbar:
        # 先采样
        sampled_label_index = sample_random_index_with_portion(Train_Data_Dict, Model_Config_dict['train_sample_size'], 
                                                               Model_Config_dict['train_pos_sample_percent'])
        Sampled_Data_Dict = sample_for_Complex_Path_Transformer(Train_Data_Dict, sampled_label_index)
        
        # 再训练模型
        model.train()

        h_output = model(Sampled_Data_Dict)

        loss = BCE_loss(h_output, Sampled_Data_Dict['Label'])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 查看其他指标
        if args_cuda:
            source_data_label_np = Sampled_Data_Dict['Label'].data.cpu().numpy()
            h_output_squeeze_np = h_output.data.cpu().numpy()
        else:
            source_data_label_np = Sampled_Data_Dict['Label'].data.numpy()
            h_output_squeeze_np = h_output.data.numpy()
        
        roc_auc = roc_auc_score(source_data_label_np, h_output_squeeze_np)
        average_precision = average_precision_score(source_data_label_np, h_output_squeeze_np)
        top_k_acc_dict = {}
        for tmp_aim_k in [100, 500, 1000, 5000, 10000]:
            top_k_acc_dict[tmp_aim_k] = top_k_accuracy_score(source_data_label_np, h_output_squeeze_np, k = tmp_aim_k)
        
        metric_list_dict['sample']['loss'].append(loss.item())
        metric_list_dict['sample']['roc_auc'].append(roc_auc)
        metric_list_dict['sample']['avg_precision'].append(average_precision)
        metric_list_dict['sample']['top_k_acc_dict'].append(top_k_acc_dict)
        
        pbar.set_postfix({'loss':loss.item(), 'roc_auc': roc_auc, 'avg_precision': average_precision})
    
    # 更改学习率
#     scheduler.step()
    
    # 查看效果
    train_loss, train_roc_auc, train_average_precision, train_top_k_acc_dict = evaluate(model, Train_Data_Dict)
    val_loss, val_roc_auc, val_average_precision, val_top_k_acc_dict = evaluate(model, Valid_Data_Dict)
    test_loss, test_roc_auc, test_average_precision, test_top_k_acc_dict = evaluate(model, Test_Data_Dict)
    
    print('Epoch:', epoch)
    
    metric_list_dict['train']['loss'].append(train_loss)
    metric_list_dict['train']['roc_auc'].append(train_roc_auc)
    metric_list_dict['train']['avg_precision'].append(train_average_precision)
    metric_list_dict['train']['top_k_acc_dict'].append(train_top_k_acc_dict)
    print('Train - loss:', train_loss, 'roc_auc:',train_roc_auc, 'avg_precision:',train_average_precision, 
          'top_k_acc_dict:',train_top_k_acc_dict)
    
    metric_list_dict['val']['loss'].append(val_loss)
    metric_list_dict['val']['roc_auc'].append(val_roc_auc)
    metric_list_dict['val']['avg_precision'].append(val_average_precision)
    metric_list_dict['val']['top_k_acc_dict'].append(val_top_k_acc_dict)
    
    print('Validation - loss:', val_loss, 'roc_auc:',val_roc_auc, 'avg_precision:',val_average_precision, 
          'top_k_acc_dict:',val_top_k_acc_dict)
    
    metric_list_dict['test']['loss'].append(test_loss)
    metric_list_dict['test']['roc_auc'].append(test_roc_auc)
    metric_list_dict['test']['avg_precision'].append(test_average_precision)
    metric_list_dict['test']['top_k_acc_dict'].append(test_top_k_acc_dict)
    
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

# In[ ]:


plt.plot(range(len(metric_list_dict['train']['loss'])), metric_list_dict['train']['loss'])
plt.plot(range(len(metric_list_dict['val']['loss'])), metric_list_dict['val']['loss'])
plt.plot(range(len(metric_list_dict['test']['loss'])), metric_list_dict['test']['loss'])
plt.show()


# In[ ]:


plt.plot(range(len(metric_list_dict['train']['roc_auc'])), metric_list_dict['train']['roc_auc'])
plt.plot(range(len(metric_list_dict['val']['roc_auc'])), metric_list_dict['val']['roc_auc'])
plt.plot(range(len(metric_list_dict['test']['roc_auc'])), metric_list_dict['test']['roc_auc'])
plt.show()


# In[ ]:


plt.plot(range(len(metric_list_dict['train']['avg_precision'])), metric_list_dict['train']['avg_precision'])
plt.plot(range(len(metric_list_dict['val']['avg_precision'])), metric_list_dict['val']['avg_precision'])
plt.plot(range(len(metric_list_dict['test']['avg_precision'])), metric_list_dict['test']['avg_precision'])
plt.show()


# In[ ]:


# 读取模型
temp_model_file_name_list = os.listdir(tmp_model_parameter_output_dir)
temp_model_file_name_list = [x for x in temp_model_file_name_list if 'model_parameter_' in x]
temp_model_file_name_list.sort(key = lambda i:int(re.search('_0.(\d+).pt',i).group(1)))
model_used_file_name = temp_model_file_name_list[-1]
print('Used Model:', tmp_model_parameter_output_dir + model_used_file_name)

model.load_state_dict(torch.load(tmp_model_parameter_output_dir + model_used_file_name))


# In[ ]:


train_loss, train_roc_auc, train_average_precision, train_top_k_acc_dict = evaluate(model, Train_Data_Dict, print_figure = True)
print(train_loss, train_roc_auc, train_average_precision, train_top_k_acc_dict)

val_loss, val_roc_auc, val_average_precision, val_top_k_acc_dict = evaluate(model, Valid_Data_Dict, print_figure = True)
print(val_loss, val_roc_auc, val_average_precision, val_top_k_acc_dict)

test_loss, test_roc_auc, test_average_precision, test_top_k_acc_dict = evaluate(model, Test_Data_Dict, print_figure = True)
print(test_loss, test_roc_auc, test_average_precision, test_top_k_acc_dict)


# In[ ]:




