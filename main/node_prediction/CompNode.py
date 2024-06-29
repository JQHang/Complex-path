#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import io
import time
import copy
import re
import gc
import json
import os
import sys
import math
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from random import sample
from collections import defaultdict
from tqdm import tqdm
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta
from py4j.protocol import Py4JError, Py4JJavaError, Py4JNetworkError

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import torch.optim as optim

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

# Self packages path
Src_Dir_path = "../../../.."
sys.path.append(Src_Dir_path)

# Self packages
from Utils.Pyspark_utils import ResilientSparkRunner
from Utils.utils import read_json_config_file
from Utils.utils import mkdir
from Utils.utils import Log_save


# In[2]:


#显示所有列
pd.set_option('display.max_columns', None)

#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

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

if torch.cuda.is_available():
    print('cuda')
    args_cuda = True
else:
    print('cpu')
    args_cuda = False


# # 读取参数

# In[3]:


# 数据保存的位置
Data_Output_path = Src_Dir_path + "/.."

# Name for data
Data_Dir_Name = '/02_27_Complex_Path_V1'

Graph_Output_path = Data_Output_path +'/Graph'
mkdir(Graph_Output_path)

Feature_Data_From_Online_Store_dir = Graph_Output_path + Data_Dir_Name
mkdir(Feature_Data_From_Online_Store_dir)

# 目标节点类型
Aim_Node_Type = 'Mobile_Node'

Feature_Month_Range = 1

Label_Data_Config_file = '../../Config/Target_Node_Dataset_Config/Target_mobile_node_classification_1116.json'
Feature_Data_From_Online_Config_file = '../../Config/ComplexPath_Seq_Feature_Config/ComplexPath_Sequential_Feature_Config_V6.json'

Label_Data_Config_dict = read_json_config_file(Label_Data_Config_file)
Feature_Data_From_Online_Config_dict = read_json_config_file(Feature_Data_From_Online_Config_file)


# In[4]:


Model_Config_dict = {}

Model_Config_dict['Data_Regenerate'] = False

Model_Config_dict['Feature_Preprocess_Type_List'] = ['Norm']

Model_Config_dict['Meta_path_drop_list'] = {}

Model_Config_dict['Meta_path_Column_drop_dict'] = {}

Model_Config_dict['train_sample_size'] = 30000
Model_Config_dict['eval_sample_size'] = 150000
Model_Config_dict['train_pos_sample_percent'] = 0
Model_Config_dict['train_epoch'] = 10000
Model_Config_dict['sample_num_for_eval'] = 100
Model_Config_dict['early_stop'] = 2000
Model_Config_dict['learning_rate'] = 1e-4
Model_Config_dict['weight_decay'] = 1e-5

Model_Config_dict['hid_len'] = 300
Model_Config_dict['dropout'] = 0.5

print(Model_Config_dict)


# In[5]:


# 目标时间及月份(左闭右开)
Train_time_range_list = [datetime(2023, 3, 1), datetime(2023, 4, 1)]
Validation_time_range_list = [datetime(2023, 4, 1), datetime(2023, 4, 7)]
Test_time_range_list = [datetime(2023, 4, 7), datetime(2023, 4, 15)]

All_aim_time_range_dict = {'Train': Train_time_range_list,
                           'Val': Validation_time_range_list,
                           'Test': Test_time_range_list}

# 获取全部涉及到的时间区间
all_aim_time_range = [Train_time_range_list[0].strftime("%Y-%m-%d"), Test_time_range_list[1].strftime("%Y-%m-%d")]
# all_aim_time_range = ['2023-08-01', '2023-09-01']
print('all_aim_time_range:', all_aim_time_range)


# # Read Data To HDFS

# In[6]:


from Utils.Pyspark_utils import Data_In_HDFS_Path
from pyspark.sql import SparkSession
from Utils.ComplexPath_Data_Load_Pyspark import All_ComplexPath_Data_Load
from Utils.Target_Node_Dataloader import Read_Target_Node_with_Label
from py4j.protocol import Py4JJavaError

spark_restart_count = 0

start_time = datetime.now()
print(start_time)

# Spark_Runner = ResilientSparkRunner({"spark.default.parallelism": "1600", 
#                                      "spark.sql.shuffle.partitions": "8000", 
#                                      "spark.executor.cores": "4",
#                                      "spark.executor.instances": "400"})

Spark_Runner = ResilientSparkRunner(mode = 'local')

time_range_str = (str(all_aim_time_range[0]) + '-to-' + str(all_aim_time_range[1]))

# 根据目标时间创建对应文件夹
Feature_Data_From_Online_Time_Store_dir = Feature_Data_From_Online_Store_dir + '/' + time_range_str
mkdir(Feature_Data_From_Online_Time_Store_dir)
print('目标时间数据存储文件夹', Feature_Data_From_Online_Time_Store_dir)

# HDFS Store dir
HDFS_Store_dir = Data_In_HDFS_Path + Data_Dir_Name + '/' + time_range_str


# In[7]:


Complex_Path_Vector_len_dict = {"Target_Node": 1699,
                                "Mobile_to_Top_Total_Address": [4078, 4078],
                                "Mobile_to_Top_Total_Pin": [973],
                                "mobile_to_company_by_legal_person": [510, 234, 234, 234, 234]}


# In[8]:


def Complex_Path_Format_Transfer(spark, Target_Node_Info_Dict, Complex_Path_Pyspark_Data_Dict):
    # Path Sequential Feature 结果字典
    Result_Data_dict = {}
    
    # 记录各个node/edge对应的特征长度
    Result_Data_dict["Type_to_Feature_Len"] = {}
    
    # target node and order
    Aim_Node_type = Target_Node_Info_Dict['Node_Types'][0]
    Aim_Node_UID_Name = Aim_Node_type + '_UID'
    
    target_id_columns = [Aim_Node_UID_Name, 'Feature_Time']
    target_extra_info_columns = ['Label', 'Source_Time']

    # 只保留rdd中的目标节点列，去缩减特征表
    sample_with_label_df = Target_Node_Info_Dict['Data'].select(target_id_columns + target_extra_info_columns)
    
    sample_with_label_pd_file = Feature_Data_From_Online_Time_Store_dir + '/Sample.pkl'
    
    if Model_Config_dict['Data_Regenerate'] or not os.path.exists(sample_with_label_pd_file):
        # 转pandas
        sample_with_label_pd = sample_with_label_df.toPandas()

        # 保存标签的结果
        sample_with_label_pd.to_pickle(sample_with_label_pd_file)
    else:
        sample_with_label_pd = pd.read_pickle(sample_with_label_pd_file)
    
    Result_Data_dict["Sample"] = sample_with_label_pd
    
    ####################################################################################################################
    # Local File Store Path
    local_pd_store_dir = Feature_Data_From_Online_Time_Store_dir + '/Feature_Data'
    mkdir(local_pd_store_dir)
    
    # 确定要保留的特征列的列名
    feature_vector_columns = []
    if 'Raw' in Model_Config_dict['Feature_Preprocess_Type_List']:
        feature_vector_columns.append('Feature_Raw')
    elif 'Norm' in Model_Config_dict['Feature_Preprocess_Type_List']:
        feature_vector_columns.append('Feature_Normalized')
    elif 'Std' in Model_Config_dict['Feature_Preprocess_Type_List']:
        feature_vector_columns.append('Feature_Standard')
    
    ##############################################################################################################
    # 设定原始对应的存储位置
    target_node_local_pd_file = local_pd_store_dir + '/Target_Node_Feat.pkl'

    if Model_Config_dict['Data_Regenerate'] or not os.path.exists(target_node_local_pd_file):
        tmp_add_agg_data = Complex_Path_Pyspark_Data_Dict['Raw_Feature']['data'].select(target_id_columns + feature_vector_columns)

        # Join all the feature data to node with label
        target_node_df = Pyspark_Left_Join_and_Fill_Null_Vectors(spark, sample_with_label_df, 
                                                                 tmp_add_agg_data, target_id_columns, 
                                                                 feature_vector_columns,
                                                                 Complex_Path_Vector_len_dict['Target_Node'])

        # 合并Vector列
        print("Merge Feature Vectors:", feature_vector_columns)
        feature_vector_assembler = VectorAssembler(inputCols = feature_vector_columns, outputCol = "Feature_Vector")
        target_node_df = feature_vector_assembler.transform(target_node_df)
        target_node_df = target_node_df.drop(*feature_vector_columns)

        # 将vector转array
        print("vector to array")
        target_node_df = target_node_df.withColumn("Feature_Vector", vector_to_array(target_node_df["Feature_Vector"]))

        # 去重
        target_node_df = target_node_df.dropDuplicates(target_id_columns)

        # 转pandas
        target_node_pd = target_node_df.toPandas()

        # 确保pandas文件中顺序一致
        target_node_pd = sample_with_label_pd[target_id_columns].merge(target_node_pd, on = target_id_columns, how = 'left')

        # 保存结果
        target_node_pd.to_pickle(target_node_local_pd_file)

    else:
        target_node_pd = pd.read_pickle(target_node_local_pd_file)

    features_np = np.vstack(target_node_pd["Feature_Vector"].values)
        
    Result_Data_dict["Type_to_Feature_Len"][Aim_Node_type] = features_np.shape[1]
        
    Result_Data_dict["Raw_Feature"] = {}
    Result_Data_dict["Raw_Feature"]['data'] = features_np
    Result_Data_dict["Raw_Feature"]['node_type'] = Aim_Node_type
    Result_Data_dict["Raw_Feature"]['feature_comments'] = Complex_Path_Pyspark_Data_Dict['Raw_Feature']['feature_comments']

    ##############################################################################################################
    # 记录各个复杂路的结果
    Result_Data_dict["Complex_Paths"] = {}
    
    # 记录各个Meta_Path对应的节点类型，及包含的复杂路名称
    Result_Data_dict["Meta_Paths"] = {}
    
    # 依次处理各个复杂路
    for tmp_complex_path_name in Complex_Path_Pyspark_Data_Dict['Complex_Paths'].keys():
        
        print("Process ", tmp_complex_path_name)
        
        Result_Data_dict["Complex_Paths"][tmp_complex_path_name] = {}
        Result_Data_dict["Complex_Paths"][tmp_complex_path_name]['data'] = []
        
        # 记录complex_path类型(最好改名为type list)
        Result_Data_dict["Complex_Paths"][tmp_complex_path_name]['node_type'] = Complex_Path_Pyspark_Data_Dict['Complex_Paths'][tmp_complex_path_name]['node_type']
        
        # 基于该complex_path上的节点类型，给complex_path一个对应的meta-path分类
        meta_path_name = '__'.join(Complex_Path_Pyspark_Data_Dict['Complex_Paths'][tmp_complex_path_name]['node_type'])
        
        # 存储该complex_path对应的meta-path分类
        Result_Data_dict["Complex_Paths"][tmp_complex_path_name]['meta_path_type'] = meta_path_name
        
        # 记录该meta-path对应的各节点类型
        if meta_path_name not in Result_Data_dict["Meta_Paths"]:
            Result_Data_dict["Meta_Paths"][meta_path_name] = {}
            Result_Data_dict["Meta_Paths"][meta_path_name]['node_type'] = Complex_Path_Pyspark_Data_Dict['Complex_Paths'][tmp_complex_path_name]['node_type']
            Result_Data_dict["Meta_Paths"][meta_path_name]['complex_paths'] = [tmp_complex_path_name]
        else:
            if tmp_complex_path_name not in Result_Data_dict["Meta_Paths"][meta_path_name]['complex_paths']:
                Result_Data_dict["Meta_Paths"][meta_path_name]['complex_paths'].append(tmp_complex_path_name)
            
        # 依次处理路径中的各位数据
        for tmp_add_agg_index, tmp_add_agg_data in enumerate(Complex_Path_Pyspark_Data_Dict['Complex_Paths'][tmp_complex_path_name]['data']):
            path_feat_local_pd_file = local_pd_store_dir + f'/{tmp_complex_path_name}_{tmp_add_agg_index}.pkl'

            if Model_Config_dict['Data_Regenerate'] or not os.path.exists(path_feat_local_pd_file):
                tmp_add_agg_data = tmp_add_agg_data.select(target_id_columns + feature_vector_columns)

                path_feat_df = Pyspark_Left_Join_and_Fill_Null_Vectors(spark, sample_with_label_df, 
                                                                        tmp_add_agg_data, target_id_columns, feature_vector_columns,
                                                                        Complex_Path_Vector_len_dict[tmp_complex_path_name][tmp_add_agg_index])

                # 合并Vector列
                print("Merge Feature Vectors:", feature_vector_columns)
                feature_vector_assembler = VectorAssembler(inputCols = feature_vector_columns, outputCol = "Feature_Vector")
                path_feat_df = feature_vector_assembler.transform(path_feat_df)
                path_feat_df = path_feat_df.drop(*feature_vector_columns)

                # 将vector转array
                print("vector to array")
                path_feat_df = path_feat_df.withColumn("Feature_Vector", vector_to_array(path_feat_df["Feature_Vector"]))

                # 去重
                path_feat_df = path_feat_df.dropDuplicates(target_id_columns)

                # 转pandas
                path_feat_pd = path_feat_df.toPandas()

                # 确保pandas文件中顺序一致
                path_feat_pd = sample_with_label_pd[target_id_columns].merge(path_feat_pd, on = target_id_columns, how = 'left')

                # 保存结果
                path_feat_pd.to_pickle(path_feat_local_pd_file)
            else:
                path_feat_pd = pd.read_pickle(path_feat_local_pd_file)

            features_np = np.vstack(path_feat_pd["Feature_Vector"].values)
            Result_Data_dict["Complex_Paths"][tmp_complex_path_name]['data'].append(features_np)
            
            # 基于节点类型记录对应的特征长度
            node_type = Complex_Path_Pyspark_Data_Dict['Complex_Paths'][tmp_complex_path_name]['node_type'][tmp_add_agg_index]
            Result_Data_dict["Type_to_Feature_Len"][node_type] = features_np.shape[1]
            
    return Result_Data_dict


# In[9]:


import Utils.Pyspark_utils
from pyspark.ml.functions import vector_to_array
from Utils.Pyspark_utils import path_exists_on_hdfs, Pyspark_Merge_Vectors, DEFAULT_STORAGE_LEVEL
from Utils.Pyspark_utils import Pyspark_Left_Join_and_Fill_Null_Vectors, hdfs_create_marker_file, hdfs_read_marker_file
from pyspark.ml.feature import VectorAssembler

def Read_Target_Node_Feature(spark):
    # 获取整个目标时间对应的目标点
    Target_Node_Info_Dict = Read_Target_Node_with_Label(spark, Label_Data_Config_dict, All_aim_time_range_dict, 
                                                        HDFS_Store_dir, Regernerate = Model_Config_dict['Data_Regenerate'])
    
    # 获取目标特征到本地
    Complex_Path_Pyspark_Data_Dict = All_ComplexPath_Data_Load(spark, Target_Node_Info_Dict, 
                                                               Feature_Data_From_Online_Config_dict, HDFS_Store_dir, 
                                                               Model_Config_dict['Data_Regenerate'], 
                                                               Model_Config_dict['Meta_path_drop_list'], 
                                                               Model_Config_dict['Meta_path_Column_drop_dict'])
    
    # 转换数据格式
    Result_Data_dict = Complex_Path_Format_Transfer(spark, Target_Node_Info_Dict, Complex_Path_Pyspark_Data_Dict)
                
    return Result_Data_dict

Result_Data_dict = Spark_Runner.run(Read_Target_Node_Feature)


# # Read Data to Memory and Change the Form to Fit Model

# In[10]:


def Get_Required_Complex_Path_Transformer_Data_by_Time(Result_Data_dict, aim_time_monthly_range_list):
    Required_Time_Data_dict = {}
    
    # 获取目标时间内的点对应的index
    tmp_aim_node_pd = Result_Data_dict['Sample']
    tmp_required_time_node_index = ((tmp_aim_node_pd['Source_Time'] >= aim_time_monthly_range_list[0].strftime("%Y-%m-%d")) &
                                    (tmp_aim_node_pd['Source_Time'] < aim_time_monthly_range_list[1].strftime("%Y-%m-%d")))
    
    # 获取对应点的标签
    Required_Time_Data_dict['Label'] = torch.FloatTensor(Result_Data_dict['Sample']['Label'][tmp_required_time_node_index].values)
    
    # 查看正负样本总数
    All_Label_np = Required_Time_Data_dict['Label'].data.numpy().astype(int)
    print('Positive Sample Count:', np.sum(All_Label_np == 1))
    print('Negative Sample Count:', np.sum(All_Label_np == 0))
    
    # 记录正负样本对应序号
    Required_Time_Data_dict['Pos_Label_loc'] = np.argwhere(All_Label_np == 1).T[0]
    Required_Time_Data_dict['Neg_Label_loc'] = np.argwhere(All_Label_np == 0).T[0]
    
    ###################################################################################################
    # 获取目标点本身对应特征(各种信息保持不变，只是特征进行采样)
    Required_Time_Data_dict["Raw_Feature"] = {}
    for key in Result_Data_dict['Raw_Feature'].keys():
        if key != 'data':
            Required_Time_Data_dict["Raw_Feature"][key] = copy.deepcopy(Result_Data_dict['Raw_Feature'][key])
        else:
            Required_Time_Data_dict["Raw_Feature"]['data'] = torch.FloatTensor(Result_Data_dict['Raw_Feature']['data'][tmp_required_time_node_index])

    # 获取各元路径的各列对应的特征
    Required_Time_Data_dict['Complex_Paths'] = {}
    for tmp_complex_path_name in Result_Data_dict['Complex_Paths']:
        Required_Time_Data_dict['Complex_Paths'][tmp_complex_path_name] = {}
        for key in Result_Data_dict['Complex_Paths'][tmp_complex_path_name].keys():
            if key != 'data':
                Required_Time_Data_dict["Complex_Paths"][tmp_complex_path_name][key] = copy.deepcopy(Result_Data_dict['Complex_Paths'][tmp_complex_path_name][key])
            else:
                Required_Time_Data_dict['Complex_Paths'][tmp_complex_path_name]['data'] = []
                for index, tmp_feat in enumerate(Result_Data_dict['Complex_Paths'][tmp_complex_path_name]['data']):
                    tmp_feat_torch = torch.FloatTensor(tmp_feat[tmp_required_time_node_index])

                    Required_Time_Data_dict['Complex_Paths'][tmp_complex_path_name]['data'].append(tmp_feat_torch)
    
    # 可以加入删除已取出数据的代码
    
    return Required_Time_Data_dict


# In[11]:


Train_Data_Dict = Get_Required_Complex_Path_Transformer_Data_by_Time(Result_Data_dict, Train_time_range_list)
Valid_Data_Dict = Get_Required_Complex_Path_Transformer_Data_by_Time(Result_Data_dict, Validation_time_range_list)
Test_Data_Dict = Get_Required_Complex_Path_Transformer_Data_by_Time(Result_Data_dict, Test_time_range_list)


# # 根据需求随机采样

# In[12]:


def sample_random_index_with_portion(Source_Data_Dict, sample_size, positive_percent):
    if positive_percent > 0:
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
    else:
        tmp_sampled_label_index = np.random.choice(np.arange(0, Source_Data_Dict['Label'].size(0)), size = sample_size, replace = False)
        
    return tmp_sampled_label_index


# In[13]:


def sample_for_Complex_Path_Transformer(Source_Data_Dict, tmp_sampled_index_np):
    Sampled_Data_Dict = {}
    
    # 提取对应标签
    Sampled_Data_Dict['Label'] = Source_Data_Dict['Label'][tmp_sampled_index_np]

    # 提取目标节点对应特征
    Sampled_Data_Dict["Raw_Feature"] = {}
    for key in Source_Data_Dict['Raw_Feature'].keys():
        if key != 'data':
            Sampled_Data_Dict["Raw_Feature"][key] = copy.deepcopy(Source_Data_Dict['Raw_Feature'][key])
        else:
            Sampled_Data_Dict["Raw_Feature"]['data'] = Source_Data_Dict['Raw_Feature']['data'][tmp_sampled_index_np]
    
    # 提取元路径对应特征
    Sampled_Data_Dict['Complex_Paths'] = {}
    for complex_path_name in Source_Data_Dict['Complex_Paths']:
        Sampled_Data_Dict['Complex_Paths'][complex_path_name] = {}
        
        for key in Source_Data_Dict['Complex_Paths'][complex_path_name].keys():
            if key != 'data':
                Sampled_Data_Dict["Complex_Paths"][complex_path_name][key] = copy.deepcopy(Source_Data_Dict['Complex_Paths'][complex_path_name][key])
            else:
                Sampled_Data_Dict["Complex_Paths"][complex_path_name]['data'] = []
                for index, tmp_feat in enumerate(Source_Data_Dict['Complex_Paths'][complex_path_name]['data']):
                    Sampled_Data_Dict['Complex_Paths'][complex_path_name]['data'].append(tmp_feat[tmp_sampled_index_np])
            
    # 放入cuda
    if args_cuda:
        Sampled_Data_Dict['Label'] = Sampled_Data_Dict['Label'].cuda()
        Sampled_Data_Dict['Raw_Feature']['data'] = Sampled_Data_Dict['Raw_Feature']['data'].cuda()
        for complex_path_name in Sampled_Data_Dict['Complex_Paths']:
            for index, tmp_feat in enumerate(Sampled_Data_Dict['Complex_Paths'][complex_path_name]['data']):
                Sampled_Data_Dict['Complex_Paths'][complex_path_name]['data'][index] = tmp_feat.cuda()

    return Sampled_Data_Dict


# # 模型

# In[14]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from torch.nn import TransformerEncoder, TransformerEncoderLayer

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, dropout, layer_num = 3):
        super().__init__()

        self.input = nn.Linear(input_size, hidden_size) 
        
        self.mlp = nn.ModuleList([nn.Linear(hidden_size, hidden_size) for _ in range(layer_num)])
        
        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        self.output = nn.Linear(hidden_size, output_size)
        
    def forward(self, hidden_states):
        hidden_states = self.input(hidden_states)
        
#         hidden_states_shortcut = hidden_states
        for i,layer_module in enumerate(self.mlp):
            hidden_states = layer_module(hidden_states)
            hidden_states = self.LayerNorm(hidden_states)
            hidden_states = self.activation(hidden_states)
            hidden_states = self.dropout(hidden_states)

#             hidden_states = hidden_states_shortcut + hidden_states
        
        hidden_states = self.output(hidden_states)
        
        return hidden_states

class PositionEmbeddings(nn.Module):
    def __init__(self, nfeat, seq_length, dropout):
        super().__init__()
        
        self.seq_length = seq_length
        
        self.position_embeddings = nn.Embedding(seq_length, nfeat)
        
        self.LayerNorm = nn.LayerNorm(nfeat)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, features_embeddings):
        
        position_embeddings = self.position_embeddings.weight.unsqueeze(1).expand(self.seq_length, 
                                                                                  features_embeddings.size(1), 
                                                                                  features_embeddings.size(2))
        
        embeddings = features_embeddings + position_embeddings
        
        embeddings = self.LayerNorm(embeddings)
        
        embeddings = self.dropout(embeddings)
        
        return embeddings

class Complex_Path_Transformer(nn.Module):
    def __init__(self, data_info, hid_len, dropout=0.5):
        super().__init__()
        
        # 基础函数
        self.LayerNorm = nn.LayerNorm(hid_len)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Feature_Uniform
        self.feature_uniform_dict = {}
        for tmp_node_type in data_info["Type_to_Feature_Len"]:
            self.feature_uniform_dict[tmp_node_type] = MLP(data_info["Type_to_Feature_Len"][tmp_node_type], hid_len,
                                                          hid_len, dropout, 2)
            
            self.add_module('Feature_Uniform_{}'.format(tmp_node_type), self.feature_uniform_dict[tmp_node_type])
        
        # Complex-Path Level
        encoder_layers = TransformerEncoderLayer(hid_len, 1, hid_len, dropout)
        self.complex_path_embed = TransformerEncoder(encoder_layers, 1)
        
        self.complex_path_pos_dict = {}
        self.complex_path_fc_dict = {}
        for complex_path_name in data_info["Complex_Paths"]:
            path_len = len(data_info["Complex_Paths"][complex_path_name]['data']) + 1
            
            # Pos
            self.complex_path_pos_dict[complex_path_name] = PositionEmbeddings(hid_len, path_len, dropout)
            self.add_module(f'Complex_Path_Level_Pos_{complex_path_name}', self.complex_path_pos_dict[complex_path_name])
            
            # project
            self.complex_path_fc_dict[complex_path_name] = nn.Linear(path_len*hid_len, hid_len)
            self.add_module(f'Complex_Path_Level_FC_{complex_path_name}', self.complex_path_fc_dict[complex_path_name])
            
        # Node Level
        encoder_layers = TransformerEncoderLayer(hid_len, 1, hid_len, dropout)
        self.node_level_embed = TransformerEncoder(encoder_layers, 1)
        
        # 最后的输出函数
        self.predict_layer = MLP(hid_len*(1 + len(data_info["Complex_Paths"].keys())), hid_len, 1, dropout, 2)
        self.predict_activation = nn.Sigmoid()
        
    def forward(self, data_dict):
        # 先转化目标节点本身特征
        target_node_type = data_dict["Raw_Feature"]['node_type']
        target_node_h = self.feature_uniform_dict[target_node_type](data_dict['Raw_Feature']['data'])
        
        target_node_h = self.LayerNorm(target_node_h)
        target_node_h = self.activation(target_node_h)
        target_node_h = self.dropout(target_node_h)
        
        ###################################################################################################################
        # Embed for each complex_path
        complex_path_h_list = [target_node_h]
        
        # complex-path level
        for complex_path_name in data_dict['Complex_Paths']:
            complex_path_item_h_list = [target_node_h]
            
            for type_index, type_data in enumerate(data_dict['Complex_Paths'][complex_path_name]['data']):
                type_name = data_dict["Complex_Paths"][complex_path_name]['node_type'][type_index]
                
                type_h = self.feature_uniform_dict[type_name](type_data)
                type_h = self.LayerNorm(type_h)
                type_h = self.activation(type_h)
                type_h = self.dropout(type_h)
        
                complex_path_item_h_list.append(type_h)
                
            # 合并转换后的原始特征
            complex_path_h_stack = torch.stack(complex_path_item_h_list, 0)

            # 通过complex_path_level_transformer
            complex_path_h_stack = self.complex_path_pos_dict[complex_path_name](complex_path_h_stack)
            complex_path_h_stack = self.complex_path_embed(complex_path_h_stack)
            
            complex_path_h_stack = self.LayerNorm(complex_path_h_stack)
            complex_path_h_stack = self.activation(complex_path_h_stack)
            complex_path_h_stack = self.dropout(complex_path_h_stack)
            
            # complex_path_level_fc
            batch_num = complex_path_h_stack.size(1)
            complex_path_h_stack = complex_path_h_stack.permute(1, 0, 2).reshape(batch_num, -1)
            complex_path_h_stack = self.complex_path_fc_dict[complex_path_name](complex_path_h_stack)
            
            complex_path_h_stack = self.LayerNorm(complex_path_h_stack)
            complex_path_h_stack = self.activation(complex_path_h_stack)
            complex_path_h_stack = self.dropout(complex_path_h_stack)
            
            # 存储结果
            complex_path_h_list.append(complex_path_h_stack)
            
        ###################################################################################################################
        # Node level
        target_h_stack = torch.stack(complex_path_h_list, 0)
        
        target_h_stack = self.node_level_embed(target_h_stack)
        
        target_h_stack = self.LayerNorm(target_h_stack)
        target_h_stack = self.activation(target_h_stack)
        target_h_stack = self.dropout(target_h_stack)
        
        batch_num = target_h_stack.size(1)
        target_h_stack = target_h_stack.permute(1, 0, 2).reshape(batch_num, -1)
        h_output = self.predict_layer(target_h_stack)
        
        h_output = h_output.squeeze()
        h_output = self.predict_activation(h_output)

        return h_output


# In[15]:


# from kg_model.Complex_Path_Transformer import Complex_Path_Transformer
from torch.optim.lr_scheduler import LambdaLR, StepLR

# 建立模型
model = Complex_Path_Transformer(Result_Data_dict, hid_len = Model_Config_dict['hid_len'], dropout = Model_Config_dict['dropout'])
if args_cuda:
    model.cuda()
# print(model)


# In[16]:


# 优化器
optimizer = optim.Adam(model.parameters(), lr = Model_Config_dict['learning_rate'], weight_decay = Model_Config_dict['weight_decay'])

# 动态调节学习率
scheduler = StepLR(optimizer, step_size = 30, gamma = 0.33)

# def get_linear_schedule(optimizer, base_lr, num_training_steps, last_epoch=-1):
#     def lr_lambda(current_step):
#         return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps)))
#     return LambdaLR(optimizer, lr_lambda, last_epoch)

# scheduler = get_linear_schedule(optimizer, Model_Config_dict['learning_rate'], Model_Config_dict['train_epoch'])


# In[17]:


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy(inputs, targets, reduction='none')
        targets = targets.type(torch.float32)
        at = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        pt = torch.exp(-BCE_loss)
        F_loss = at * (1-pt)**self.gamma * BCE_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss

# 损失函数
# BCE_loss = torch.nn.BCELoss()

BCE_loss = FocalLoss()


# # 评价函数

# In[18]:


def top_k_accuracy_score(y_true, y_score, k):
    sorted_pred = np.argsort(y_score)
    sorted_pred = sorted_pred[::-1]
    sorted_pred = sorted_pred[:k]

    hits = y_true[sorted_pred]
    
    return np.sum(hits)/k


# In[19]:


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

# In[20]:


from Utils.utils import mkdir

localtime = time.strftime("%m-%d-%H:%M", time.localtime())

temp_train_list_name = ('Train_' + Train_time_range_list[0].strftime('%Y_%m_%d') + '-' 
                        + Train_time_range_list[1].strftime('%Y_%m_%d'))

# 模型参数的输出文件夹
tmp_model_parameter_output_dir = Data_Output_path + '/Model_Parameter/Meta_Path_Transformer/' + localtime + '_' + temp_train_list_name + '/'
mkdir(Data_Output_path + '/Model_Parameter')
mkdir(Data_Output_path + '/Model_Parameter/Meta_Path_Transformer/')
mkdir(tmp_model_parameter_output_dir)

# # 保存基本信息
# f = open(tmp_model_parameter_output_dir + 'Node_Type_to_Feature_len_dict.json', 'w')
# f.write(json.dumps(Node_Type_to_Feature_len_dict, sort_keys = False, indent = 4))
# f.close()

# np.save(tmp_model_parameter_output_dir + 'All_Meta_Path_Name_list.npy', np.array(All_Meta_Path_Name_list))

# f = open(tmp_model_parameter_output_dir + 'Meta_Path_Column_Type_dict.json', 'w')
# f.write(json.dumps(Meta_Path_Column_Type_dict, sort_keys = False, indent = 4))
# f.close()

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
    scheduler.step()
    
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


# In[ ]:


# 优化loss函数


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

