#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
import io
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

# Self packages path
Src_Dir_path = "../../.."
sys.path.append(Src_Dir_path)

# Self packages
from Utils.Pyspark_utils import ResilientSparkRunner
from Utils.Pyspark_utils import DEFAULT_STORAGE_LEVEL
from Utils.utils import read_json_config_file
from Utils.utils import mkdir
from Utils.utils import Log_save


# In[3]:


#显示所有列
pd.set_option('display.max_columns', None)

#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)

# 固定随机值
def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
setup_seed(42)


# # 配置信息

# In[6]:


from Utils.utils import read_json_config_file

Time_range = ["2023-06-01", "2023-10-01"]

Label_Data_Config_file = './Config/Target_Node_Dataset_Config/KP_Task_2023_12_18.json'
Subgraph_Config_file = './Config/K_Hop_Config/k_hop_config_for_paths2pairs_2023_12_30.json'
Node_Config_file = './Config/Node_Config/Node_Config_Paths2Pairs_12_29.json'

Label_Data_Config_dict = read_json_config_file(Label_Data_Config_file)
Subgraph_Config_dict = read_json_config_file(Subgraph_Config_file)
Node_Config_dict = read_json_config_file(Node_Config_file)


# In[7]:


Time_range = ["2023-06-01", "2023-10-01"]

task_name = '01_20_test'

Data_In_HDFS_Path = '/user/mart_coo/mart_coo_innov/K_Hop_Sampling/' + task_name


# # 读取数据

# In[ ]:


from Utils.Target_Node_Dataloader import Read_Target_Node_with_Label
from Utils.K_hop_Path_Sampling import Pyspark_K_Hop_Path_Sampling, k_hop_path_pairing_to_sample
from Utils.Complex_Path_Basic_PySpark import complex_path_sampling

def get_train_sample_data(spark, Target_Node_Info_Dict, subgraph_hop_k = 4, max_main_path = 5):
    # 训练集对应信息存储位置
    Train_Data_Store_dir = HDFS_Store_dir + '/Train_K_Hop_Path'
    
    # 只取出训练集对应的信息
    tmp_time_range_start = All_aim_time_range_dict["Train"][0]
    tmp_time_range_end = All_aim_time_range_dict["Train"][1]
    tmp_time_range_limit = f"Source_Time >= '{tmp_time_range_start}' AND Source_Time < '{tmp_time_range_end}'"
    Pairs_for_train_df = Target_Node_Info_Dict['Data'].filter(tmp_time_range_limit)
    Pairs_for_train_df.persist(DEFAULT_STORAGE_LEVEL)
    
    # 获取训练集涉及的全部特征时间
    train_df_feature_times = [row['Feature_Time'] for row in Pairs_for_train_df.select("Feature_Time").distinct().collect()]
    train_df_feature_times.sort()
    
    # 将训练集中的公司节点为采样目标点
    K_Hop_Target_Info = {}
    K_Hop_Target_Info['Target_Node_df'] = Pairs_for_train_df.select(['Company_Node_UID', "Feature_Time"]).distinct()
    K_Hop_Target_Info['Target_Node_Type'] = "Company_Node"
    K_Hop_Target_Info['End_Node_Type'] = 'Mobile_Node'
    K_Hop_Target_Info['Target_Node_UID_name'] = "Company_Node_UID"
    K_Hop_Target_Info['Feature_Times'] = train_df_feature_times

    # 取出训练集中包含的公司-手机号组合对作为目标组合对
    Pair_Nodes_Info = {}
    Pair_Nodes_Info['Pair_Data'] = Pairs_for_train_df.select(["Company_Node_UID", "Mobile_Node_UID", "Feature_Time"]).distinct()
    Pair_Nodes_Info['Start_Node_Type'] = 'Company_Node'
    Pair_Nodes_Info['End_Node_Type'] = 'Mobile_Node'
    Pair_Nodes_Info['Start_Node_name'] = "Company_Node_UID"
    Pair_Nodes_Info['End_Node_name'] = "Mobile_Node_UID"
    Pair_Nodes_Info['Feature_Times'] = train_df_feature_times
    
    # 针对目标节点做k-hop采样
    Pyspark_K_Hop_Path_Sampling(spark, K_Hop_Target_Info, Subgraph_Config_dict, subgraph_hop_k, Train_Data_Store_dir, 
                                Pair_Nodes_Info)
    
    # 针对采样结果检测覆盖情况，给出主要路径、辅助路径，以及对应的样本
    train_data = k_hop_path_pairing_to_sample(spark, Train_Data_Store_dir, subgraph_hop_k, Pair_Nodes_Info, max_main_path)
    
    return train_data


# In[ ]:





# In[8]:


from kg_lib.Target_Node_Dataloader import get_aim_UID_with_label_rdd
from kg_lib.Subgraph_Dataloader import get_sub_graph
from kg_lib.Node_Feature_Dataloader import get_node_related_features
from kg_lib.utils import mkdir

def Get_Subgraph_Required_Data(data_source_description_str, tmp_aim_time_monthly_range, tmp_subgraph_hop):
    tmp_start_time = datetime.now()
    print('开始对时间区间' + str(tmp_aim_time_monthly_range) + '内的数据的处理')
    
    # 数据存储位置
    tmp_all_output_data_base_dir = '../../Data/'
    mkdir(tmp_all_output_data_base_dir)
    
    # 先是数据来源
    tmp_all_output_data_source_dir = (tmp_all_output_data_base_dir + data_source_description_str + '/')
    mkdir(tmp_all_output_data_source_dir)
    
    # 再是数据对应时间区间
    time_range_str = ('Time_Range:' + str(tmp_aim_time_monthly_range))
    tmp_all_output_data_time_range_dir = (tmp_all_output_data_source_dir + time_range_str + '/')
    mkdir(tmp_all_output_data_time_range_dir)
    
    #######################################################################################################################
    # 从线上表中取标签数据，并保存
    tmp_aim_entity_info_dict = get_aim_UID_with_label_rdd(Spark_Session, Label_Data_Config_dict, tmp_aim_time_monthly_range[0], 
                                                          tmp_aim_time_monthly_range[1], tmp_all_output_data_time_range_dir)
    
    #######################################################################################################################
    # 设定子图关系表存储文件夹
    tmp_all_output_data_time_range_subgraph_dir = tmp_all_output_data_time_range_dir + 'Subgraph/'
    mkdir(tmp_all_output_data_time_range_subgraph_dir)
    
    # 设定节点表存储文件夹
    tmp_all_output_data_time_range_node_dir = tmp_all_output_data_time_range_dir + 'Node/'
    mkdir(tmp_all_output_data_time_range_node_dir)
    
    # 从线上表中取元路径数据及对应的节点特征，并保存
    get_sub_graph(Spark_Session, tmp_aim_entity_info_dict, Subgraph_Config_dict, tmp_subgraph_hop, 
                  tmp_all_output_data_time_range_subgraph_dir, tmp_all_output_data_time_range_node_dir)
    
    #######################################################################################################################
    # 设定特征表存储文件夹
    tmp_all_output_data_time_range_feature_dir = tmp_all_output_data_time_range_dir + 'Feature/'
    mkdir(tmp_all_output_data_time_range_feature_dir)
    
    # 从线上表中取特征数据，并保存
    get_node_related_features(Spark_Session, tmp_aim_entity_info_dict, Feature_Dataset_Config_dict, 
                              tmp_all_output_data_time_range_node_dir, tmp_all_output_data_time_range_feature_dir)
    
    #######################################################################################################################
    tmp_end_time = datetime.now()

    print('完成时间区间' + time_range_str + '内的数据的处理，花费时间：', tmp_end_time - tmp_start_time)
    print('**************************************************************************')
    
    return


# In[9]:


curr_time = datetime.now()
print(curr_time) 
print(type(curr_time)) 

# 要计算的时间区间
tmp_aim_time_monthly_range_list = (KG_train_time_monthly_range_list + KG_validation_time_monthly_range_list + 
                                   KG_test_time_monthly_range_list)

for tmp_aim_time_monthly_range in tmp_aim_time_monthly_range_list:
    Get_Subgraph_Required_Data(data_source_description_str, tmp_aim_time_monthly_range, tmp_subgraph_hop = 2)
    
curr_time2 = datetime.now()
print(curr_time2)
print(curr_time2-curr_time)


# In[ ]:




