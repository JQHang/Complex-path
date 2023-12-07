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


import re
import xlrd
import warnings
import matplotlib.ticker as mtick
from matplotlib.font_manager import FontProperties
get_ipython().run_line_magic('matplotlib', 'inline')

# spark相关配置
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
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


# In[3]:


#显示所有列
pd.set_option('display.max_columns', None)

#设置value的显示长度为100，默认为50
pd.set_option('max_colwidth',100)


# In[4]:


# 创建spark_session
os.environ['PYSPARK_PYTHON']="/usr/local/anaconda3/bin/python"

def Start_Spark():
    Spark_Session = SparkSession.builder                     .appName("kg_spark_for_model")                     .enableHiveSupport()                     .config("spark.sql.shuffle.partitions", "500")                     .config("spark.sql.broadcastTimeout","3600")                    .config("spark.driver.memory", "200g")                     .config("spark.executor.memory", "40g")                    .config("spark.yarn.appMasterEnv.yarn.nodemanager.container-executor.class", "DockerLinuxContainer")                    .config("spark.executorEnv.yarn.nodemanager.container-executor.class", "DockerLinuxContainer")                    .config("spark.yarn.appMasterEnv.yarn.nodemanager.docker-container-executor.image-name", 
                            "bdp-docker.jd.com:5000/wise_mart_bag:latest")\
                    .config("spark.executorEnv.yarn.nodemanager.docker-container-executor.image-name", 
                            "bdp-docker.jd.com:5000/wise_mart_bag:latest")\
                    .config("spark.sql.crossJoin.enabled", "true")\
                    .config("spark.driver.maxResultSize", "40g")\
                    .config("spark.driver.memory","20g")\
                    .config("spark.sql.autoBroadcastJoinThreshold","-1")\
                    .getOrCreate()
    
    return Spark_Session


# In[5]:


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


# # 配置信息

# In[8]:


from datetime import datetime
from kg_lib.utils import read_json_config_file
from kg_lib.utils import mkdir
from kg_lib.utils import Log_save

# 数据来源描述
# Output_Columns_Type = "Head_And_Tail"
Output_Columns_Type = "ALL_Nodes"
# Output_Columns_Type = "ALL_Nodes_And_Edges"

tmp_feature_month_range = 1

data_source_description_str = '07_13-签约标签11至12月数据-过去' + str(tmp_feature_month_range) + '个月的特征-' + Output_Columns_Type + '格式'
print(data_source_description_str)

# 日志信息保存文件名
mkdir('../../Log/')
log_file_name = '../../Log/' + data_source_description_str + '-' + datetime.now().strftime("%Y-%m-%d-%H:%M") + '.txt'

Feature_Dataset_Config_file = './kg_config/Node_Feature_Dataset_Config/Feature_Dataset_Config_2023_06_10_Raw_Feature.json'
Metapath_Config_file = './kg_config/Complexpath_FeatureAgg_Config/Metapath_Feature_Config_2023_06_05_ComplexPath_All.json'
Label_Data_Config_file = './kg_config/Target_Node_Dataset_Config/Sign_Label_Train_Target_Data_Config_2023_05_20.json'

Feature_Dataset_Config_dict = read_json_config_file(Feature_Dataset_Config_file)
Metapath_Config_dict = read_json_config_file(Metapath_Config_file)
Label_Data_Config_dict = read_json_config_file(Label_Data_Config_file)


# In[9]:


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


# # 读取数据

# In[10]:


from kg_lib.Target_Node_Dataloader import get_aim_UID_with_label_rdd
from kg_lib.Metapath_Dataloader import get_meta_path
from kg_lib.Node_Feature_Dataloader import get_node_related_features
from kg_lib.utils import mkdir

def get_meta_path_related_pandas_data_and_store_by_time(Spark_Session, data_source_description_str, tmp_aim_time_monthly_range, 
                                                        tmp_output_columns_type):
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
    # 设定元路径关系表存储文件夹
    tmp_all_output_data_time_range_meta_path_dir = tmp_all_output_data_time_range_dir + 'Meta_Path/'
    mkdir(tmp_all_output_data_time_range_meta_path_dir)
    
    # 设定节点表存储文件夹
    tmp_all_output_data_time_range_node_dir = tmp_all_output_data_time_range_dir + 'Node/'
    mkdir(tmp_all_output_data_time_range_node_dir)
    
    # 从线上表中取元路径数据及对应的节点特征，并保存
    get_meta_path(Spark_Session, tmp_aim_entity_info_dict, Metapath_Config_dict, tmp_all_output_data_time_range_meta_path_dir,
                  tmp_all_output_data_time_range_node_dir, tmp_output_columns_type)
    
    #######################################################################################################################
    # 设定特征表存储文件夹
    tmp_all_output_data_time_range_feature_dir = tmp_all_output_data_time_range_dir + 'Feature/'
    mkdir(tmp_all_output_data_time_range_feature_dir)
    
    # 从线上表中取特征数据，并保存
    get_node_related_features(Spark_Session, tmp_aim_entity_info_dict, Feature_Dataset_Config_dict, 
                              tmp_all_output_data_time_range_node_dir, tmp_all_output_data_time_range_feature_dir,
                              tmp_feature_month_range)
    #######################################################################################################################
    tmp_end_time = datetime.now()

    print('完成时间区间' + time_range_str + '内的数据的处理，花费时间：', tmp_end_time - tmp_start_time)
    print('**************************************************************************')
    
    return


# In[11]:


from datetime import datetime
from py4j.protocol import Py4JJavaError

# 生成输出文件，并记录信息
log_file_f = open(log_file_name, 'w')
sys.stdout = Log_save(sys.stdout, log_file_f)

print('日志输出文件名:', log_file_name)
print('数据输出文件夹:', data_source_description_str)
print('使用的特征配置:', Feature_Dataset_Config_file)
print('使用的元路径配置:', Metapath_Config_file)

# 要计算的时间区间
tmp_aim_time_monthly_range_list = (KG_train_time_monthly_range_list + KG_validation_time_monthly_range_list + 
                                   KG_test_time_monthly_range_list)
print('要计算的时间区间:', tmp_aim_time_monthly_range_list)

start_time = datetime.now()
print(start_time)

spark_restart_count = 0

while True:
    try:
        # 启动spark
        Spark_Session = Start_Spark()

        for tmp_aim_time_monthly_range in tmp_aim_time_monthly_range_list:
            get_meta_path_related_pandas_data_and_store_by_time(Spark_Session, data_source_description_str, 
                                                                tmp_aim_time_monthly_range, Output_Columns_Type)
        
        break
        
    except Py4JJavaError:
        print('-------------------------------------------------------')
        
        Spark_Session.stop()
        Spark_Session._instantiatedContext = None
        
        interrupt_time = datetime.now()
        print('pyspark异常中断，时间:', interrupt_time)
        
        # 如果不是0-8点间的中断且重启次数小于10，则进行重启
        if interrupt_time.hour < 9:
            print('0-9点无法运算，故先暂停')
            
            time_sleep = (8 - interrupt_time.hour)*3600 + (60 - interrupt_time.minute) * 60
            
            print('休眠时间', time_sleep)
            
            time.sleep(time_sleep)
            
            print('已到早上8点，重启运算')
            
        elif spark_restart_count > 10:
            print('重启超过10次，故终止')
            break
            
        else:
            spark_restart_count = spark_restart_count + 1
            print('重启Spark并重新开始运算')
            
        print('-------------------------------------------------------')
        
end_time = datetime.now()
print(end_time)
print(end_time - start_time)


# In[ ]:




