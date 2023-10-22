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

from tqdm import tqdm

from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta


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
    Spark_Session = SparkSession.builder                     .appName("kg_spark_for_model")                     .enableHiveSupport()                     .config("spark.default.parallelism", "300")                     .config("spark.sql.shuffle.partitions", "300")                     .config("spark.sql.broadcastTimeout","3600")                    .config("spark.driver.memory", "200g")                     .config("spark.executor.memory", "40g")                    .config("spark.yarn.appMasterEnv.yarn.nodemanager.container-executor.class", "DockerLinuxContainer")                    .config("spark.executorEnv.yarn.nodemanager.container-executor.class", "DockerLinuxContainer")                    .config("spark.yarn.appMasterEnv.yarn.nodemanager.docker-container-executor.image-name", 
                            "bdp-docker.jd.com:5000/wise_mart_bag:latest")\
                    .config("spark.executorEnv.yarn.nodemanager.docker-container-executor.image-name", 
                            "bdp-docker.jd.com:5000/wise_mart_bag:latest")\
                    .config("spark.sql.crossJoin.enabled", "true")\
                    .config("spark.driver.maxResultSize", "40g")\
                    .config("spark.driver.memory", "20g") \
                    .config("spark.sql.autoBroadcastJoinThreshold","-1")\
                    .getOrCreate()
    
    # 获取 Spark 应用程序的 ID
    tmp_app_id = Spark_Session.sparkContext.applicationId

    # 打印应用程序 ID
    print("Application ID: http://10k2.jd.com/proxy/" + tmp_app_id)
    
    return Spark_Session


# In[5]:


# 固定随机值
def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    
setup_seed(42)


# In[6]:


from kg_lib.utils import read_json_config_file
from kg_lib.utils import mkdir

# 数据来源描述
# Output_Columns_Type = "Head_And_Tail"
# Output_Columns_Type = "ALL_Nodes"
# Output_Columns_Type = "ALL_Edges"
Output_Columns_Type = "ALL_Nodes_And_Edges"

# 输出的特征时间数目
Feature_Month_Range = 1

# 目标时间
# Aim_Meta_Path_Relation_Time_List = ['2023-04-01', '2023-03-01', '2023-02-01', '2023-01-01', '2022-12-01', '2022-11-01', '2022-10-01',
#                                     '2022-09-01', '2022-08-01', '2022-07-01', '2022-06-01', '2022-05-01', '2022-04-01']
Aim_Meta_Path_Relation_Time_List = ['2023-06-01']

# 日志信息保存文件名
mkdir('../../Log/')
log_file_name = '../../Log/Meta_Path_Feature_Generate-' + datetime.now().strftime("%Y-%m-%d-%H:%M") + '.txt'

Feature_Dataset_Config_file = './kg_config/Node_Feature_Dataset_Config/Feature_Dataset_Config_2023_06_29_Complexpath_Feature.json'
Metapath_Feature_Config_file = './kg_config/Complexpath_FeatureAgg_Config/Metapath_Feature_Config_2023_06_05_ComplexPath_All.json'

Feature_Dataset_Config_dict = read_json_config_file(Feature_Dataset_Config_file)
Metapath_Feature_Config_dict = read_json_config_file(Metapath_Feature_Config_file)


# In[7]:


# 记录pyspark重启次数
spark_restart_count = 0

# 记录当前处理到那个时间点
Time_Processed_i = 0

# 记录当前处理到第几条元路径
Meta_Path_Processed_i = 5

# 记录当前处理到哪个时间点的特征
Aim_Node_Feature_Month_Delta = 0

# 记录当前元路径处理到哪一列的节点了
Column_Processed_Count_list = [2]

# 是否还要再输出权重列
Output_Weight_Feature_Table_list = [False]

# 记录当前处理到特征表中的第几位
Feature_Table_Processed_Count_list = [0]

# 记录当前处理到特征表中的第几列特征
Feature_Table_Processed_Column_Name_list = [None]

# 记录当前处理到输出的表的句号
Feature_Table_Upload_Count_list = [0]

# 表名注释，防止有重复表名
Table_Name_Comment = '07_07'


# In[8]:


from kg_lib.Metapath_Feature_Generate import Meta_Path_Feature_Generate_and_Upload
from kg_lib.utils import Log_save
from py4j.protocol import Py4JError, Py4JJavaError, Py4JNetworkError

# 生成输出文件，并记录信息
log_file_f = open(log_file_name, 'w')
sys.stdout = Log_save(sys.stdout, log_file_f)

print('日志输出文件名:', log_file_name)
print('使用的特征配置:', Feature_Dataset_Config_file)
print('使用的元路径配置:', Metapath_Feature_Config_file)

start_time = datetime.now()
print(start_time)

while True:
    try:
        # 启动spark
        Spark_Session = Start_Spark()

        # 依次处理各时间的数据
        for tmp_time_i in range(Time_Processed_i, len(Aim_Meta_Path_Relation_Time_List)):
            tmp_start_time = datetime.now()

            Time_Processed_i = tmp_time_i

            tmp_Aim_Relation_Table_dt = Aim_Meta_Path_Relation_Time_List[tmp_time_i]

            # 依次处理各个时间点的特征
            for tmp_feature_month_delta in range(Aim_Node_Feature_Month_Delta, Feature_Month_Range):
                Aim_Node_Feature_Month_Delta = tmp_feature_month_delta

                tmp_Aim_Feature_Table_dt = (datetime.strptime(tmp_Aim_Relation_Table_dt, "%Y-%m-%d") - 
                                            relativedelta(months = tmp_feature_month_delta)).strftime("%Y-%m-%d")

                # 依次处理各个元路径
                for tmp_meta_path_i in range(Meta_Path_Processed_i, len(Metapath_Feature_Config_dict.keys())):
                    Meta_Path_Processed_i = tmp_meta_path_i

                    tmp_aim_meta_path_name = list(Metapath_Feature_Config_dict.keys())[tmp_meta_path_i]

                    print('##########################################################################################')

                    print('从第', Column_Processed_Count_list[0] + 1, '号列开始处理在时间', 
                          tmp_Aim_Relation_Table_dt, '时元路径关系', tmp_aim_meta_path_name, '在时间', tmp_Aim_Feature_Table_dt, '时的特征')

                    # 读取该元路径对应的特征并上传
                    Meta_Path_Feature_Generate_and_Upload(Spark_Session, Metapath_Feature_Config_dict, tmp_aim_meta_path_name, 
                                                          Feature_Dataset_Config_dict, tmp_Aim_Relation_Table_dt, 
                                                          tmp_Aim_Feature_Table_dt, Column_Processed_Count_list, Output_Weight_Feature_Table_list,
                                                          Feature_Table_Processed_Count_list, Feature_Table_Processed_Column_Name_list,
                                                          Feature_Table_Upload_Count_list, Table_Name_Comment, 
                                                          Output_Columns_Type = Output_Columns_Type)

                    for (tmp_id, tmp_rdd) in Spark_Session.sparkContext._jsc.getPersistentRDDs().items():
                        tmp_rdd.unpersist()

                    # 重置开始处理列的序号
                    Column_Processed_Count_list[0] = 0

                # 重置 Meta_Path_Processed_i
                Meta_Path_Processed_i = 0

            # 重置 Aim_Node_Feature_Month_Delta
            Aim_Node_Feature_Month_Delta = 0

            tmp_end_time = datetime.now()
            print('完成对时间段', tmp_Aim_Relation_Table_dt, '涉及到的数据的处理，花费时间:', tmp_end_time - tmp_start_time)

        # 重置 Time_Processed_i(实际不需要)
        Time_Processed_i = 0

        break
        
    except (Py4JError, Py4JJavaError, Py4JNetworkError):
        print('*******************************************************************************')
        
        print('中断在', Time_Processed_i, Meta_Path_Processed_i, Aim_Node_Feature_Month_Delta, Column_Processed_Count_list, Output_Weight_Feature_Table_list,
              Feature_Table_Processed_Count_list, Feature_Table_Processed_Column_Name_list, Feature_Table_Upload_Count_list)
        
        interrupt_time = datetime.now()
        print('pyspark异常中断，时间:', interrupt_time)
        
#         print('清空persist变量')
#         for (tmp_id, tmp_rdd) in Spark_Session.sparkContext._jsc.getPersistentRDDs().items():
#             tmp_rdd.unpersist()
        
        print('终止Spark_Session')
        Spark_Session.stop()
        Spark_Session._instantiatedContext = None
        
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
            print('重启Spark并重新开始运算', datetime.now())
            
        print('*******************************************************************************')
        
end_time = datetime.now()
print(end_time)
print(end_time - start_time)


# In[ ]:




