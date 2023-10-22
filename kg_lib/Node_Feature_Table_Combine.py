from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import VectorAssembler
from pyspark.ml import Pipeline
from pyspark.sql import Row
from pyspark.sql.functions import udf
from pyspark.sql.types import DoubleType
from pyspark.sql.types import *
from pyspark.sql.functions import broadcast

from kg_lib.Pyspark_utils import sample_random_n_samples_for_samll_rdd
from kg_lib.Pyspark_utils import sample_top_n_groupby_samples_for_samll_rdd
from kg_lib.Pyspark_utils import sample_random_n_groupby_samples_for_samll_rdd
from kg_lib.Pyspark_utils import sample_rdd_from_aim_row, sample_rdd_from_aim_range
from kg_lib.Pyspark_utils import Groupby_Feature_Table
from kg_lib.Pyspark_utils import Pyspark_Create_Table, Upload_RDD_Data_to_Database

import pandas as pd
import numpy as np
import os
import json
from tqdm import tqdm
from datetime import datetime


# """
# 作用：
#     合并同一节点的各个特征表
#
# 返回值：

# """
def Node_Feature_Table_Combine(Spark_Session, Aim_Node_Type, Aim_Node_Feature_Table_Info, Feature_Table_Processed_Count_list, 
                     Feature_Table_Upload_Count_list, Aim_Feature_Table_dt, Feature_Table_Comment):
    # 记录该节点涉及的全部特征
    tmp_all_useful_feature_cols_list = []

    # 记录该节点涉及的全部特征的注释
    tmp_all_useful_feature_cols_comments_list = []
    
    if "Max_Column_Number" in Aim_Node_Feature_Table_Info:
        Max_Column_Number = Aim_Node_Feature_Table_Info["Max_Column_Number"]
    else:
        Max_Column_Number = -1
    
    tmp_combine_start_time = datetime.now()
    
    # 按顺序读取各个表
    for tmp_feature_table_i in range(Feature_Table_Processed_Count_list[0], len(Aim_Node_Feature_Table_Info["Feature_Data_List"])):
        Feature_Table_Info = Aim_Node_Feature_Table_Info["Feature_Data_List"][tmp_feature_table_i]
        
        tmp_feature_table_name = Feature_Table_Info['Table_Name']
        tmp_aim_column_name = Feature_Table_Info['UID']
        
        print('开始处理特征表:', tmp_feature_table_name)
        
        tmp_sql_command = """
                        SELECT
                            *
                        FROM
                            """ + tmp_feature_table_name + """
                        WHERE 
                            """ + tmp_aim_column_name +""" IS NOT NULL AND
                            dt = '""" + Aim_Feature_Table_dt + """'
                        """
        tmp_feature_table_rdd = Spark_Session.sql(tmp_sql_command)

        # 确保添加特征目标列的列名为tmp_node_class + '_UID'
        tmp_feature_table_rdd = tmp_feature_table_rdd.withColumnRenamed(tmp_aim_column_name, Aim_Node_Type + '_UID')
        
        # 通过persist保留计算结果
        tmp_feature_table_rdd = tmp_feature_table_rdd.persist()

        tmp_feature_table_rdd_raw_count = tmp_feature_table_rdd.count()

        if tmp_feature_table_rdd_raw_count == 0:
            print('Error: 特征表', tmp_feature_table_name, '为空，得及时处理')
        else:
            # 对特征表去重（理论上不需要，但防止有不符合规范的表）
            tmp_feature_table_rdd = tmp_feature_table_rdd.dropDuplicates([Aim_Node_Type + '_UID'])

            # 通过persist保留计算结果
            tmp_feature_table_rdd = tmp_feature_table_rdd.persist()

            tmp_feature_table_rdd_count = tmp_feature_table_rdd.count()

            if tmp_feature_table_rdd_raw_count != tmp_feature_table_rdd_count:
                print('Error: 特征表特征表', tmp_feature_table_name, '在时间', Aim_Feature_Table_dt, 
                    '内部有重复UID，得及时修改, 目前先保留第一条信息，原始行数为:', tmp_time_feature_table_rdd_raw_count,
                    '去重后为:', tmp_time_feature_table_rdd_count)
        
        # 获取特征表的格式信息
        tmp_feature_table_rdd_json = json.loads(tmp_feature_table_rdd.schema.json())['fields']

        # 记录有效的特征列名
        tmp_useful_feature_cols_list = []

        # 记录有效的特征注释名
        tmp_useful_feature_cols_comments_list = []

        # 只保留其中有效的列（entity_id加数值类型的列，*待优化，数据格式标准化后改为entity_id及其之后的列）
        for tmp_col_info in tmp_feature_table_rdd_json:
            col = tmp_col_info['name']
            col_type = tmp_col_info['type']

            if col == (Aim_Node_Type + '_UID'):
                continue

            if 'entity_id' in col:
                continue

            if col_type in ['int', 'integer', 'float', 'bigint','double', 'long']:
                tmp_transferred_column_name = (col + '___' + tmp_feature_table_name.split('.')[-1])
                if 'comment' in tmp_col_info['metadata']:
                    col_comment = (tmp_col_info['metadata']['comment'] + '___' + tmp_feature_table_name.split('.')[-1])
                else:
                    col_comment = tmp_transferred_column_name

                tmp_feature_table_rdd = tmp_feature_table_rdd.withColumnRenamed(col, tmp_transferred_column_name)

                tmp_useful_feature_cols_list.append(tmp_transferred_column_name)

                tmp_useful_feature_cols_comments_list.append(col_comment)
            elif col_type != 'string':
                print('-----------------------------------------------------------')
                print('WARNING:stange_type:', col, col_type)
                print('-----------------------------------------------------------')

        tmp_feature_table_rdd = tmp_feature_table_rdd.select([Aim_Node_Type + '_UID'] + tmp_useful_feature_cols_list)

        tmp_all_useful_feature_cols_list.extend(tmp_useful_feature_cols_list)
        tmp_all_useful_feature_cols_comments_list.extend(tmp_useful_feature_cols_comments_list)

        print('特征表'+ tmp_feature_table_name + '添加特征数:', len(tmp_useful_feature_cols_list), '当前累计的特征列的总数为:',
            len(tmp_all_useful_feature_cols_list))

        # 通过persist保留计算结果
        tmp_feature_table_rdd = tmp_feature_table_rdd.persist()

        # 计算除了UID列的min, max, mean, std，并转化为pandas
        tmp_feature_table_summary_pd = tmp_feature_table_rdd.drop(Aim_Node_Type + '_UID').summary("min", "max", "mean", "stddev").toPandas()

        # 查看是否有无效列(特征都为同一值)，及时提醒
        tmp_summary_min = tmp_feature_table_summary_pd[tmp_feature_table_summary_pd['summary'] == 'min'].values[0]
        tmp_summary_max = tmp_feature_table_summary_pd[tmp_feature_table_summary_pd['summary'] == 'max'].values[0]

        tmp_problem_columns = np.array(tmp_feature_table_summary_pd.columns)[tmp_summary_min == tmp_summary_max]

        if tmp_problem_columns.shape[0] > 0:
            print('ERROR: 特征表', tmp_feature_table_name, '在时间', Aim_Feature_Table_dt, 
                '存在一些列的全部行都是一个值，具体情况如下，得及时修改')
            print(dict(tmp_feature_table_summary_pd[tmp_problem_columns].iloc[0]))
        
        # 合并各个特征表的数据
        if len(tmp_useful_feature_cols_list) == len(tmp_all_useful_feature_cols_list):
            tmp_combined_feature_table_rdd = tmp_feature_table_rdd
        else:
            tmp_combined_feature_table_rdd = tmp_combined_feature_table_rdd.join(tmp_feature_table_rdd, Aim_Node_Type + '_UID', 'outer')
           
        # 如果累计的特征列已经超过目标长度，或全部的特征表已经读取完毕，则进行上传
        if ((Max_Column_Number > 0 and len(tmp_all_useful_feature_cols_list) >= Max_Column_Number) or 
           (tmp_feature_table_i == (len(Aim_Node_Feature_Table_Info["Feature_Data_List"]) - 1))):
            # 确保顺序正确
            tmp_combined_feature_table_rdd = tmp_combined_feature_table_rdd.select([Aim_Node_Type + '_UID'] + 
                                                            tmp_all_useful_feature_cols_list)

            tmp_combined_feature_table_rdd = tmp_combined_feature_table_rdd.persist()

            tmp_combined_feature_table_rdd_count = tmp_combined_feature_table_rdd.count()

            # 设置对应表名
            tmp_combined_feature_table_name = ('tmp.tmp___JingLian_Combined_Feature_Table___' + Aim_Node_Type + '___' +
                                    Feature_Table_Comment + '_' + str(Feature_Table_Upload_Count_list[0]))
            print('输出表名为', tmp_combined_feature_table_name, '包含UID总数为:', tmp_combined_feature_table_rdd_count)

            # 获取各列的类型
            tmp_all_feature_type_list = []
            for _, col_type in tmp_combined_feature_table_rdd.dtypes: 
                tmp_all_feature_type_list.append(col_type)

            # 创建表（如果特征表已存在，会自动不进行创建）
            Pyspark_Create_Table(Spark_Session, tmp_combined_feature_table_name, 
                          [Aim_Node_Type + '_UID'] + tmp_all_useful_feature_cols_list, 
                          tmp_all_feature_type_list,
                          [Aim_Node_Type + '_UID'] + tmp_all_useful_feature_cols_comments_list)

            # 设定临时view的名称
            tmp_view_name = tmp_combined_feature_table_name.split('tmp.tmp___')[1]

            # 创建临时view
            tmp_combined_feature_table_rdd.createTempView(tmp_view_name)

            # 上传特征
            sql_str = """insert overwrite table """ + tmp_combined_feature_table_name + """ 
                     partition(dt='""" + Aim_Feature_Table_dt + """')(
                     select * from """ + tmp_view_name + """
                    )    
                    """

            Spark_Session.sql(sql_str)

            # 删除临时view
            Spark_Session.catalog.dropTempView(tmp_view_name)
            
            tmp_combine_end_time = datetime.now()
            print('完成合并后特征的上传，此次合并总共花费时间:', (tmp_combine_end_time - tmp_combine_start_time))
            
            # 统计合并后的表的信息
            tmp_combined_feature_table_summary_rdd = tmp_combined_feature_table_rdd.drop(Aim_Node_Type + '_UID').summary("min", "max", 
                                                                                     "mean", "stddev")
            
            # 设置上传对应的表名
            tmp_combined_feature_table_summary_name = ('tmp.tmp___JingLian_Combined_Feature_Table___' + Aim_Node_Type + '_Summary___' +
                                         Feature_Table_Comment + '_' + str(Feature_Table_Upload_Count_list[0]))
            if len(tmp_combined_feature_table_summary_name) > 128:
                tmp_combined_feature_table_summary_name = tmp_combined_feature_table_summary_name[:128]
                print('只能保留表名的前128位')

            print('输出总结表名为', tmp_combined_feature_table_summary_name)
            
            # 上传统计信息
            Upload_RDD_Data_to_Database(Spark_Session, tmp_combined_feature_table_summary_name, tmp_combined_feature_table_summary_rdd,
                               Aim_Feature_Table_dt, [], [])
            
            tmp_summary_end_time = datetime.now()
            print('完成统计信息的上传,总共花费时间:', (tmp_summary_end_time - tmp_combine_end_time))
            
            # 清空记录
            tmp_feature_table_rdd = tmp_feature_table_rdd.unpersist()
            tmp_combined_feature_table_rdd = tmp_combined_feature_table_rdd.unpersist()
            
            tmp_all_useful_feature_cols_list = []
            tmp_all_useful_feature_cols_comments_list = []
            Feature_Table_Upload_Count_list[0] = Feature_Table_Upload_Count_list[0] + 1
            Feature_Table_Processed_Count_list[0] = tmp_feature_table_i + 1
            
            tmp_combine_start_time = datetime.now()
            
            print('--------------------------------------------------------------------------------')
                
    return