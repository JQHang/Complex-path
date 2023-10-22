from pyspark.sql.window import Window
import pyspark.sql.functions as F
from pyspark.sql.functions import col, row_number,lit, broadcast
from pyspark.sql.types import *

import json
import random
import numpy as np
import pandas as pd
from datetime import datetime

"""
作用：
    根据指定表名和格式创建表(若已存在该表，则会进行drop)

输入：
    Spark_Session：pyspark接口
    Table_Name：目标表名
    Table_Columns_List：目标表列名
    Table_Columns_Type_List: 目标表列的类型
    Table_Columns_Comment_List: 目标表列的注释
    
返回值：
    无
"""
def Pyspark_Create_Table(Spark_Session, Table_Name, Table_Columns_List, Table_Columns_Type_List, Table_Columns_Comment_List):
    tmp_sql = """CREATE TABLE IF NOT EXISTS """ + Table_Name + """ ( """
    
    for tmp_column_i in range(len(Table_Columns_List)):
        tmp_column = Table_Columns_List[tmp_column_i]
        tmp_column_type = Table_Columns_Type_List[tmp_column_i]
        tmp_column_comment = Table_Columns_Comment_List[tmp_column_i]
        
        tmp_sql = tmp_sql + tmp_column + ' ' + tmp_column_type + ' COMMENT \"' + tmp_column_comment + "\""
        
        if tmp_column_i != (len(Table_Columns_List) - 1):
            tmp_sql = tmp_sql + ','
                
    tmp_sql = tmp_sql + """ )
                PARTITIONED BY
                (
                    dt string
                )
                stored AS orc tblproperties
                 (
                     'orc.compress' = 'SNAPPY'
                 )"""
    
    Spark_Session.sql(tmp_sql)
    
    return

"""
作用：
    根据指定表名和数据创建表并上传

输入：
    Spark_Session：pyspark接口
    Table_Name：目标表名
    Data_To_Upload:要上传的数据
    Upload_table_dt:目标时间分区
    Table_Columns_List：目标表列名
    Table_Columns_Comment_List: 目标表列的注释
    
返回值：
    无
"""
def Upload_RDD_Data_to_Database(Spark_Session, Table_Name, Data_To_Upload, Upload_table_dt, Table_Columns_List, Table_Columns_Comment_List):
    upload_start_time = datetime.now()
    
    if len(Table_Columns_List) != 0:
        print('预设列名为:', Table_Columns_List[0:4])
    
    # 如果列名长度不等于注释名长度，则直接清空
    if len(Table_Columns_List) != len(Table_Columns_Comment_List):
        Table_Columns_List = []
        Table_Columns_Comment_List = []
    
    # 是否有目标列名和注释
    has_table_columns = False
    if len(Table_Columns_List) != 0:
        has_table_columns = True
        
        Data_To_Upload = Data_To_Upload.select(Table_Columns_List)
    
    # 获取要上传的表的信息
    tmp_upload_table_rdd_json = json.loads(Data_To_Upload.schema.json())['fields']

    # 获取各列的类型
    Table_Columns_Type_List = []
    for tmp_col_info in tmp_upload_table_rdd_json:
        col = tmp_col_info['name']
        col_type = tmp_col_info['type']

        Table_Columns_Type_List.append(col_type)
        
        if not has_table_columns:
            Table_Columns_List.append(col)
            Table_Columns_Comment_List.append(col)
        
    # 创建表（如果特征表已存在，会自动不进行创建）
    Pyspark_Create_Table(Spark_Session, Table_Name, Table_Columns_List, Table_Columns_Type_List, Table_Columns_Comment_List)

    # 创建临时view
    Data_To_Upload.createTempView("tmp_view_for_upload")

    # 上传特征（如果是第一次上传，则清空对应dt的数据）
    sql_str = """insert overwrite table """ + Table_Name + """ 
             partition(dt='""" + Upload_table_dt + """')(
             select * from tmp_view_for_upload
            )    
            """

    Spark_Session.sql(sql_str)

    # 删除临时view
    Spark_Session.catalog.dropTempView("tmp_view_for_upload")

    upload_end_time = datetime.now()
    print('完成目标表的上传, 上传函数消耗时间为:', (upload_end_time - upload_start_time))
    
    return

"""
作用：
    针对指定的rdd数据，计算groupby后的结果

输入：
    Spark_Session：pyspark接口
    Aim_Table_Rdd：目标pyspark数据
    Aim_Column_name: 目标列列名
    Feature_Columns_List：目标特征列
    Groupby_Type_List：目标groupby运算方式

返回值：
    采样后的pyspark数据
"""
def Groupby_Feature_Table(Spark_Session, Aim_Table_Rdd, Aim_Column_name, Feature_Columns_List, Groupby_Type_List):
    Aim_Table_Rdd.createOrReplaceTempView("EMP")
    
    sql_str = "SELECT " + Aim_Column_name + ','
    for tmp_groupby_type in Groupby_Type_List:
        if tmp_groupby_type in ['AVG', 'SUM', 'MAX', 'MIN']:
            for tmp_feature_column in Feature_Columns_List:
                sql_str = (sql_str + ' ' + tmp_groupby_type + "(" + tmp_feature_column + ") as " + tmp_groupby_type  
                        + '_' + tmp_feature_column + ',')
        elif tmp_groupby_type == 'COUNT':
            sql_str = (sql_str + ' ' + tmp_groupby_type + "(*) as Groupby_COUNT,")
    
    # 删去最后的逗号
    sql_str = sql_str[:-1]
    
    sql_str = sql_str + " FROM EMP GROUP BY " + Aim_Column_name
    
    tmp_groupby_result = Spark_Session.sql(sql_str)
    
    return tmp_groupby_result

"""
作用：
    针对指定的rdd数据，计算groupby后的结果

输入：
    Spark_Session：pyspark接口
    Aim_Table_Rdd：目标pyspark数据
    Groupby_Column_List: 目标列列名
    Aggregate_Columns_List：聚合列列名
    Groupby_Type_List：目标groupby运算方式

返回值：
    采样后的pyspark数据
"""
def Groupby_Pyspark_Table(Spark_Session, Aim_Table_Rdd, Groupby_Column_List, Aggregate_Columns_List, Groupby_Type_List):
    Aim_Table_Rdd.createOrReplaceTempView("EMP")
    
    sql_str = "SELECT" 
    for tmp_column_name in Groupby_Column_List:
        sql_str = sql_str + ' ' + tmp_column_name + ','
    
    for tmp_groupby_type in Groupby_Type_List:
        if tmp_groupby_type in ['AVG', 'SUM', 'MAX', 'MIN']:
            for tmp_feature_column in Aggregate_Columns_List:
                sql_str = (sql_str + ' ' + tmp_groupby_type + "(" + tmp_feature_column + ") as " + tmp_groupby_type  
                        + '_' + tmp_feature_column + ',')
        elif tmp_groupby_type == 'COUNT':
            sql_str = (sql_str + ' ' + tmp_groupby_type + "(*) as Groupby_COUNT,")
    
    # 删去最后的逗号
    sql_str = sql_str[:-1]
    
    sql_str = sql_str + " FROM EMP GROUP BY " + Groupby_Column_List[0]
    
    for tmp_column_name in Groupby_Column_List[1:]:
        sql_str = sql_str + ', ' + tmp_column_name
    
    tmp_groupby_result = Spark_Session.sql(sql_str)
    
    return tmp_groupby_result


"""
作用：
    根据范围，保留pyspark文件中的指定范围的行

输入：
    Spark_Session：pyspark接口
    tmp_aim_small_rdd：目标pyspark数据
    tmp_node_range_np：目标行范围

返回值：
    采样后的pyspark数据
"""
def sample_rdd_from_aim_row(Spark_Session, tmp_aim_small_rdd, tmp_node_range_np, show_info = True):
    # 加上临时id号
    w = Window().orderBy(lit('tmp_order_lit'))
    tmp_aim_small_rdd = tmp_aim_small_rdd.withColumn("tmp_id", row_number().over(w) - 1)

    if show_info:
        print('保留指定范围行:已完成临时id号的生成')
    
    # 将行号转化为rdd格式
    aim_tmp_id_rdd = Spark_Session.createDataFrame(pd.DataFrame({'tmp_id':tmp_node_range_np}),["tmp_id"])
    
    if show_info:
        print('生成目标id号表')

    # 通过join获取保留的行号(如果列数较少，就先broadcast再join)
    if tmp_node_range_np.shape[0] > 100000:
        tmp_sampled_aim_small_rdd = aim_tmp_id_rdd.join(tmp_aim_small_rdd, 'tmp_id', 'inner')
    else:
        tmp_sampled_aim_small_rdd = aim_tmp_id_rdd.join(broadcast(tmp_aim_small_rdd), 'tmp_id', 'inner')
        
    # 删去临时id号
    tmp_sampled_aim_small_rdd = tmp_sampled_aim_small_rdd.drop('tmp_id')
    
    return tmp_sampled_aim_small_rdd

"""
作用：
    根据范围，保留pyspark文件中的指定范围的行

输入：
    Spark_Session：pyspark接口
    tmp_aim_small_rdd：目标pyspark数据
    tmp_node_range_start：目标行起始范围
    tmp_node_range_end：目标行终止范围

返回值：
    采样后的pyspark数据
"""
def sample_rdd_from_aim_range(Spark_Session, tmp_aim_small_rdd, tmp_node_range_start, tmp_node_range_end, show_info = True):
    # index号得从低到高，且连续，且保证一致
    
    # 加上临时id号
    w = Window().orderBy(lit('tmp_order_lit'))
    tmp_aim_small_rdd = tmp_aim_small_rdd.withColumn("tmp_id", row_number().over(w) - 1)

    if show_info:
        print('保留指定范围行:已完成临时id号的生成')
    
    tmp_aim_small_rdd = tmp_aim_small_rdd.where((tmp_aim_small_rdd.tmp_id >= tmp_node_range_start) & 
                                  (tmp_aim_small_rdd.tmp_id < tmp_node_range_end))
        
    # 删去临时id号
    tmp_aim_small_rdd = tmp_aim_small_rdd.drop('tmp_id')
    
    return tmp_aim_small_rdd

"""
作用：
    随机采样pyspark数据中的n行（最好是小文件，因为非常耗时）

输入：
    Spark_Session：pyspark接口
    tmp_aim_small_rdd：目标pyspark数据
    tmp_sample_n_num：要随机采样的行数
    tmp_aim_small_rdd_count：目标pyspark数据的总行数（可以不设置，但就得算一遍count，较为耗时）

返回值：
    采样后的pyspark数据
"""
def sample_random_n_samples_for_samll_rdd(Spark_Session, tmp_aim_small_rdd, tmp_sample_n_num, tmp_aim_small_rdd_count = 0):
    # 加上临时id号
    w = Window().orderBy(lit('tmp_order_lit'))
    tmp_aim_small_rdd = tmp_aim_small_rdd.withColumn("tmp_id", row_number().over(w) - 1)

    print('随机采样n个样本任务:已完成临时id号的生成')
#     print('最大行号为:', tmp_aim_small_rdd.agg({'tmp_id': "max"}).collect()[0])

    if tmp_aim_small_rdd_count < 1:
        tmp_aim_small_rdd_count = tmp_aim_small_rdd.count()
    
    # 生成要选取的行号
    tmp_sample_ids = random.sample(range(0, tmp_aim_small_rdd_count), tmp_sample_n_num)
    tmp_sample_ids.sort()

    # 将行号转化为rdd格式
    aim_tmp_id_rdd = Spark_Session.createDataFrame(pd.DataFrame({'tmp_id':tmp_sample_ids}),["tmp_id"])
    print('生成目标id号表')

    # 通过join获取保留的行号
    tmp_sampled_aim_small_rdd = aim_tmp_id_rdd.join(tmp_aim_small_rdd, 'tmp_id', 'inner')

    # 删去临时id号
    tmp_sampled_aim_small_rdd = tmp_sampled_aim_small_rdd.drop('tmp_id')
    
    return tmp_sampled_aim_small_rdd

"""
作用：
    在pyspark中对每个样本groupby后再选取权重最高的n行（最好是小文件，因为非常耗时，如果有同权重的，则从中随机采样）

输入：
    Spark_Session：pyspark接口
    tmp_aim_small_rdd：目标pyspark数据
    tmp_aim_column_name: groupby目标列
    tmp_sample_n_num：要采样的行数
    tmp_Weight_Column_name:权重列列名
    random_sample_for_same_number:是否给相同的数值生成不同的序号

返回值：
    采样后的pyspark数据
"""
def sample_top_n_groupby_samples_for_samll_rdd(Spark_Session, tmp_aim_small_rdd, tmp_aim_column_name, tmp_sample_n_num, tmp_Weight_Column_name,
                                random_sample_for_same_number = True):
    if random_sample_for_same_number:
        groupby_window = Window.partitionBy(tmp_aim_small_rdd[tmp_aim_column_name]).orderBy(col(tmp_Weight_Column_name).desc(), F.rand())
    else:
        groupby_window = Window.partitionBy(tmp_aim_small_rdd[tmp_aim_column_name]).orderBy(col(tmp_Weight_Column_name).desc())
    
    tmp_sampled_aim_small_rdd = tmp_aim_small_rdd.select('*', F.rank().over(groupby_window).alias('rank')).filter(F.col('rank') <= 
                                                                              tmp_sample_n_num).drop('rank')
    
    return tmp_sampled_aim_small_rdd


"""
作用：
    在pyspark中对每个样本groupby后再随机采样n行（最好是小文件，因为非常耗时）

输入：
    Spark_Session：pyspark接口
    tmp_aim_small_rdd：目标pyspark数据
    tmp_aim_column_name: groupby目标列
    tmp_sample_n_num：groupby后再随机采样的行数

返回值：
    采样后的pyspark数据
"""
def sample_random_n_groupby_samples_for_samll_rdd(Spark_Session, tmp_aim_small_rdd, tmp_aim_column_name, tmp_sample_n_num):
    
    groupby_window = Window.partitionBy(tmp_aim_small_rdd[tmp_aim_column_name]).orderBy(F.rand())
    tmp_sampled_aim_small_rdd = tmp_aim_small_rdd.select('*', F.rank().over(groupby_window).alias('rank')).filter(F.col('rank') <= 
                                                                              tmp_sample_n_num).drop('rank')
    
    return tmp_sampled_aim_small_rdd
