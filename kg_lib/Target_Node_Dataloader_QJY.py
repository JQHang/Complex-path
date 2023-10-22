import pandas as pd
import os
from pyspark.sql.types import *

"""
作用：
    根据目标点的配置文件和限制时间给出全量的目标点及相关信息

输入：
    Spark_Session：pyspark接口
    Label_Data_Config_dict：目标点配置文件
    tmp_aim_time_start：目标样本的起始时间
    tmp_aim_time_end：目标样本的终止时间
    tmp_store_dir: 标签表存储文件夹

返回值：
    包含全量的目标点及相关信息的字典，键值'Node_Type'对应目标点的节点类型，键值'Data'对应目标点的全量数据，键值'Monthly_dt'对应目标点对应的月份的1号，从而去取对应的关系表和特征表
"""


def get_aim_UID_with_label_rdd(Spark_Session, Label_Data_Config_dict, tmp_aim_time_start, tmp_aim_time_end,
                               tmp_store_dir):
    Aim_Node_type = Label_Data_Config_dict['Node_Type']
    Aim_Node_Column = Label_Data_Config_dict['Node_Column']
    Aim_Label_Column = Label_Data_Config_dict['Label_Column']
    Aim_Time_Column = Label_Data_Config_dict['Time_Column']
    Aim_table_name = Label_Data_Config_dict['Table_Name']
    Aim_table_dt = Label_Data_Config_dict['dt']
    # if 'Host_UID' in Label_Data_Config_dict.keys():
    #     Aim
    Aim_Node_UID_Name = Aim_Node_type + '_UID'

    # 标签表存储位置+文件名
    tmp_output_data_label_file = tmp_store_dir + 'Target_Node.pkl'

    # 查询是否已存在
    if not os.path.exists(tmp_output_data_label_file):
        # 不存在则进行运算
        if 'Positive_Label_Requirements' not in Label_Data_Config_dict.keys():
            tmp_sql_command = """
                SELECT
                    """ + Aim_Node_Column + """ AS """ + Aim_Node_UID_Name + """,
                    MAX(""" + Aim_Label_Column + """) AS Label
                    {}
                FROM
                    """ + Aim_table_name + """
                WHERE 
                    dt = '""" + Aim_table_dt + """'
                    AND """ + Aim_Time_Column + """ >= '""" + tmp_aim_time_start + """'
                    AND """ + Aim_Time_Column + """ <  '""" + tmp_aim_time_end + """'
                    AND """ + Aim_Label_Column + """ IN (0, 1)"""
        else:
            tmp_sql_command = """
                SELECT
                    b."""+Aim_Node_UID_Name+""",
                    b.Label IN (0,1)
                FROM
                (
                SELECT
                    a."""+Aim_Node_UID_Name+""",
                    (case 
                        when a.Label IN ("""+Label_Data_Config_dict['Positive_Label_Requirements']+""") then 1
                        when a.Label IN ("""+Label_Data_Config_dict['Negative_Label_Requirements']+""") then 0
                        else -1
                    end) as Label
                FROM
                    (
                    SELECT
                        """ + Aim_Node_Column + """ AS """ + Aim_Node_UID_Name + """,
                        MAX(""" + Aim_Label_Column + """) AS Label
                        {}
                    FROM
                        """ + Aim_table_name + """
                    WHERE 
                        dt = '""" + Aim_table_dt + """'
                        AND """ + Aim_Time_Column + """ >= '""" + tmp_aim_time_start + """'
                        AND """ + Aim_Time_Column + """ <  '""" + tmp_aim_time_end + """'
                    )a
                )b
                """
        if 'Host_UID' in Label_Data_Config_dict.keys():
            tmp_sql_command = tmp_sql_command.format(',' + Label_Data_Config_dict['User_Column'] + 'AS User_UID')
        else:
            tmp_sql_command = tmp_sql_command.format('')
        if 'Limits' in Label_Data_Config_dict and Label_Data_Config_dict['Limits'] != '':
            tmp_sql_command = tmp_sql_command + '\nAND ' + Label_Data_Config_dict['Limits']
            print('标签表限制条件为', Label_Data_Config_dict['Limits'])

        tmp_sql_command = tmp_sql_command + """\nGROUP BY
                """ + Aim_Node_Column + """
        """
        print('sql_get_target_node:%s' % (tmp_sql_command))
        tmp_aim_entity_rdd = Spark_Session.sql(tmp_sql_command)

        tmp_aim_entity_info_dict = {'Node_Type': Aim_Node_type,
                                    'Data': tmp_aim_entity_rdd,
                                    'Monthly_dt': tmp_aim_time_start[0:8] + '01'}

        # 存储运算结果
        tmp_aim_entity_pd = tmp_aim_entity_info_dict['Data'].toPandas()

        tmp_aim_entity_pd.to_pickle(tmp_output_data_label_file)

    else:
        print('标签表已存在，直接读取')

        # 已存在则直接读取
        tmp_aim_entity_pd = pd.read_pickle(tmp_output_data_label_file)

        # 转为rdd格式
        tmp_target_node_table_schema = StructType([StructField(Aim_Node_UID_Name, StringType(), True),
                                                   StructField("Label", IntegerType(), True)])
        tmp_aim_entity_rdd = Spark_Session.createDataFrame(tmp_aim_entity_pd, tmp_target_node_table_schema)

        tmp_aim_entity_info_dict = {'Node_Type': Aim_Node_type,
                                    'Data': tmp_aim_entity_rdd,
                                    'Monthly_dt': tmp_aim_time_start[0:8] + '01'}

    print('正样本数目:', tmp_aim_entity_pd[tmp_aim_entity_pd['Label'] == 1].shape)
    print('负样本数目:', tmp_aim_entity_pd[tmp_aim_entity_pd['Label'] == 0].shape)

    return tmp_aim_entity_info_dict