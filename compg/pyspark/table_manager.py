# 该文件主要用于

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
def Pyspark_Create_Table(Spark_Session, Table_Name, Table_Columns_List, Table_Columns_Type_List, Table_Columns_Comment_List = None):
    tmp_sql = """CREATE TABLE IF NOT EXISTS """ + Table_Name + """ ( """
    
    for tmp_column_i in range(len(Table_Columns_List)):
        tmp_column = Table_Columns_List[tmp_column_i]
        tmp_column_type = Table_Columns_Type_List[tmp_column_i]
        
        tmp_sql = tmp_sql + tmp_column + ' ' + tmp_column_type
        
        if Table_Columns_Comment_List != None:
            tmp_column_comment = Table_Columns_Comment_List[tmp_column_i]
            tmp_sql = tmp_sql + ' COMMENT \"' + tmp_column_comment + "\""
        
        if tmp_column_i != (len(Table_Columns_List) - 1):
            tmp_sql = tmp_sql + ', '
                
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
def Upload_RDD_Data_to_Database(Spark_Session, Table_Name, Data_To_Upload, Upload_table_dt, Set_Table_Columns_List = [], 
                      Set_Table_Columns_Comment_List = [], batch_count = 1):
    upload_start_time = datetime.now()
    
    # Get Table Info
    Table_Columns_List = []
    Table_Columns_Type_List = []
    tmp_upload_table_rdd_json = json.loads(Data_To_Upload.schema.json())['fields']
    for tmp_col_info in tmp_upload_table_rdd_json:
        col = tmp_col_info['name']
        col_type = tmp_col_info['type']

        Table_Columns_List.append(col)
        Table_Columns_Type_List.append(col_type)
    
    # Change column name if len is equal
    if len(Table_Columns_List) == len(Set_Table_Columns_List):
        Table_Columns_List = Set_Table_Columns_List
    
    # Clear comment list if len isnot equal
    if len(Table_Columns_List) == len(Set_Table_Columns_Comment_List):
        # Create Table
        Pyspark_Create_Table(Spark_Session, Table_Name, Table_Columns_List, Table_Columns_Type_List, Set_Table_Columns_Comment_List)
    else:
        Pyspark_Create_Table(Spark_Session, Table_Name, Table_Columns_List, Table_Columns_Type_List)

    if batch_count > 1:
        # 定义批次大小的比例
        batch_ratios = [1.0 / batch_count] * (batch_count - 1)
        batch_ratios.append(1.0 - sum(batch_ratios))

        # 使用 randomSplit 将 DataFrame 分成多个小 DataFrame
        Data_To_Upload_list = Data_To_Upload.randomSplit(batch_ratios)
    else:
        Data_To_Upload_list = [Data_To_Upload]
        
    for idx, small_df in tqdm(enumerate(Data_To_Upload_list), total=len(Data_To_Upload_list), desc="Uploading batches"):
        # 为每个小 DataFrame 创建一个临时视图
        view_name = f"temp_view_{idx}"
        small_df.createOrReplaceTempView(view_name)

        # 对于第一个 DataFrame, 使用 INSERT OVERWRITE。对于后续的 DataFrame, 使用 INSERT INTO。
        if idx == 0:
            query = f"""
                INSERT OVERWRITE TABLE {Table_Name} PARTITION (dt='{Upload_table_dt}')
                SELECT * FROM {view_name}
            """
        else:
            query = f"""
                INSERT INTO TABLE {Table_Name} PARTITION (dt='{Upload_table_dt}')
                SELECT * FROM {view_name}
            """
            
        Spark_Session.sql(query)
    
    Spark_Session.catalog.dropTempView(view_name)
    
    upload_end_time = datetime.now()
    print('完成目标表的上传, 上传函数消耗时间为:', (upload_end_time - upload_start_time))
    
    return