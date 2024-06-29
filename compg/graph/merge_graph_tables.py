from ..python import setup_logger, time_costing, ensure_logger
from ..hdfs import create_hdfs_directory, list_files_in_hdfs_dir
from ..hdfs import check_hdfs_file_exists, create_hdfs_marker_file, read_hdfs_marker_file
from ..graph import edge_tables_dir, node_tables_dir, label_tables_dir

def read_edge_table(spark, edge_table_config, edge_times, target_node = None, return_edge_feats = True, vectorize_feats = False, logger = None):
    """
    读取关系表，并基于node_limit和edge_limit过滤出有效边
    """
    # edge_limit里面有属性限制和关联度-Connectivity
    print('Read edge table:', edge_table_config['Table_Name'], 'at times:', edge_times)
    
    table_data_path = edge_tables_dir + edge_table_config['Table_Name']
    table_data_paths = [table_data_path + f"/dt={dt}" for dt in edge_times]

    edge_table_df = spark.read.option("basePath", table_data_path).parquet(*table_data_paths)
    
    # ID columns
    edge_id_columns = []
    
    # Change the node column and time column name
    head_node_std_name = edge_table_config['Head_Node_Index'] + '_' + edge_table_config['Head_Node_Column']
    edge_table_df = edge_table_df.withColumnRenamed(edge_table_config['Head_Node_Column'], head_node_std_name)
    edge_table_df = edge_table_df.filter(col(head_node_std_name).isNotNull() & (col(head_node_std_name) != ""))
    edge_id_columns.append(head_node_std_name)
    
    if 'tail_node_std_name' in edge_table_config:
        tail_node_std_name = edge_table_config['tail_node_std_name']
        edge_table_df = edge_table_df.withColumnRenamed(edge_table_config['tail_node_raw_name'], tail_node_std_name)
        edge_table_df = edge_table_df.filter(col(tail_node_std_name).isNotNull() & (col(tail_node_std_name) != ""))
        edge_id_columns.append(tail_node_std_name)
    else:
        for tail_node_index in range(len(edge_table_config['tail_node_raw_names'])):
            tail_node_raw_name = edge_table_config['tail_node_raw_names'][tail_node_index]
            tail_node_std_name = edge_table_config['tail_node_std_names'][tail_node_index]
            
            edge_table_df = edge_table_df.withColumnRenamed(tail_node_raw_name, tail_node_std_name)
            edge_table_df = edge_table_df.filter(col(tail_node_std_name).isNotNull() & (col(tail_node_std_name) != ""))
            edge_id_columns.append(tail_node_std_name)
        
    edge_table_df = edge_table_df.withColumnRenamed('dt', 'Feature_Time')
    edge_id_columns.append('Feature_Time')
    
    # Change column name with specific characters
    edge_table_df = sanitize_column_names(edge_table_df)
    
    # Get feature information for the edge
    if 'edge_target_feature_columns' in edge_table_config and isinstance(edge_table_config['edge_target_feature_columns'], list):
        feature_columns = edge_table_config['edge_target_feature_columns']
    else:
        feature_columns, feature_columns_comments = check_numeric_columns(edge_table_df.schema)
    
    # Only Keep the target node columns, time column and edge feature columns
    edge_table_df = edge_table_df.select(edge_id_columns + feature_columns)
    
    ##############################################################################################
    # Only keep edges start from the start node, if provided
    if target_node is not None:
        target_node_df = target_node['data']
        target_node_column_name = target_node['node_column']
        
        target_node_df = target_node_df.withColumnRenamed(target_node_column_name, head_node_std_name)
        target_node_df = target_node_df.select([head_node_std_name, "Feature_Time"]).distinct()
        
        edge_table_df = edge_table_df.join(target_node_df, [head_node_std_name, "Feature_Time"], 'inner')
        
        edge_table_df = edge_table_df.repartition(head_node_std_name, "Feature_Time")
        
    # Groupby edges with the same node and accumulate the edge features
    if len(feature_columns) > 0:
        edge_table_df = Groupby_Feature_Table(spark, edge_table_df, edge_id_columns, feature_columns, ['SUM'])
        for feat_col in feature_columns:
            # 待优化命名**
            edge_table_df = edge_table_df.withColumnRenamed('SUM_' + feat_col, feat_col)
    else:
        edge_table_df = edge_table_df.distinct()
        
    ##############################################################################################
    # Add feature limitation
    if 'Edge_Feature_Limits' in edge_table_config and edge_table_config['Edge_Feature_Limits'] != '':
        edge_table_df = edge_table_df.where(edge_table_config['Edge_Feature_Limits'])
        print('Edge Limitation:', edge_table_config['Edge_Feature_Limits'])
    
    if 'Node_Feature_Limits' in edge_table_config and len(edge_table_config['Node_Feature_Limits']) > 0:
        # 设定Node Feature Limits的目标节点
        target_node_limit = {}
        target_node_limit['data'] = edge_table_df
        target_node_limit['node_column'] = edge_table_config['Node_Feature_Limits']['node_std_name']
        
        # 获取对应的节点类型
        node_feature_table = read_node_table(spark, target_node_limit['node_column'], edge_table_config['Node_Feature_Limits']['node_tables'], 
                                 node_times, target_node_limit, 'columns')
        
        print('Node Limits:', edge_table_config['Node_Feature_Limits']['limit'])
        edge_table_df = node_feature_table['data'].where(edge_table_config['Node_Feature_Limits']['limit'])

        # Only keep the raw columns in the edge table
        edge_table_df = edge_table_df.select(edge_id_columns + feature_columns)
    
    ##############################################################################################
    # Add edge neighbor limits
    if 'Edge_Neighbor_Limits' in edge_table_config:
        edge_neighbor_limit_type = 'Random_N'
        if 'Type' in edge_table_config['Edge_Neighbor_Limits']:
            edge_neighbor_limit_type = edge_table_config['Edge_Neighbor_Limits']['Type']
            
        edge_neighbor_limit_max_num = edge_table_config['Edge_Neighbor_Limits']['Max_Num']
        
        edge_neighbor_limit_feat_columns = []
        if 'Feat_Columns' in edge_table_config['Edge_Neighbor_Limits']:
            edge_neighbor_limit_feat_columns = edge_table_config['Edge_Neighbor_Limits']['Feat_Columns']
        
        print(f'{edge_neighbor_limit_type} Sampling')
        print(f'Max neighbor count: {edge_neighbor_limit_max_num}, Target feature columns: {edge_neighbor_limit_feat_columns}')

        if edge_neighbor_limit_type == 'Threshold_N':
            edge_table_df = Spark_Threshold_N_Sample(spark, edge_table_df, [head_node_std_name], edge_neighbor_limit_max_num)
        elif edge_neighbor_limit_type == 'Top_N':
            edge_table_df = Spark_Top_N_Sample(spark, edge_table_df, [head_node_std_name], edge_neighbor_limit_max_num, 
                                    tmp_Max_Sample_Feat_Columns)
        else:
            edge_table_df = Spark_Random_N_Sample(spark, edge_table_df, [head_node_std_name], edge_neighbor_limit_max_num)

    edge_table = {}
    edge_table['data'] = edge_table_df
    edge_table['feature_columns'] = feature_columns
    edge_table['feature_comments'] = feature_columns_comments
    
    return edge_table


def get_merged_edge_or_path_tables(spark, edge_table_configs, edge_times, target_node = None, return_edge_feats = True, vectorize_feats = False, 
                        logger = None):
    """
    读取一组有同样起始和终止点的关系表或路径表，合并后再返回
    """
    
    # 转化为单独的edge table config
    
    return