from compg.python import time_costing, ensure_logger
from compg.hdfs import create_hdfs_directory, check_hdfs_file_exists, hdfs_read_dict, hdfs_save_dict, hdfs_save_text_file
from compg.graph import read_node_table, read_edge_table
from compg.graph import read_complex_path_instances
from compg.pyspark import random_n_sample, top_n_sample, threshold_n_sample, rename_columns

@time_costing
@ensure_logger
def complex_paths_sampling(spark, graph, complex_paths_config, result_data_dir, target_nodes = None, 
                           partition_values = None, return_edge_feats = True, return_node_feats = True, 
                           vectorize_feats = False, logger = None):
    complex_paths = []
    for complex_path_config in complex_paths_config:
        complex_path = complex_path_sampling(spark, graph, complex_path_config, result_data_dir, target_nodes, partition_values, 
                                             return_edge_feats, return_node_feats, vectorize_feats, logger = logger)
        
        complex_paths.append(complex_path)
    
    return complex_paths

@time_costing
@ensure_logger
def complex_path_sampling(spark, graph, complex_path_config, result_data_dir, target_nodes = None, 
                          partition_values = None, return_edge_feats = True, return_node_feats = True, 
                          vectorize_feats = False, logger = None):
    """
    针对给定的复杂路类型获取对应的全部complex-path instances

    输入：

    返回值：
        
    """
    # 函数结束时需要被释放的变量
    df_to_unpersist = []
    
    complex_path_name = complex_path_config["path_name"]
    complex_path_schema = complex_path_config["path_schema"]
    
    complex_path_result_data_dir = graph.output_dir + f'/{result_data_dir}/{complex_path_name}'
    
    logger.info(f"Sampling instances for complex-path: {complex_path_name}")
    logger.info(f"The reulst will be output to: {complex_path_result_data_dir}")

    existing_complex_path, missing_partition_values = read_complex_path_instances(spark, graph, complex_path_config, complex_path_result_data_dir, 
                                                                                  partition_values, logger = logger)
    
    if existing_complex_path is not None:
        return existing_complex_path

    if missing_partition_values is not None:
        partition_values = missing_partition_values
            
    # Record sequential node and edges
    seq_node_edges = []
    for index in range(len(complex_path_schema[0]["head_node_types"])):
        head_node_type = complex_path_schema[0]["head_node_types"][index]
        head_node_index = complex_path_schema[0]["head_node_indexes"][index]
        seq_node_edges.append(("Node", head_node_type, head_node_index))
        
    add_node_features = []
    for hop_k, hop_config in enumerate(complex_path_schema):
        logger.info(f"Process the {hop_k}-th hop config")

        hop_head_columns = []
        for index in range(len(hop_config["head_node_types"])):
            head_node_type = hop_config["head_node_types"][index]
            head_node_index = hop_config["head_node_indexes"][index]
            hop_head_columns.append(f"{head_node_type}_{head_node_index}")

        hop_tail_columns = []
        for index in range(len(hop_config["tail_node_types"])):
            tail_node_type = hop_config["tail_node_types"][index]
            tail_node_index = hop_config["tail_node_indexes"][index]
            hop_tail_columns.append(f"{tail_node_type}_{tail_node_index}")
        
        hop_head_key_columns = list(hop_head_columns)
        hop_key_columns = hop_head_columns + hop_tail_columns
        if graph.partition_key != None:
            hop_head_key_columns.append(graph.partition_key)
            hop_key_columns.append(graph.partition_key)
            
        # 获得该跳的目标节点
        if hop_k == 0:
            hop_target_nodes = target_nodes
        else:
            # 以该跳的头结点在现有路径表内的点作为目标节点
            hop_target_nodes = path_table.select(hop_head_key_columns).distinct().persist()
            df_to_unpersist.append(hop_target_nodes)
            
        # Iterate over relations within the hop configuration
        for relation_k, relation_config in enumerate(hop_config["relation_list"]):
            relation_type = relation_config["relation_type"]

            logger.info(f"Process the {relation_k}-th relation_config with type {relation_type}")
            
            if relation_type == "edge":
                edge_table_schema = relation_config["edge_schema"]
                edge_type = edge_table_schema['edge_table_name']

                for index in range(len(edge_table_schema["head_node_columns"])):
                    head_node_type = edge_table_schema["head_node_types"][index]
                    head_node_column = edge_table_schema["head_node_columns"][index]
                    head_node_index = edge_table_schema["head_node_indexes"][index]

                    hop_target_nodes = hop_target_nodes.withColumnRenamed(f"{head_node_type}_{head_node_index}", head_node_column)
                
                # Get the required edge table
                edge_table = read_edge_table(spark, graph, edge_table_schema, hop_target_nodes, partition_values, logger = logger)
                
                # Change head tail node column names to node type and index
                edge_rename_columns = {}
                for index in range(len(edge_table_schema["head_node_columns"])):
                    head_node_type = edge_table_schema["head_node_types"][index]
                    head_node_column = edge_table_schema["head_node_columns"][index]
                    head_node_index = edge_table_schema["head_node_indexes"][index]

                    edge_rename_columns[head_node_column] = f"{head_node_type}_{head_node_index}"
                    hop_target_nodes = hop_target_nodes.withColumnRenamed(head_node_column, f"{head_node_type}_{head_node_index}")

                for index in range(len(edge_table_schema["tail_node_columns"])):
                    tail_node_type = edge_table_schema["tail_node_types"][index]
                    tail_node_column = edge_table_schema["tail_node_columns"][index]
                    tail_node_index = edge_table_schema["tail_node_indexes"][index]

                    edge_rename_columns[tail_node_column] = f"{tail_node_type}_{tail_node_index}"
            
                # Add feature column name with edge index
                edge_index = edge_table_schema['edge_index']
                for feat_col in graph.graph_summary["edges"][edge_type]["feat_cols"]:
                    edge_rename_columns[feat_col] = f"Edge_{edge_index}_{feat_col}"

                edge_table = rename_columns(spark, edge_table, edge_rename_columns)
                
                seq_node_edges.append(("Edge", edge_type, edge_index))
                
            elif relation_type == "path":
                # Get the required path table
                sub_complex_path = complex_path_sampling(spark, graph, relation_config, result_data_dir, hop_target_nodes, 
                                                                              partition_values, return_edge_feats, return_node_feats, 
                                                                              vectorize_feats, logger = logger)

                edge_table = sub_complex_path["path_table"]
                contained_seq_node_edges = sub_complex_path["seq_node_edges"]
                
                head_nodes_counts = len(hop_config["head_node_types"])
                tail_nodes_counts = len(hop_config["tail_node_types"])
                for element in contained_seq_node_edges[head_nodes_counts:-tail_nodes_counts]:
                    seq_node_edges.append(element)
            
            # Merge with existing relations
            if relation_k > 0:
                relation_join_type = relation_config["join_type"]
                    
                hop_table = hop_table.join(edge_table, hop_key_columns, relation_join_type)
            else:
                hop_table = edge_table

            hop_table = hop_table.repartition(*hop_key_columns)
        
        for index in range(len(hop_config["tail_node_types"])):
            tail_node_type = hop_config["tail_node_types"][index]
            tail_node_index = hop_config["tail_node_indexes"][index]
            seq_node_edges.append(("Node", tail_node_type, tail_node_index))
        
        # Node_limit
        if "node_schemas" in hop_config:
            for node_table_schema in hop_config["node_schemas"]:
                node_type = node_table_schema["node_type"]
                node_column = node_table_schema["node_column"]
                node_index = node_table_schema["node_index"]

                logger.info(f"Node limit for node: {node_type}_{node_index}")
                
                limit_target_nodes = None
                
                node_table = read_node_table(spark, graph, node_table_schema, limit_target_nodes, partition_values, logger = logger)

                # rename
                node_table = node_table.withColumnRenamed(node_column, f"{node_type}_{node_index}")

                for feat_col in node_table_schema["feature_columns"]:
                    node_table = node_table.withColumnRenamed(feat_col, f"{node_type}_{node_index}_{feat_col}")
                    add_node_features.append(f"{node_type}_{node_index}_{feat_col}")
                    
                # Join 
                node_join_type = node_table_schema["join_type"]

                hop_table = hop_table.join(node_table, f"{node_type}_{node_index}", node_join_type)
        
        # Join with previous hops
        if hop_k > 0:
            hop_join_columns = []
            for index in range(len(hop_config["head_node_types"])):
                head_node_type = edge_table_schema["head_node_types"][index]
                head_node_index = edge_table_schema["head_node_indexes"][index]
                hop_join_columns.append(f"{head_node_type}_{head_node_index}")
            
            path_table = path_table.join(hop_table, hop_join_columns, "left")

            path_table = path_table.repartition(*hop_join_columns)
        else:
            path_table = hop_table
            
        # Path Limit 
        if 'path_limit' in hop_config and hop_config['path_limit'] != '':
            path_table = path_table.where(hop_config['path_limit'])
            logger.info(f'Path Limitation: {hop_config['path_limit']}') 

        # Path Sample 
        if 'path_sample' in hop_config:
            path_sample_type = hop_config['path_sample']['type']
            path_sample_count = hop_config['path_sample']['count']
            
            print(f'Path Sampling: {path_sample_type}, {path_sample_count}')
    
            if path_sample_type == 'random':
                path_table = random_n_sample(spark, path_table, complex_path_schema[0]["head_node_columns"], path_sample_count)
            elif path_sample_type == 'threshold':
                path_table = threshold_n_sample(spark, path_table, complex_path_schema[0]["head_node_columns"], path_sample_count)

        path_table.persist()
        df_to_unpersist.append(path_table)
        
    # Drop Node Features 
    path_table = path_table.drop(*add_node_features)
    
#     # 计算最优分区数
#     best_partitions = estimate_partitions(complex_path_df, approximate_row_count = complex_path_count)
#     print("最优分区数:", best_partitions)
#     complex_path_df = complex_path_df.repartition(best_partitions, "Feature_Time", complex_path_config[0]['head_node_std_name'], 
#                                    complex_path_config[-1]['head_node_std_name'])
    
    if graph.partition_key != None:
        path_table.write.partitionBy(graph.partition_key).mode("overwrite").parquet(complex_path_result_data_dir)
        
        # 为每个分区创建成功标志
        for partition_value in partition_values:            
            hdfs_save_text_file(complex_path_result_data_dir + f"/{graph.partition_key}={partition_value}", '_SUCCESS')
    
    else:
        path_table.write.mode("overwrite").parquet(complex_path_result_data_dir)

    # Save Result
    hdfs_save_dict(complex_path_result_data_dir, "_Seq", seq_node_edges, logger = logger)

    # unpersist all the data
    for df in df_to_unpersist:
        df.unpersist()

    # Read the result
    complex_path, _ = read_complex_path_instances(spark, graph, complex_path_config, complex_path_result_data_dir, 
                                                  partition_values, logger = logger)
    
    return complex_path
