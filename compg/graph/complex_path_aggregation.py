from compg.python import time_costing, ensure_logger
from compg.hdfs import create_hdfs_directory, check_hdfs_file_exists, hdfs_read_dict, hdfs_save_dict, hdfs_save_text_file
from compg.graph import read_node_table, read_edge_table
from compg.graph import read_agg_complex_path_elem, read_agg_complex_path
from compg.graph import complex_path_sampling
from compg.pyspark import aggregate_features, vectorize_and_scale, rename_columns

@time_costing
@ensure_logger
def complex_paths_aggregation(spark, graph, complex_paths_config, sample_result_data_dir, agg_result_data_dir, agg_funcs = ["AVG"], target_nodes = None, 
                              partition_values = None, return_edge_feats = True, return_node_feats = True, vectorize_feats = False, scale_funcs = ["raw"],
                              logger = None):
    complex_paths = []
    
    for complex_path_config in complex_paths_config:
        complex_path = complex_path_aggregation(spark, graph, complex_path_config, sample_result_data_dir, agg_result_data_dir, agg_funcs, target_nodes, 
                                                 partition_values, return_edge_feats, return_node_feats, vectorize_feats, scale_funcs, logger = logger)

        complex_paths.append(complex_path)
    
    return complex_paths

@time_costing
@ensure_logger
def complex_path_aggregation(spark, graph, complex_path_config, sample_result_data_dir, agg_result_data_dir, agg_funcs = ["AVG"], target_nodes = None, 
                             partition_values = None, return_edge_feats = True, return_node_feats = True, vectorize_feats = False, scale_funcs = ["raw"],
                             logger = None):

    # Complex path Sampling
    complex_path = complex_path_sampling(spark, graph, complex_path_config, sample_result_data_dir, target_nodes, partition_values, 
                                         return_edge_feats, return_node_feats, vectorize_feats, logger = logger)

    complex_path_name = complex_path_config["path_name"]
    
    complex_path_agg_result_data_dir = graph.output_dir + f'/{agg_result_data_dir}/{complex_path_name}'
    
    logger.info(f"Aggregate features for complex-path: {complex_path_name}")
    logger.info(f"The result will be output to: {complex_path_agg_result_data_dir}")

    # 先保存下对应的seq信息
    hdfs_save_dict(complex_path_agg_result_data_dir, "_Seq", complex_path["seq_node_edges"], logger = logger)

    # 尝试直接读取结果(以后加上对partition_values的优化)
    agg_complex_path_elems = read_agg_complex_path(spark, graph, complex_path_agg_result_data_dir, partition_values, logger = logger)
    if agg_complex_path_elems is not None:
        logger.info(f"The result already exists")
        complex_path["agg_path"] = agg_complex_path_elems
        return complex_path

    # 记录运算结果
    agg_complex_path_elems = {}
    
    # 获得起点对应的key
    start_nodes_count = len(complex_path_config["path_schema"][0]["head_node_types"])
    
    agg_key_columns = []
    for element in complex_path["seq_node_edges"][:start_nodes_count]:
        elem_node_type = element[1]
        elem_node_index = element[2]
        elem_node_column = f"{elem_node_type}_{elem_node_index}"
        element_name = f"{element[0]}___{element[1]}___{element[2]}"

        # 记录起始节点名称
        agg_key_columns.append(elem_node_column)

        # 查看是否有对应特征
        if graph.graph_summary["feat_count"][element[1]] == 0:
            logger.info(f"Skip element:{element[0]}, {element[1]}, {element[2]}")
            continue
            
        # 提取对应的节点特征
        element_agg_result_data_dir = complex_path_agg_result_data_dir + f'/{element_name}'
        logger.info(f"Process element:{element[0]}, {element[1]}, {element[2]}")

        # 查看是否已有该元素的结果
        elem_df = read_agg_complex_path_elem(spark, graph, element_agg_result_data_dir, partition_values, logger = logger)
        if elem_df is not None:
            continue
        
        # 获得该节点对应的key
        elem_key_columns = [elem_node_column]
        if graph.partition_key != None:
            elem_key_columns.append(graph.partition_key)
        
        # 获得对应的目标节点列
        elem_node_key_df = complex_path["path_table"].select(elem_key_columns).distinct().persist()
        elem_df = elem_node_key_df.select("*")
        
        # 记录该节点对应的特征列
        elem_feats = []
        
        # 补全特征
        for elem_node_table_name in graph.graph_summary["nodes"][elem_node_type]:
            node_table_schema = {}
            node_table_schema['node_type'] = elem_node_type
            node_table_schema['node_table_name'] = elem_node_table_name
            node_table_schema['node_column'] = list(graph.graph_summary["nodes"][elem_node_type][elem_node_table_name]["node_col_to_types"].keys())[0]
            node_table_schema['feature_columns'] = graph.graph_summary["nodes"][elem_node_type][elem_node_table_name]["feat_cols"]

            if len(node_table_schema['feature_columns']) == 0:
                logger.info(f"Skip node table:{elem_node_table_name}")
                continue
                
            tgt_node_df = elem_node_key_df.withColumnRenamed(elem_node_column, node_table_schema['node_column'])
            
            # 读取对应的node table
            node_table = read_node_table(spark, graph, node_table_schema, tgt_node_df, partition_values, logger = logger)

            # 修正列名
            node_rename_columns = {}
            node_rename_columns[node_table_schema['node_column']] = elem_node_column
            for feat_col in node_table_schema['feature_columns']:
                new_feat_col = f"Node_{elem_node_type}_{elem_node_table_name}_{feat_col}"
                node_rename_columns[feat_col] = new_feat_col
                elem_feats.append(new_feat_col)

            node_table = rename_columns(spark, node_table, node_rename_columns)
            
            # 将结果合并到path table
            elem_df = elem_df.join(node_table, elem_key_columns, "left")

            # 没有的特征设为0
            elem_df = elem_df.fillna(0)
        
        # 向量化
        vectorized_feat_df = vectorize_and_scale(elem_df, scale_funcs, elem_key_columns, elem_feats)
        
        # 保存 
        if graph.partition_key != None:
            vectorized_feat_df.write.partitionBy(graph.partition_key).mode("overwrite").parquet(element_agg_result_data_dir)
            
            # 为每个分区创建成功标志
            for partition_value in partition_values:            
                hdfs_save_text_file(element_agg_result_data_dir + f"/{graph.partition_key}={partition_value}", '_SUCCESS')
        
        else:
            vectorized_feat_df.write.mode("overwrite").parquet(element_agg_result_data_dir)
        
        elem_node_key_df.unpersist()
    
    if graph.partition_key != None:
        agg_key_columns.append(graph.partition_key)
        if partition_values is None:
            partition_values = graph.graph_summary["partition_values"]

    # 依次处理要聚合的节点
    for element in complex_path["seq_node_edges"][start_nodes_count:]:
        element_name = f"{element[0]}___{element[1]}___{element[2]}"

        element_agg_result_data_dir = complex_path_agg_result_data_dir + f'/{element_name}'

        logger.info(f"Process element:{element[0]}, {element[1]}, {element[2]}")

        # 查看是否有对应的特征
        if graph.graph_summary["feat_count"][element[1]] == 0:
            logger.info(f"Skip element:{element[0]}, {element[1]}, {element[2]}")
            continue
            
        # 查看是否已有该元素的结果
        elem_df = read_agg_complex_path_elem(spark, graph, element_agg_result_data_dir, partition_values, logger = logger)
        if elem_df is not None:
            continue
        
        # 提取特征
        if element[0] == "Node":
            elem_node_type = element[1]
            elem_node_index = element[2]
            elem_node_column = f"{element[1]}_{element[2]}"

            # 目标路径表
            elem_df = complex_path["path_table"].select(agg_key_columns + [elem_node_column])

            # 获得特征来源节点对应的key
            elem_key_columns = [elem_node_column]
            if graph.partition_key != None:
                elem_key_columns.append(graph.partition_key)
            
            # 获得对应的目标节点列
            elem_node_key_df = complex_path["path_table"].select(elem_key_columns).distinct().persist()

            # 记录要聚合的特征列名
            elem_feats = []
            
            # 补全特征
            for elem_node_table_name in graph.graph_summary["nodes"][elem_node_type]:
                node_table_schema = {}
                node_table_schema['node_type'] = elem_node_type
                node_table_schema['node_table_name'] = elem_node_table_name
                node_table_schema['node_column'] = list(graph.graph_summary["nodes"][elem_node_type][elem_node_table_name]["node_col_to_types"].keys())[0]
                node_table_schema['feature_columns'] = graph.graph_summary["nodes"][elem_node_type][elem_node_table_name]["feat_cols"]

                if len(node_table_schema['feature_columns']) == 0:
                    logger.info(f"Skip node table:{elem_node_table_name}")
                    continue
                    
                tgt_node_df = elem_node_key_df.withColumnRenamed(elem_node_column, node_table_schema['node_column'])
                
                # 读取对应的node table
                node_table = read_node_table(spark, graph, node_table_schema, tgt_node_df, partition_values, logger = logger)

                # 修正列名
                node_rename_columns = {}
                node_rename_columns[node_table_schema['node_column']] = elem_node_column
                for feat_col in node_table_schema['feature_columns']:
                    new_feat_col = f"Node_{elem_node_type}_{elem_node_table_name}_{feat_col}"
                    node_rename_columns[feat_col] = new_feat_col
                    elem_feats.append(new_feat_col)

                node_table = rename_columns(spark, node_table, node_rename_columns)
                
                # 将结果合并到path table
                elem_df = elem_df.join(node_table, elem_key_columns, "left")

                # 没有的特征设为0
                elem_df = elem_df.fillna(0)
        
        elif element[0] == "Edge":
            elem_edge_type = element[1]
            elem_edge_index = element[2]

            # 获得对应特征列
            elem_feats = [col for col in complex_path["path_table"].columns if col.startswith(f"Edge_{elem_edge_index}_")]

            # 取出对应数据
            elem_df = complex_path["path_table"].select(agg_key_columns + elem_feats)
            
        # 聚合 
        aggregated_feat_df, aggregated_feat_cols = aggregate_features(spark, elem_df, agg_key_columns, elem_feats, agg_funcs)

        # 向量化
        vectorized_feat_df = vectorize_and_scale(aggregated_feat_df, scale_funcs, agg_key_columns, aggregated_feat_cols)
        
        # 保存 
        if graph.partition_key != None:
            vectorized_feat_df.write.partitionBy(graph.partition_key).mode("overwrite").parquet(element_agg_result_data_dir)
            
            # 为每个分区创建成功标志
            for partition_value in partition_values:            
                hdfs_save_text_file(element_agg_result_data_dir + f"/{graph.partition_key}={partition_value}", '_SUCCESS')
        
        else:
            vectorized_feat_df.write.mode("overwrite").parquet(element_agg_result_data_dir)
        
        # vectorized_feat_df.show()

        # 释放文件
        if element[0] == "Node":
            elem_node_key_df.unpersist()

    # 读取最终结果
    agg_complex_path_elems = read_agg_complex_path(spark, graph, complex_path_agg_result_data_dir, partition_values, logger = logger)
    complex_path["agg_path"] = agg_complex_path_elems
    
    return complex_path