from compg.python import setup_logger, time_costing, ensure_logger
from compg.pyspark import identify_numeric_columns
from compg.hdfs import create_hdfs_directory, check_hdfs_file_exists
from compg.hdfs import hdfs_read_dict
from compg.pyspark import random_n_sample, top_n_sample, threshold_n_sample
from pyspark.sql.functions import broadcast, col

def read_edge_table(spark, graph, edge_table_schema, target_nodes = None, partition_values = None, return_edge_feats = True, vectorize_feats = False, logger = None):
    """
    读取关系表，并过滤出有效边
    """
    edge_type = edge_table_schema['edge_table_name']
    
    # 获取该表对应的路径
    edge_table_path = graph.edge_tables_dir + '/' + edge_type

    logger.info(f'Read edge table: {edge_type} from directory: {edge_table_path}')

    # 要保留的列名 头尾节点列、数值列和分区列
    useful_columns = []
    useful_columns.extend(edge_table_schema["head_node_columns"])
    useful_columns.extend(graph.graph_summary["edges"][edge_type]["feat_cols"])
    useful_columns.extend(edge_table_schema["tail_node_columns"])

    # 头结点列
    head_key_columns = list(edge_table_schema["head_node_columns"])
    
    if graph.partition_key != None:
        useful_columns.append(graph.partition_key)
        head_key_columns.append(graph.partition_key)
        
        # 读取数据
        if partition_values is None:
            partition_values = graph.graph_summary["partition_values"]
            
        edge_table_paths = [edge_table_path + f"/{graph.partition_key}={partition_value}" for partition_value in partition_values]
        edge_table_df = spark.read.option("basePath", edge_table_path).parquet(*edge_table_paths)
    else:
        edge_table_df = spark.read.parquet(edge_table_path)
    
    # 保证各个列都没有null值
    for useful_column in useful_columns:
        edge_table_df = edge_table_df.filter(col(useful_column).isNotNull())

    # Only keep edges start from the start node, if provided(一定是head nodes)
    if target_nodes is not None:
        edge_table_df = edge_table_df.join(target_nodes, head_key_columns, 'inner')
        edge_table_df = edge_table_df.repartition(*head_key_columns)
        
    # Edge limit
    if 'edge_limit' in edge_table_schema and edge_table_schema['edge_limit'] != '':
        edge_table_df = edge_table_df.where(edge_table_schema['edge_limit'])
        logger.info('Edge Limitation:', edge_table_schema['edge_limit'])

    # Drop edges with the same head and tail nodes
    if graph.partition_key != None:
        edge_table_df = edge_table_df.dropDuplicates(edge_table_schema["head_node_columns"] + edge_table_schema["tail_node_columns"] + [graph.partition_key])
    else:
        edge_table_df = edge_table_df.dropDuplicates(edge_table_schema["head_node_columns"] + edge_table_schema["tail_node_columns"])

    # Only keep useful columns
    edge_table_df = edge_table_df.select(useful_columns)
    
    # Edge Sample
    if 'edge_sample' in edge_table_schema:
        edge_sample_type = edge_table_schema['edge_sample']['type']
        edge_sample_count = edge_table_schema['edge_sample']['count']
        
        logger.info(f'Edge Sampling: {edge_sample_type}, {edge_sample_count}')

        if edge_sample_type == 'random':
            edge_table_df = random_n_sample(spark, edge_table_df, edge_table_schema["head_node_columns"], edge_sample_count)
        elif edge_sample_type == 'threshold':
            edge_table_df = threshold_n_sample(spark, edge_table_df, edge_table_schema["head_node_columns"], 
                                               edge_sample_count)
    
    return edge_table_df

def read_node_table(spark, graph, node_table_schema, target_nodes = None, partition_values = None, return_node_feats = True, vectorize_feats = False, logger = None):
    """
    读取节点表中的目标信息
    """
    node_type = node_table_schema['node_type']
    
    # 获取该表对应的路径
    node_table_path = graph.node_tables_dir + '/' + node_type + '/' + node_table_schema["node_table_name"]

    logger.info(f'Read node table: {node_table_schema["node_table_name"]} from directory: {node_table_path}')

    # 要保留的列名 头尾节点列、数值列和分区列
    key_columns = []
    key_columns.append(node_table_schema["node_column"])
    if graph.partition_key != None:
        key_columns.append(graph.partition_key)

    useful_columns = list(key_columns)
    if "feature_columns" in node_table_schema:
        useful_columns.extend(node_table_schema["feature_columns"])
    else:
        useful_columns.extend(graph.graph_summary["nodes"][node_type][node_table_schema["node_table_name"]]["feat_cols"])
    
    # 读取数据
    if graph.partition_key != None:
        if partition_values is None:
            partition_values = graph.graph_summary["partition_values"]
        
        node_table_paths = [node_table_path + f"/{graph.partition_key}={partition_value}" for partition_value in partition_values]
        node_table_df = spark.read.option("basePath", node_table_path).parquet(*node_table_paths)
    else:
        node_table_df = spark.read.parquet(node_table_path)
    
    # 保证各个列都没有null值
    for useful_column in useful_columns:
        node_table_df = node_table_df.filter(col(useful_column).isNotNull())

    # Only keep target nodes
    if target_nodes is not None:
        node_table_df = node_table_df.join(target_nodes, key_columns, "inner")
    
    # Node limit
    if 'node_limit' in node_table_schema and node_table_schema['node_limit'] != '':
        node_table_df = node_table_df.where(node_table_schema['node_limit'])
        logger.info(f'Node Limitation: {node_table_schema['node_limit']}')

    # Drop edges with the same head and tail nodes
    if graph.partition_key != None:
        node_table_df = node_table_df.dropDuplicates([node_table_schema["node_column"], graph.partition_key])
    else:
        node_table_df = node_table_df.dropDuplicates([node_table_schema["node_column"]])

    # Only keep useful columns
    node_table_df = node_table_df.select(useful_columns)
    
    return node_table_df

def read_node_tables():
    
    return

# 读取标点表
def read_label_table():
    return

# 如果还没完成就返回未完成的区间
# 返回一位结果数据，一位缺失的分区
def read_complex_path_instances(spark, graph, complex_path_config, complex_path_dir, target_nodes = None, partition_values = None, logger = None):
    # 尝试读取结果
    if graph.partition_key != None:
        if partition_values is None:
            partition_values = graph.graph_summary["partition_values"]
            
        # Check existing results
        existing_partition_values = []
        for partition_value in partition_values:            
            if check_hdfs_file_exists(complex_path_dir + f"/{graph.partition_key}={partition_value}/_SUCCESS"): 
                existing_partition_values.append(partition_value)
                
        if len(partition_values) > len(existing_partition_values):
            partition_values = [x for x in partition_values if x not in existing_partition_values]
            logger.info(f"There are unfinished partition values for this complex-path: {partition_values}")

            return None, partition_values
            
        logger.info("The results for all the partition values of this complex-path already exist")

        path_table_paths = [complex_path_dir + f"/{graph.partition_key}={partition_value}" for partition_value in partition_values]
        path_table = spark.read.option("basePath", complex_path_dir).parquet(*path_table_paths)
        
    else:
        if not check_hdfs_file_exists(complex_path_dir + f"/_SUCCESS"): 
            return None, None
            
        logger.info("The results of this complex-path already exist")

        path_table = spark.read.parquet(complex_path_dir)
    
    seq_node_edges = hdfs_read_dict(complex_path_dir, "_Seq", logger = logger)

    # 获得起始点列名list
    start_nodes_count = len(complex_path_config["path_schema"][0]["head_node_types"])
    target_node_key_columns = [f"{e[1]}_{e[2]}" for e in seq_node_edges[:start_nodes_count]]
    if graph.partition_key != None:
        target_node_key_columns.append(graph.partition_key)
    
    # 基于target_node采样
    if target_nodes is not None:
        path_table = path_table.join(target_nodes, on = target_node_key_columns, how = "inner")
    
    complex_path = {}
    complex_path["path_table"] = path_table
    complex_path["seq_node_edges"] = seq_node_edges
        
    return complex_path, None

def read_agg_complex_path_elem(spark, graph, agg_complex_path_elem_dir, target_nodes = None, partition_values = None, logger = None):
    # 检查对应数据是否已存在
    if graph.partition_key != None:
        if partition_values is None:
            partition_values = graph.graph_summary["partition_values"]
            
        # Check existing results
        existing_partition_values = []
        for partition_value in partition_values:            
            if check_hdfs_file_exists(agg_complex_path_elem_dir + f"/{graph.partition_key}={partition_value}/_SUCCESS"): 
                existing_partition_values.append(partition_value)

        if len(partition_values) > len(existing_partition_values):
            return None
            
        logger.info("The results for all the partition values of this complex-path element already exist")

        elem_df_dirs = [agg_complex_path_elem_dir + f"/{graph.partition_key}={partition_value}" for partition_value in partition_values]
        elem_df = spark.read.option("basePath", agg_complex_path_elem_dir).parquet(*elem_df_dirs)
        
    else:
        if not check_hdfs_file_exists(agg_complex_path_elem_dir + f"/_SUCCESS"):
            return None
            
        logger.info("The results for this complex-path element already exist")

        elem_df = spark.read.parquet(agg_complex_path_elem_dir)
        
    # 基于目标节点过滤
            
    return elem_df

# 数据都存在，则返回结果，否则返回None
def read_agg_complex_path(spark, graph, agg_complex_path_dir, target_nodes = None, partition_values = None, logger = None):
    # 先读取包含的各个element的顺序
    seq_node_edges = hdfs_read_dict(agg_complex_path_dir, "_Seq", logger = logger)

    # 补全partition_values
    if graph.partition_key != None and partition_values is None:
        partition_values = graph.graph_summary["partition_values"]
    
    # 存储路径上各位对应的结果
    agg_complex_path_elems = {}
    
    # 依次获取路径上其他元素对应的聚合后的特征
    for element in seq_node_edges:
        element_name = f"{element[0]}___{element[1]}___{element[2]}"
        
        # 查看对应元素是否有特征
        if graph.graph_summary["feat_count"][element[1]] == 0:
            logger.info(f"Skip element due to no features: {element[0]}, {element[1]}, {element[2]}")
            continue
        
        logger.info(f"Read element: {element[0]}, {element[1]}, {element[2]}")
        
        agg_complex_path_elem_dir = agg_complex_path_dir + f'/{element_name}'
        
        elem_df = read_agg_complex_path_elem(spark, graph, agg_complex_path_elem_dir, target_nodes, partition_values, logger = logger)

        if elem_df is None:
            return None
            
        agg_complex_path_elems[element_name] = elem_df
        
    return agg_complex_path_elems