import copy
from compg.python import time_costing, ensure_logger
from compg.hdfs import hdfs_list_contents, hdfs_read_dict

# HeteroGraphBuilder HeteroGraphManager
# 初始化图工作空间（存储图对应的相关路径配置）
class ComplexGraph():
    @time_costing
    @ensure_logger
    def __init__(self, graph_config, logger = None):
        self.data_dir = graph_config["data_dir"]
        
        self.graph_dir = self.data_dir + graph_config["graph_dir"]
        self.node_tables_dir = self.graph_dir + graph_config["node_tables_dir"]
        self.edge_tables_dir = self.graph_dir + graph_config["edge_tables_dir"]
        
        self.output_dir = self.data_dir + graph_config["output_dir"]

        if "partition_key" in graph_config:
            self.partition_key = graph_config["partition_key"]
        else:
            self.partition_key = None
            
        self.logger = logger

        subgraph_config = graph_config["subgraph"] if "subgraph" in graph_config else None
        self.graph_summary = self.get_graph_summary(subgraph_config)
        
        self.show_brief_summary()

        # 加入对节点表和边表的检查代码(是否有重复数据、无效数据等)
    
    def get_graph_summary(self, subgraph_config = None):
        graph_summary = {}
        
        # 获得各节点表对应的信息
        graph_summary["nodes"] = self.node_tables_summary()
        
        # 获得各边表对应的信息
        graph_summary["edges"] = self.edge_tables_summary()

        # 检查边表中是否出现了新的类型的节点，有的话，添加到nodes中
        for edge_type in graph_summary["edges"]:
            # 检查该边表中包含的节点类型
            for node_type in graph_summary["edges"][edge_type]["node_col_to_types"].values():
                if node_type not in graph_summary["nodes"]:
                    graph_summary["nodes"][node_type] = {}

        # 查看是否有采样子图的要求
        if subgraph_config is not None:
            graph_summary = self.subgraph_selection(graph_summary, subgraph_config)

        # 统计各个节点和边类型对应的特征数量 
        graph_summary["feat_count"] = {}
        for node_type in graph_summary["nodes"]:
            graph_summary["feat_count"][node_type] = 0
            for node_table_name in graph_summary["nodes"][node_type]:
                graph_summary["feat_count"][node_type] = graph_summary["feat_count"][node_type] + len(graph_summary["nodes"][node_type][node_table_name]["feat_cols"])

        for edge_type in graph_summary["edges"]:
            graph_summary["feat_count"][edge_type] = len(graph_summary["edges"][edge_type]["feat_cols"])
        
        # 将所有表都包含的时间分区作为这个graph所拥有的时间分区
        if self.partition_key != None:
            graph_summary["partition_values"] = None
            for node_type in graph_summary["nodes"]:
                for node_table_name in graph_summary["nodes"][node_type]:
                    if graph_summary["partition_values"] is None:
                        graph_summary["partition_values"] = graph_summary["nodes"][node_type][node_table_name]["partition_values"]
                    else:
                        graph_summary["partition_values"] = list(set(graph_summary["partition_values"]) & set(graph_summary["nodes"][node_type][node_table_name]["partition_values"]))
    
            for edge_type in graph_summary["edges"]:
                if graph_summary["partition_values"] is None:
                    graph_summary["partition_values"] = graph_summary["edges"][edge_type]["partition_values"]
                else:
                    graph_summary["partition_values"] = list(set(graph_summary["partition_values"]) & set(graph_summary["edges"][edge_type]["partition_values"]))
                    
            graph_summary["partition_values"].sort()
        
        return graph_summary

    def subgraph_selection(self, graph_summary, subgraph_config):
        # 只保留指定的节点表
        if "update_nodes" in subgraph_config:
            for update_node_config in subgraph_config["update_nodes"]:
                raw_node_type = update_node_config["raw_node_type"]
                updated_node_type = update_node_config["updated_node_type"]

                updated_node_config = copy.deepcopy(graph_summary["nodes"][raw_node_type])
                updated_node_config = {node_table_name: updated_node_config[node_table_name] for node_table_name in update_node_config["kept_node_tables"]} 

                graph_summary["nodes"][updated_node_type] = updated_node_config
        
        # 只保留边表中指定的节点列
        if "update_edges" in subgraph_config:
            for update_edge_config in subgraph_config["update_edges"]:
                raw_edge_type = update_edge_config["raw_edge_type"]
                updated_edge_type = update_edge_config["updated_edge_type"]

                updated_edge_config = copy.deepcopy(graph_summary["edges"][raw_edge_type])
                updated_edge_config["node_col_to_types"] = update_edge_config["node_col_to_types"]

                graph_summary["edges"][updated_edge_type] = updated_edge_config

        # 只保留指定的节点类型
        if "kept_nodes" in subgraph_config:
            # 只保留对应的节点涉及到的节点表
            graph_summary["nodes"] = {node_type: graph_summary["nodes"][node_type] for node_type in subgraph_config["kept_nodes"]} 
            
            # 只保留边表中相关的节点列
            for edge_type in graph_summary["edges"]:
                raw_node_col_to_types = graph_summary["edges"][edge_type]["node_col_to_types"]
                kept_node_col_to_types = {key: value for key, value in raw_node_col_to_types.items() if value in subgraph_config["kept_nodes"]}
                graph_summary["edges"][edge_type]["node_col_to_types"] = kept_node_col_to_types

        # 只保留指定的边表
        if "kept_edges" in subgraph_config:
            graph_summary["edges"] = {edge_type: graph_summary["edges"][edge_type] for edge_type in subgraph_config["kept_edges"]} 
        
        return graph_summary
    
    def node_tables_summary(self):
        nodes_summary = {}
        
        node_type_dirs = hdfs_list_contents(self.node_tables_dir, content_type = "directories")
        
        for node_type_dir in node_type_dirs:
            # 记录该节点类型
            node_type = node_type_dir.split('/')[-1]

            nodes_summary[node_type] = {}
            
            # 各种节点有多少个表
            node_table_dirs = hdfs_list_contents(node_type_dir, content_type = "directories")

            for node_table_dir in node_table_dirs:
                # 记录表名
                node_table_name = node_table_dir.split('/')[-1]
                
                # 读取配置信息
                node_table_config = hdfs_read_dict(node_table_dir, "_Config", logger = self.logger)

                # 如果有分区键，那记录全部的分区值
                if self.partition_key != None:
                    node_table_partition_values = [x.split(f'/{self.partition_key}=')[-1] for x in hdfs_list_contents(node_table_dir, content_type = "directories")]
                    node_table_config["partition_values"] = node_table_partition_values
                
                # 记录配置信息
                nodes_summary[node_type][node_table_name] = node_table_config
        
        return nodes_summary
    
    def edge_tables_summary(self):
        edges_summary = {}
        
        # 多少种边
        edge_table_dirs = hdfs_list_contents(self.edge_tables_dir, content_type = "directories")

        for edge_table_dir in edge_table_dirs:
            # 获得表名
            edge_table_name = edge_table_dir.split('/')[-1]

            # 将表名作为边的类型名
            edge_type = edge_table_name
            
            # 读取配置信息
            edge_table_config = hdfs_read_dict(edge_table_dir, "_Config", logger = self.logger)

            # 记录表名
            edge_table_config["edge_table_name"] = edge_table_name
            
            # 如果有分区键，那记录全部的分区值
            if self.partition_key != None:
                edge_table_partition_values = [x.split(f'/{self.partition_key}=')[-1] for x in hdfs_list_contents(edge_table_dir, content_type = "directories")]
                edge_table_config["partition_values"] = edge_table_partition_values
            
            # 记录配置信息
            edges_summary[edge_type] = edge_table_config
        
        return edges_summary
    
    # 显示图中的简单的统计信息
    def show_brief_summary(self):
        # 数据位置
        self.logger.info(f"Data directory: {self.data_dir}")
        self.logger.info(f"Graph directory: {self.graph_dir}")
        self.logger.info(f"Node tables directory: {self.node_tables_dir}")
        self.logger.info(f"Edge tables directory: {self.edge_tables_dir}")
        
        # 有几种类型的点，各种节点有几种类型的节点表各节点表有多少个特征
        self.logger.info(f"Graph contains {len(self.graph_summary['nodes'].keys())} types of nodes:")
        for node_type in self.graph_summary['nodes'].keys():
            self.logger.info(f"Node type {node_type} with {len(self.graph_summary['nodes'][node_type].keys())} node tables: {list(self.graph_summary['nodes'][node_type].keys())}")
            
        # 多少种边，每个节点有多少个特征表，分别有多少特征
        self.logger.info(f"Graph contains {len(self.graph_summary['edges'].keys())} types of edges:")
        for edge_type in self.graph_summary['edges'].keys():
            self.logger.info(f"Edge type {edge_type} with {len(self.graph_summary['edges'][edge_type]['node_col_to_types'].keys())} node columns: {self.graph_summary['edges'][edge_type]['node_col_to_types']}")

        # 各个节点和边分别有几个特征
        for type_name in self.graph_summary['feat_count'].keys():
            self.logger.info(f"Type {type_name} with {self.graph_summary['feat_count'][type_name]} features")
            
        # 有哪几个共享的分区
        if self.partition_key != None:
            self.logger.info(f"Graph partition values: {self.partition_key} in {self.graph_summary['partition_values']}")
        
        return
    