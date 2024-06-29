#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datetime import datetime
from compg.graph import ComplexGraph
from compg.python import mkdir, setup_logger, time_costing, read_json_file


# In[2]:


# 日志信息保存文件名
log_files_dir = '../../../result_data/log_files/CompAgg'
log_filename = log_files_dir + f'/{datetime.now().strftime("%Y-%m-%d-%H:%M")}.log'
mkdir(log_files_dir)

logger = setup_logger(log_filename, logger_name = "main")

# Complex paths config
# complex_graph_config_file = '../../config/Complex_Graphs/Ogbn_Mag_V1.json'
complex_graph_config_file = '../../config/Complex_Graphs/Mag240m_V1.json'
complex_graph_config = read_json_file(complex_graph_config_file)


# In[3]:


# 调整节点表配置
update_node_config = {}
update_node_config["raw_node_type"] = "Paper"
update_node_config["updated_node_type"] = "Paper"
update_node_config["kept_node_tables"] = ['Paper_Area', 'Paper_Year']

subgraph = {}
subgraph["update_nodes"] = [update_node_config]
complex_graph_config["subgraph"] = subgraph

graph = ComplexGraph(complex_graph_config, logger = logger)


# In[4]:


complex_paths_config_file = '../../config/Complex_Paths/Mag240m_Paths_Example_V1.json'
# complex_paths_config_file = '../../config/Complex_Paths/Paths_Example_V1.json'
complex_paths_config = read_json_file(complex_paths_config_file)


# In[5]:


from compg.pyspark import start_spark_session

# spark = start_spark_session(mode='local', logger = logger)

config_dict = {
                "spark.default.parallelism": "100",
                "spark.sql.shuffle.partitions": "200",
                "spark.sql.broadcastTimeout": "3600",
                "spark.driver.memory": "20g",
                "spark.driver.cores": "4",
                "spark.driver.maxResultSize": "0",
                "spark.executor.memory": "40g",
                "spark.executor.cores": "4",
                "spark.executor.instances": "25",
            }
spark = start_spark_session(mode='cluster', config_dict = config_dict, logger = logger)

# spark = start_spark_session(mode='cluster', logger = logger)

# Spark_Runner = ResilientSparkRunner()
# Spark_Runner.run(Complex_Paths_Aggregation)


# In[6]:


# 读取标签节点
# label_node_df = spark.read.parquet("/user/mart_coo/mart_coo_innov/CompGraph/Ogbn_Mag/Graph/Label_Tables/Paper_Venue")
label_node_df = spark.read.parquet("/user/mart_coo/mart_coo_innov/CompGraph/Mag240m/Graph/Label_Tables/Paper_Area")

target_node_column = "Paper_1"

label_node_df = label_node_df.withColumnRenamed("paper", target_node_column)

target_node_df = label_node_df.select(target_node_column).distinct().persist()


# In[7]:


from compg.graph import complex_paths_aggregation
complex_paths = complex_paths_aggregation(spark, graph, complex_paths_config, "Mag240m/Complex_Paths_V2", "Mag240m/Agg_Complex_Paths_Labels_V2", 
                                          target_nodes = target_node_df, logger = logger)


# In[8]:


from pyspark.sql.functions import udf
from pyspark.sql.types import BooleanType
import numpy as np

# 先获得对应的标签
covered_paper_label_df = complex_paths[2]["agg_path"]["Node___Paper___1"].withColumnRenamed("features_raw", "labels")

# 获得各路径的结果
same_author_paper_df = complex_paths[0]["agg_path"]["Node___Paper___3"].withColumnRenamed("features_raw", "same_author_paper_avg_labels")
covered_paper_label_df = covered_paper_label_df.join(same_author_paper_df, on = "Paper_1", how = "inner")

cited_paper_df = complex_paths[1]["agg_path"]["Node___Paper___2"].withColumnRenamed("features_raw", "cited_paper_avg_labels")
covered_paper_label_df = covered_paper_label_df.join(cited_paper_df, on = "Paper_1", how = "left")

cited_same_author_paper_df = complex_paths[2]["agg_path"]["Node___Paper___3"].withColumnRenamed("features_raw", "cited_same_author_paper_avg_labels")
covered_paper_label_df = covered_paper_label_df.join(cited_same_author_paper_df, on = "Paper_1", how = "left")

covered_paper_label_pd = covered_paper_label_df.toPandas()


# In[12]:


true_labels = covered_paper_label_pd['labels'].apply(lambda x: np.argmax(x))


# In[18]:


from sklearn.metrics import accuracy_score

predicted_labels = covered_paper_label_pd["cited_paper_avg_labels"].apply(lambda x: np.argmax(x))

accuracy_score(true_labels, predicted_labels)


# In[17]:


predicted_labels = covered_paper_label_pd["same_author_paper_avg_labels"].apply(lambda x: np.argmax(x))

accuracy_score(true_labels, predicted_labels)


# In[16]:


predicted_labels = covered_paper_label_pd["cited_same_author_paper_avg_labels"].apply(lambda x: np.argmax(x))

accuracy_score(true_labels, predicted_labels)

