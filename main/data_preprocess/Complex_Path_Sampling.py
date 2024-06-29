#!/usr/bin/env python
# coding: utf-8

# In[1]:


from datetime import datetime
from compg.graph import ComplexGraph
from compg.python import mkdir, setup_logger, time_costing, read_json_file


# In[2]:


# 日志信息保存文件名
log_files_dir = '../../result_data/log_files/CompAgg'
log_filename = log_files_dir + f'/{datetime.now().strftime("%Y-%m-%d-%H:%M")}.log'
mkdir(log_files_dir)

logger = setup_logger(log_filename, logger_name = "main")

# Complex paths config
complex_graph_config_file = '../config/Complex_Graphs/Ogbn_Mag_V1.json'
complex_graph_config = read_json_file(complex_graph_config_file)


# In[3]:


graph = ComplexGraph(complex_graph_config, logger = logger)


# In[4]:


complex_paths_config_file = '../config/Complex_Paths/Paths_Example_V1.json'
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

# Spark_Runner = ResilientSparkRunner()
# Spark_Runner.run(Complex_Paths_Aggregation)


# In[6]:

from compg.graph import complex_paths_sampling
complex_paths = complex_paths_sampling(spark, graph, complex_paths_config, "Complex_Paths_Debug_V7", logger = logger)

