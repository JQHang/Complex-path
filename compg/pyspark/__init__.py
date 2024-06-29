import os
import sys

os.environ['SPARK_HOME']="/software/servers/10k/mart_scr/spark_3.0"
os.environ['PYTHONPATH']="/software/servers/10k/mart_scr/spark_3.0/python:/software/servers/10k/mart_scr/spark_3.0/python/lib/py4j-0.10.9-src.zip"
os.environ['LD_LIBRARY_PATH']="/software/servers/jdk1.8.0_121/lib:/software/servers/jdk1.8.0_121/jre/lib/amd64/server:/software/servers/hope/mart_sch/hadoop/lib/native"
os.environ['PYSPARK_PYTHON']="/usr/local/anaconda3/bin/python3.6"
os.environ['PYSPARK_DRIVER_PYTHON']="/usr/local/anaconda3/bin/python3.6"

sys.path.insert(0, '/software/servers/10k/mart_scr/spark_3.0/python/lib/py4j-0.10.9-src.zip')
sys.path.insert(0, '/software/servers/10k/mart_scr/spark_3.0/python')
sys.path.insert(0, '/software/servers/10k/mart_scr/spark_3.0/python/lib/pyspark.zip')

from .spark_runner import start_spark_session, ResilientSparkRunner
from .dataframe_manager import identify_numeric_columns, estimate_row_size, estimate_partitions, sanitize_column_names
from .dataframe_manager import check_uniform_columns, check_enum_column, pivot_dataframe, ensure_numeric_column, bin_and_count
from .dataframe_manager import rename_columns
from .aggregate import aggregate_features
from .vector import vectorize_and_scale, parse_string_to_vector, merge_vectors, left_join_vectors
from .table_sampler import random_n_sample, top_n_sample, threshold_n_sample