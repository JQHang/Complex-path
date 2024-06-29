from pyspark.sql import SparkSession
from pyspark.sql import DataFrame
from pyspark.sql.types import IntegerType, FloatType, DoubleType, DecimalType, LongType, ShortType, ByteType
import pyspark.sql.functions as F
from pyspark.sql.functions import col, when, count
import re
import math
from compg.python import setup_logger, time_costing, ensure_logger

def identify_numeric_columns(df_schema):
    """
    Identifies numeric columns in a DataFrame schema and returns their names.

    :param df_schema: The schema of the DataFrame.
    :return: A lists containing the names of numeric columns.
    """
    numeric_types = (IntegerType, FloatType, DoubleType, DecimalType, LongType, ShortType, ByteType)

    numeric_columns = []
    for field in df_schema.fields:
        if isinstance(field.dataType, numeric_types):
            numeric_columns.append(field.name)

    return numeric_columns

def estimate_row_size(df):
    """
    Estimates the size of a row based on the data type of columns in a DataFrame.

    :param df: A Spark DataFrame whose row size needs to be estimated.
    :return: The estimated size of one row in bytes.
    """
    size_estimates = {
        'IntegerType': 4,
        'LongType': 8,
        'FloatType': 4,
        'DoubleType': 8,
        'StringType': 20,  # Average estimate
    }

    row_size = sum(size_estimates.get(field.dataType.simpleString(), 16) for field in df.schema.fields)
    return row_size

@time_costing
def estimate_partitions(df, partition_target_size_mb=300, row_size_estimate=None, approximate_row_count=None, max_partitions=2000):
    """
    Estimates the number of partitions required for a DataFrame based on target partition size.

    :param df: DataFrame to estimate partitions for.
    :param partition_target_size_mb: Target size for each partition in megabytes.
    :param row_size_estimate: Pre-estimated row size in bytes (optional).
    :param approximate_row_count: Pre-estimated row count (optional).
    :return: The estimated number of partitions.
    """
    partition_target_size_bytes = partition_target_size_mb * 1024 * 1024

    if row_size_estimate is None:
        row_size_estimate = estimate_row_size(df)
        
    if approximate_row_count is None:
        approximate_row_count = df.rdd.countApprox(timeout=1000, confidence=0.95)

    estimated_total_size_bytes = approximate_row_count * row_size_estimate
    num_partitions = int(math.ceil(estimated_total_size_bytes / partition_target_size_bytes))
    
    if num_partitions > max_partitions:
        num_partitions = max_partitions 
    
    return num_partitions

@ensure_logger
def sanitize_column_names(df, logger = None):
    """
    Sanitizes DataFrame column names by replacing special characters with underscores.

    :param df: The original PySpark DataFrame.
    :return: DataFrame with sanitized column names.
    """  
    def replace_special_chars(col_name):
        pattern = r'[^\u4e00-\u9fa5a-zA-Z0-9_]'
        return re.sub(pattern, '_', col_name) if re.search(pattern, col_name) else col_name

    new_columns = [(col, replace_special_chars(col)) for col in df.columns]
    for old_col, new_col in new_columns:
        if old_col != new_col:
            logger.info(f'Renaming column: {old_col} to {new_col}')
            df = df.withColumnRenamed(old_col, new_col)
    return df

@time_costing
@ensure_logger
def check_uniform_columns(df, feature_columns=None, logger = None):
    """
    Identifies columns in a DataFrame where all values are identical.

    :param df: A Spark DataFrame to check.
    :param feature_columns: Optional list of column names to check. Checks all numeric columns if None.
    :return: None, prints out the results directly.
    """
    # Check if a valid logger is provided in keyword arguments
    if not isinstance(logger, logging.Logger):
        logger = setup_logger()
    
    if feature_columns is None:
        feature_columns, _ = check_numeric_columns(df.schema)
    feat_summary_df = df.select(feature_columns).summary("min", "max")
    min_values = feat_summary_df.filter(F.col("summary") == "min").drop("summary").collect()[0].asDict()
    max_values = feat_summary_df.filter(F.col("summary") == "max").drop("summary").collect()[0].asDict()
    invalid_cols_values = {col_name: min_values[col_name] for col_name in min_values if min_values[col_name] == max_values[col_name]}
    if invalid_cols_values:
        logger.error('ERROR: Some columns have identical values across all rows. Details:', invalid_cols_values)
    else:
        logger.info("All columns contain diverse values.")

@time_costing
@ensure_logger
def check_enum_column(df, column_name, top_n=30, logger=None):
    """
    Count the occurrence of each unique value in a specified column of a DataFrame,
    log the total number of distinct types of values, and log the top N most frequent values.
    Returns the top N values as a dictionary.

    Parameters:
        df (DataFrame): The DataFrame to analyze.
        column_name (str): The name of the column to analyze.
        top_n (int): The number of most frequent values to log.
        logger (logging.Logger): Optional logger for logging information.

    Returns:
        dict: Dictionary of the top N most frequent values and their counts.
    """
    # Calculate the count of each unique value in the column
    value_counts = df.groupBy(column_name).count()

    # Order by count descending to get the most frequent values
    most_common_values = value_counts.orderBy(col("count").desc())

    # Log the total number of distinct values
    distinct_count = most_common_values.count()
    logger.info(f"Total distinct types of values in column '{column_name}': {distinct_count}")

    # Collect the top N most frequent values
    top_values = most_common_values.limit(top_n).collect()

    # Convert top values to dictionary
    top_values_dict = {row[column_name]: row['count'] for row in top_values}

    # Log the top N most frequent values
    logger.info(f"Top {top_n} most frequent values and their counts:")
    for key, value in top_values_dict.items():
        logger.info(f"{key}: {value}")

    return top_values_dict

def pivot_dataframe(df, id_col, value_col, values_list):
    """
    Pivot a DataFrame based on a list of possible values in a specified column, count occurrences for each ID.

    Args:
        df (DataFrame): The input DataFrame.
        id_col (str): The name of the ID column.
        value_col (str): The name of the data column.
        values_list (list): List of all possible values to pivot on.

    Returns:
        DataFrame: A pivoted DataFrame with ID, counts of each possible value, and an 'else' column for other values.
    """  
    # Create a modified version of value_col, marking values not in values_list as 'Extra'
    modified_df = df.withColumn(value_col, when(col(value_col).isin(values_list), col(value_col)).otherwise('Extra'))

    # Check if 'Extra' is not in the list and append it if necessary
    if 'Extra' not in values_list:
        values_list.append('Extra')

    # Group by ID, pivot on the modified value column, count occurrences, and fill NA with 0
    pivot_df = modified_df.groupBy(id_col).pivot(value_col, values_list).count().fillna(0)

    # Rename columns to include the name of the data column as a prefix
    for value in values_list:
        pivot_df = pivot_df.withColumnRenamed(str(value), f"{value_col}_{value}")

    return pivot_df

@ensure_logger
def ensure_numeric_column(df, column_name, logger=None):
    """
    Checks if the column in the DataFrame is numeric, and converts it to float if not.

    Parameters:
        df (DataFrame): The input DataFrame.
        column_name (str): The column to check and convert if necessary.
        logger (logging.Logger): Optional logger for warnings.

    Returns:
        DataFrame: Modified DataFrame with the column converted if necessary.
    """
    # Check column data type
    column_type = df.schema[column_name].dataType

    # Convert column to FloatType if not a numeric type
    if not isinstance(column_type, (DoubleType, FloatType, IntegerType, DecimalType, LongType, ShortType, ByteType)):
        if logger:
            logger.warning(f"Column '{column_name}' is not a numeric type, attempting to convert to float.")
        df = df.withColumn(column_name, F.col(column_name).cast(FloatType()))
        df = df.na.fill({column_name: -1.0})
        converted_count = df.filter(F.col(column_name) == -1.0).count()
        if converted_count > 0 and logger:
            logger.warning(f"{converted_count} rows failed to convert and were set to -1.")

    return df

@time_costing
def bin_and_count(df, column_name, num_bins=20):
    """
    Bins the specified numeric column into a number of quantiles and returns data suitable for histogram plotting.

    Parameters:
        df (DataFrame): The input DataFrame.
        column_name (str): The numeric column to bin.
        num_bins (int): Number of bins to split the column into.

    Returns:
        list: List of tuples (bin_start, bin_end, count) representing each bin.
    """
    # Calculate approximate quantiles to determine bins
    quantiles = df.stat.approxQuantile(column_name, [i / num_bins for i in range(num_bins + 1)], 0.01)

    # Calculate counts per bin
    histogram_data = []
    for i in range(1, len(quantiles)):
        count = df.filter((F.col(column_name) >= quantiles[i-1]) & (F.col(column_name) < quantiles[i])).count()
        histogram_data.append((quantiles[i-1], quantiles[i], count))

    return histogram_data

def rename_columns(spark, df, columns_to_rename):
    """
    Rename columns of a Spark DataFrame using Spark SQL.

    Args:
    df (DataFrame): Input Spark DataFrame.
    columns_to_rename (Dict[str, str]): A dictionary where keys are old column names and values are new column names.

    Returns:
    DataFrame: A new DataFrame with renamed columns.

    """
    # Create a unique temporary view name for the input DataFrame
    temp_view_name = f"temp_view"
    df.createOrReplaceTempView(temp_view_name)

    # Prepare SQL query to rename columns
    select_expr = [
        f"`{old_name}` as `{columns_to_rename.get(old_name, old_name)}`" 
        for old_name in df.columns
    ]
    sql_query = f"SELECT {', '.join(select_expr)} FROM {temp_view_name}"

    # Execute the query
    new_df = spark.sql(sql_query)

    # Drop the temporary view
    spark.catalog.dropTempView(temp_view_name)

    return new_df