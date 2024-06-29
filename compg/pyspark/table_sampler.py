from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, udf, rand
from pyspark.sql.window import Window
import pyspark.sql.functions as F

def random_n_sample(spark_session, df, node_columns, max_count):
    """
    Samples up to 'max_count' random records for each group defined by 'node_columns' in a DataFrame.

    :param spark_session: SparkSession instance.
    :param df: DataFrame to sample from.
    :param node_columns: Columns defining groups for sampling.
    :param max_count: Maximum number of samples per group.
    :return: Sampled DataFrame.
    """
    window_spec = Window.partitionBy(*[df[c] for c in node_columns]).orderBy(rand())
    sampled_df = df.withColumn("row_number", F.row_number().over(window_spec)) \
                   .filter(col("row_number") <= max_count) \
                   .drop("row_number")
    return sampled_df

def top_n_sample(spark_session, df, node_columns, max_count, feature_columns):
    """
    Selects the top 'max_count' records for each group defined by 'node_columns', ordered by 'feature_columns'.

    :param spark_session: SparkSession instance.
    :param df: DataFrame to sample from.
    :param node_columns: Columns defining groups for sampling.
    :param max_count: Maximum number of samples per group.
    :param feature_columns: Columns used for ordering within each group.
    :return: Sampled DataFrame.
    """
    order_columns = [col(f).desc() for f in feature_columns] + [rand()]
    window_spec = Window.partitionBy(*[df[c] for c in node_columns]).orderBy(*order_columns)
    sampled_df = df.withColumn("row_number", F.row_number().over(window_spec)) \
                   .filter(col("row_number") <= max_count) \
                   .drop("row_number")
    return sampled_df

def threshold_n_sample(spark_session, df, node_columns, max_count):
    """
    Filters rows in a DataFrame by allowing only up to 'max_count' records per group defined by 'node_columns'.

    :param spark_session: SparkSession instance.
    :param df: DataFrame to filter.
    :param node_columns: Columns defining groups for threshold filtering.
    :param max_count: Maximum number of records allowed per group.
    :return: Filtered DataFrame.
    """
    # Convert DataFrame to RDD
    rdd = df.rdd
    # Create key-value pairs
    key_value_rdd = rdd.map(lambda row: (tuple(row[col] for col in node_columns), row))

    # Define update function for aggregateByKey
    def update_accumulator(accumulator, value):
        count, records = accumulator
        if count < max_count:
            return (count + 1, records + [value])
        else:
            return (count, records)

    # Define combine function for aggregateByKey
    def combine_accumulators(acc1, acc2):
        count1, records1 = acc1
        count2, records2 = acc2
        combined_records = records1 + records2
        if len(combined_records) > max_count:
            combined_records = combined_records[:max_count]
        return (count1 + count2, combined_records)

    # Aggregate data to limit records per group
    aggregated_rdd = key_value_rdd.aggregateByKey((0, []), update_accumulator, combine_accumulators)

    # Flatten the results and convert back to DataFrame
    result_df = spark_session.createDataFrame(aggregated_rdd.flatMap(lambda x: x[1][1]))

    return result_df
