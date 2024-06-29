from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, udf, rand

def aggregate_features(spark, df, id_columns, feature_columns, aggregation_functions):
    """
    Performs aggregation on specified features of a DataFrame using provided aggregation functions.

    :param spark: SparkSession instance.
    :param df: Input DataFrame.
    :param id_columns: List of column names to group by.
    :param feature_columns: List of feature columns to aggregate.
    :param aggregation_functions: List of aggregation functions like AVG, SUM, MAX, MIN.
    :return: A dictionary containing the aggregated DataFrame, list of new feature columns and their comments.
    """
    df = df.fillna(0)
    df.createOrReplaceTempView("EMP")

    sql_agg_functions = []
    new_feature_columns = []

    for agg_function in aggregation_functions:
        if agg_function in ['AVG', 'SUM', 'MAX', 'MIN']:
            for feat_column in feature_columns:
                sql_function = f"{agg_function}({feat_column}) AS {agg_function}_{feat_column}"
                sql_agg_functions.append(sql_function)
                new_feature_columns.append(f"{agg_function}_{feat_column}")
        elif agg_function == 'COUNT':
            sql_agg_functions.append("COUNT(*) AS COUNT_All")
            new_feature_columns.append("COUNT_All")

    sql_query = f"SELECT {', '.join(id_columns)}, {', '.join(sql_agg_functions)} FROM EMP GROUP BY {', '.join(id_columns)}"
    aggregated_df = spark.sql(sql_query)

    return aggregated_df, new_feature_columns
