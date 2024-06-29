from pyspark.sql import DataFrame, SparkSession
from pyspark.sql.functions import col, udf, rand
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, MinMaxScaler, StandardScaler
from pyspark.ml.linalg import Vectors, VectorUDT, SparseVector, DenseVector
import re

def vectorize_and_scale(df, preprocess_types, stable_columns, preprocess_columns):
    """
    Preprocesses numerical data in a DataFrame by normalizing or standardizing the features.

    :param df: DataFrame to preprocess.
    :param preprocess_types: List containing types of preprocessing like 'Normalized' or 'Standard'.
    :param stable_columns: Columns to retain without changes.
    :param preprocess_columns: Columns to preprocess.
    :return: Processed DataFrame.
    """
    df = df.fillna(0)
    pipeline_stages = []

    assembler = VectorAssembler(inputCols=preprocess_columns, outputCol="features_raw")
    pipeline_stages.append(assembler)

    if 'norm' in preprocess_types:
        scaler = MinMaxScaler(inputCol="features_raw", outputCol="features_norm")
        pipeline_stages.append(scaler)

    if 'std' in preprocess_types:
        scaler = StandardScaler(inputCol="features_raw", outputCol="features_std")
        pipeline_stages.append(scaler)

    pipeline = Pipeline(stages=pipeline_stages)
    model = pipeline.fit(df)
    processed_df = model.transform(df)
    
    output_columns = stable_columns + [f"features_{type_}" for type_ in preprocess_types]
    processed_df = processed_df.select(output_columns)

    return processed_df

def parse_string_to_vector(s):
    """
    Parses a string representation of a vector into a PySpark Vector object.

    :param s: String representation of the vector.
    :return: Vector object (SparseVector or DenseVector).
    """
    if s.startswith("(") and "," in s and "[" in s:
        size, indices_str, values_str = re.match(r'\((\d+),\s*\[(.*?)\],\s*\[(.*?)\]\)', s).groups()
        indices = [int(x.strip()) for x in indices_str.split(",")]
        values = [float(x.strip()) for x in values_str.split(",")]
        return SparseVector(int(size), indices, values)
    elif s.startswith("["):
        values = [float(x.strip()) for x in re.findall(r'[-+]?\d*\.\d+|\d+', s)]
        return DenseVector(values)
    else:
        raise ValueError("Unknown vector format")

def merge_vectors(df, col1, col2):
    """
    Merges two vector columns into one in a DataFrame.

    :param df: DataFrame with vectors to merge.
    :param col1: Name of the first vector column.
    :param col2: Name of the second vector column.
    :return: DataFrame with merged vector column, drops the second vector column.
    """
    merge_udf = udf(lambda vec1, vec2: Vectors.dense(vec1.toArray().tolist() + vec2.toArray().tolist()), VectorUDT())

    df = df.withColumn(col1, merge_udf(col(col1), col(col2))).drop(col2)
    return df

def fill_null_vectors(spark_session, df, column_names, vector_length=None):
    """
    Fills null vector columns in a DataFrame with an empty SparseVector of a specified or calculated length.

    :param spark_session: SparkSession instance.
    :param df: DataFrame containing vector columns.
    :param column_names: List of vector column names to check and fill.
    :param vector_length: Optional; the length of the SparseVector to use for filling nulls.
    :return: DataFrame with null vectors filled.
    """
    if vector_length is None:
        # Attempt to determine the vector length automatically from non-null entries.
        sample_vector = df.select(column_names).dropna().limit(1).collect()
        vector_length = len(sample_vector[0][0]) if sample_vector else 0

    # Define a UDF to fill null vectors with an empty SparseVector of determined length.
    fill_vector_udf = udf(lambda v: v if v is not None else SparseVector(vector_length, []), VectorUDT())

    for col_name in column_names:
        df = df.withColumn(col_name, fill_vector_udf(col(col_name)))

    return df

def left_join_vectors(spark_session, left_df, right_df, key_columns, vector_columns, vector_length=None):
    """
    Performs a left join on two DataFrames and fills null vector columns in the result with an empty SparseVector.

    :param spark_session: SparkSession instance.
    :param left_df: Left DataFrame to join.
    :param right_df: Right DataFrame to join.
    :param key_columns: Columns to join on.
    :param vector_columns: List of vector columns to fill nulls.
    :param vector_length: Optional; the length of the SparseVector to use for filling nulls.
    :return: Joined DataFrame with null vectors filled.
    """
    joined_df = left_df.join(right_df, key_columns, "left")

    if vector_length is None:
        # Try to infer vector length if not provided.
        sample_vector = right_df.select(vector_columns).dropna().limit(1).collect()
        vector_length = len(sample_vector[0][0]) if sample_vector else 0

    # Define UDF to fill null vectors.
    fill_vector_udf = udf(lambda v: SparseVector(vector_length, []) if v is None else v, VectorUDT())

    for col_name in vector_columns:
        joined_df = joined_df.withColumn(col_name, fill_vector_udf(col(col_name)))

    return joined_df