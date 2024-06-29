import logging
from datetime import datetime
import time
import gc
from pyspark.sql import SparkSession
from py4j.protocol import Py4JError, Py4JJavaError, Py4JNetworkError
from compg.python import time_costing, ensure_logger

@time_costing
@ensure_logger
def start_spark_session(mode='local', config_dict = None, log_level = "ERROR", logger = None):
    """
    Initializes and returns a SparkSession.

    Args:
        mode (str): Execution mode ('local' or 'cluster').
        config_dict (dict): Additional configuration parameters.

    Returns:
        SparkSession: The initialized Spark session.
    """
    # Initialize the SparkSession builder
    builder = SparkSession.builder.appName(f"Complex_graph SparkSession")

    # Configure based on the specified mode
    if mode == 'local':
        builder = builder.master("local[*]")
        logger.info("Running in local mode...")
        default_configs = {
            "spark.driver.memory": "80g",
            "spark.driver.cores": "4",
            "spark.driver.maxResultSize": "0"
        }
    else:
        builder = builder.enableHiveSupport()
        logger.info("Running in cluster mode...")
        default_configs = {
            "spark.default.parallelism": "600",
            "spark.sql.shuffle.partitions": "1200",
            "spark.sql.broadcastTimeout": "3600",
            "spark.driver.memory": "80g",
            "spark.driver.cores": "4",
            "spark.driver.maxResultSize": "0",
            "spark.executor.memory": "40g",
            "spark.executor.cores": "4",
            "spark.executor.instances": "150",
            "spark.yarn.appMasterEnv.yarn.nodemanager.container-executor.class": "DockerLinuxContainer",
            "spark.executorEnv.yarn.nodemanager.container-executor.class": "DockerLinuxContainer",
            "spark.yarn.appMasterEnv.yarn.nodemanager.docker-container-executor.image-name": "bdp-docker.jd.com:5000/wise_mart_bag:latest",
            "spark.executorEnv.yarn.nodemanager.docker-container-executor.image-name": "bdp-docker.jd.com:5000/wise_mart_bag:latest",
            "spark.sql.crossJoin.enabled": "true",
            "spark.sql.execution.arrow.pyspark.enabled": "true",
            "spark.driver.allowMultipleContexts": "false",
            "spark.sql.autoBroadcastJoinThreshold": str(100 * 1024 * 1024),
            "spark.sql.adaptive.shuffle.targetPostShuffleInputSize": str(300 * 1024 * 1024)
        }

    # Merge user-provided configurations
    if config_dict:
        default_configs.update(config_dict)

    # Apply configurations
    for key, value in default_configs.items():
        builder = builder.config(key, value)

    # Create the Spark session
    spark_session = builder.getOrCreate()

    # Set log level
    spark_session.sparkContext.setLogLevel(log_level)
    
    # Print the application ID and other relevant URLs
    app_id = spark_session.sparkContext.applicationId
    logger.info("Application ID: http://10k2.jd.com/proxy/" + app_id)
    logger.info(f"History Record: http://10k.sparkhs-3.jd.com/history/{app_id}/stages/")

    return spark_session

class ResilientSparkRunner:
    def __init__(self, mode = 'cluster', config_dict = None, max_restarts = 20, active_hours = [(9, 24)], log_level = "ERROR", logger = None):
        # Check if a valid logger is provided in keyword arguments
        if not isinstance(logger, logging.Logger):
            logger = setup_logger()
        
        self.mode = mode
        self.config_dict = config_dict or {}
        self.max_restarts = max_restarts
        self.active_hours = active_hours
        self.spark_restart_count = 0
        self.log_level = log_level
        self.logger = logger
        
        self.spark_session = self.start_new_spark_session()

    def start_new_spark_session(self):
        # Assuming start_spark_session is defined elsewhere
        return start_spark_session(self.mode, self.config_dict, self.log_level, logger = self.logger)

    def handle_error(self, error):
        self.logger.error("An error occurred:", str(error))

        # Attempt to stop the Spark session
        if self.spark_session:
            try:
                self.spark_session.stop()
            except Exception as stop_exception:
                self.logger.error("Error stopping SparkSession:", str(stop_exception))
            self.spark_session = None
            gc.collect()

        interrupt_time = datetime.now()
        self.logger.info('Spark interrupted at:', interrupt_time)

        # Check if current time is within active hours
        if not self.is_active_time(interrupt_time):
            self.logger.info('Calculations cannot be performed off-hours, waiting to restart...')
            next_start_time = self.calculate_next_start_time(interrupt_time)
            time_to_wait = (next_start_time - interrupt_time).total_seconds() - 200
            time.sleep(time_to_wait)
            self.logger.info('Active hours have begun, attempting to restart...')

        if self.spark_restart_count >= self.max_restarts:
            self.logger.error(f'Maximum restarts exceeded ({self.max_restarts} times), stopping execution.')
            return False

        self.spark_restart_count += 1
        self.logger.info('Restarting Spark and resuming calculations...')
        time.sleep(5)
        self.spark_session = self.start_new_spark_session()
        
        return self.spark_session is not None

    def is_active_time(self, current_time):
        current_hour = current_time.hour
        for start, end in self.active_hours:
            if start <= current_hour < end:
                return True
        return False

    def calculate_next_start_time(self, current_time):
        current_hour = current_time.hour
        for start, end in sorted(self.active_hours):
            if current_hour < start:
                return datetime(current_time.year, current_time.month, current_time.day, start, 0)
        # If no start time is later today, return the first start time tomorrow
        start, _ = sorted(self.active_hours)[0]
        return datetime(current_time.year, current_time.month, current_time.day, start, 0) + timedelta(days=1)

    @time_costing
    def run(self, task_function, *args, **kwargs):
        result = None
        while True:
            try:
                result = task_function(self.spark_session, *args, **kwargs)
                break
            except (Py4JError, Py4JJavaError, Py4JNetworkError) as error:
                should_continue = self.handle_error(error)
                if not should_continue:
                    break
        return result