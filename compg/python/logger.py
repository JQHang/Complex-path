import logging

def setup_logger(log_filename=None, logger_name='temp', level='DEBUG'):
    """
    Set up and return a logger with handlers configured for console output and optionally file output.
    It also ensures no duplicate handlers are added.

    Args:
        log_filename (str, optional): The name of the file where logs should be written. If None, no file handler is added.
        logger_name (str): The name of the logger.
        level (str): The log level as a string (e.g., 'DEBUG', 'INFO').

    Returns:
        logging.Logger: A configured logger that logs to console and optionally to the specified file, without duplicating handlers.
    """
    # Convert the level string to a logging level
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')

    # Get or create the logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(numeric_level)

    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    # Create a log format
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Create and configure a handler for outputting to the console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # If a log_filename is provided, add a file handler as well
    if log_filename:
        file_handler = logging.FileHandler(log_filename)
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        logger.info(f"Messages stored at {log_filename}")

    return logger