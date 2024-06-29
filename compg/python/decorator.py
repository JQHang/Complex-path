from datetime import datetime
import logging
from functools import wraps
from .logger import setup_logger

def time_costing(func):
    """
    A decorator that logs or prints the start time, end time, and duration of the execution
    of a function. It checks if a 'logger' key is provided in keyword arguments and if it
    is a valid logger instance. If a valid logger is provided, it uses it for logging.
    Otherwise, it uses print for output.

    Parameters:
    func (callable): The function to be decorated.

    Returns:
    callable: The wrapped function that now logs or prints its execution details.
    """
    def wrapper(*args, **kwargs):
        # Check if a valid logger is provided in keyword arguments
        logger = kwargs.get('logger', None)
        if not isinstance(logger, logging.Logger):
            logger = setup_logger()
        
        # Log or print the beginning of the operation
        start = datetime.now()
        logger.info(f"Start Function: {func.__name__}")
        
        # Execute the decorated function
        result = func(*args, **kwargs)
        
        # Log or print the end of the operation
        end = datetime.now()
        duration = end - start
        logger.info(f"End Function: {func.__name__}, Time Costing: {duration}")
        
        return result
    return wrapper

def ensure_logger(func):
    """Decorator: Ensures that the function has a correct logging.Logger instance.
    
    This decorator checks if the function has a keyword argument named 'logger'.
    If 'logger' is not an instance of logging.Logger or not present, 
    it sets up a new logger using the `setup_logger()` function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        # Check if 'logger' is in kwargs and if it is an instance of logging.Logger
        if 'logger' in kwargs and not isinstance(kwargs['logger'], logging.Logger):
            kwargs['logger'] = setup_logger()
            kwargs['logger'].warning(f"Using a temporary logger for function: {func.__name__}")
        elif 'logger' not in kwargs:
            kwargs['logger'] = setup_logger()
            kwargs['logger'].warning(f"Using a temporary logger for function: {func.__name__}")
        return func(*args, **kwargs)
    return wrapper