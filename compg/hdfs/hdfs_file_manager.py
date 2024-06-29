import json
from pyarrow import fs as pafs
from datetime import datetime
from compg.python import time_costing, ensure_logger

@ensure_logger
def create_hdfs_directory(hdfs_path, logger = None):
    """
    Creates a directory at the specified HDFS path.

    Args:
        hdfs_path (str): The HDFS path where the directory will be created.
    """  
    hdfs = pafs.HadoopFileSystem(host="default")
    try:
        hdfs.create_dir(hdfs_path)
        logger.info(f'Created directory at {hdfs_path}')
    except Exception as e:
        logger.info(f'Failed to create directory: {e}')

def hdfs_list_contents(hdfs_path, content_type="all", recursive=False):
    """
    Lists the contents (files and/or directories) in a specified HDFS directory.

    Args:
        hdfs_path (str): The HDFS path to list contents from.
        content_type (str): The type of contents to return. Valid values are "all", "files", or "directories".
        recursive (bool): Whether to recursively list contents in subdirectories.

    Returns:
        list: A list of file and/or directory paths under the specified path.
    """
    hdfs = pafs.HadoopFileSystem(host="default")
    file_info_list = hdfs.get_file_info(pafs.FileSelector(hdfs_path, recursive=recursive))

    contents = []

    for info in file_info_list:
        if info.type == pafs.FileType.File:
            if content_type in ["all", "files"]:
                contents.append(info.path)
        elif info.type == pafs.FileType.Directory:
            if content_type in ["all", "directories"]:
                contents.append(info.path)

    return contents

def check_hdfs_file_exists(hdfs_path):
    """
    Checks if a file or directory exists in HDFS at the specified path.

    Args:
        hdfs_path (str): The HDFS path to check.

    Returns:
        bool: True if the file or directory exists, False otherwise.
    """
    hdfs = pafs.HadoopFileSystem(host="default")
    return hdfs.get_file_info(hdfs_path).type != pafs.FileType.NotFound

@ensure_logger
def hdfs_save_text_file(hdfs_path, file_name = '_SUCCESS', content = None, logger = None):
    """
    Creates a text file at the specified HDFS path.

    Args:
        hdfs_path (str): The base path where the text file will be created.
        file_name (str): Name of the text file. Defaults to '_SUCCESS'.
        content (str, optional): Content to write to the text file. If None, writes a timestamp.
    """
    hdfs = pafs.HadoopFileSystem(host="default")
    text_file_path = f"{hdfs_path}/{file_name}"
    
    try:
        with hdfs.open_output_stream(text_file_path) as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            if content is None:
                content = f"Write operation completed successfully at {timestamp}"
            f.write(content.encode('utf-8'))
            logger.info(f'{file_name} file created at {text_file_path} with content: {content}')
    except Exception as e:
        logger.info(f'Failed to create text file: {e}')

@ensure_logger
def hdfs_save_dict(hdfs_path, file_name, data_dict, logger=None):
    """
    Converts a dictionary to a string and saves it as a text file in HDFS.

    Args:
        hdfs_path (str): The base path where the text file will be created in HDFS.
        file_name (str): Name of the text file.
        data_dict (dict): The dictionary to be converted and saved.
        logger (logging.Logger): The logger object for logging messages.
    """
    try:
        # Convert dictionary to JSON string
        dict_str = json.dumps(data_dict, ensure_ascii=False)

        # Create HDFS text file with the dictionary content
        hdfs_save_text_file(hdfs_path, file_name, content=dict_str, logger=logger)

    except Exception as e:
        logger.error(f'Failed to convert and save dictionary: {e}')

@ensure_logger
def hdfs_read_text_file(hdfs_path, file_name, logger = None):
    """
    Reads content from a text file located in an HDFS path.

    Args:
        hdfs_path (str): The base path where the text file is located.
        file_name (str): Name of the text file to read. Defaults to '_SUCCESS'.

    Returns:
        str or None: The content of the text file, or None if an error occurs.
    """
    hdfs = pafs.HadoopFileSystem(host="default")
    text_file_path = f"{hdfs_path}/{file_name}"

    try:
        with hdfs.open_input_stream(text_file_path) as f:
            return f.read().decode('utf-8')
    except Exception as e:
        logger.info(f'Failed to read text file: {e}')
        return None

@ensure_logger
def hdfs_read_dict(hdfs_path, file_name, logger=None):
    """
    Reads a dictionary saved as a text file in HDFS and converts it back to a dictionary.

    Args:
        hdfs_path (str): The base path where the text file is located in HDFS.
        file_name (str): Name of the text file containing the dictionary.
        logger (logging.Logger): The logger object for logging messages.

    Returns:
        dict: The dictionary read from the text file.
    """
    try:
        # Read the content of the text file from HDFS
        dict_str = hdfs_read_text_file(hdfs_path, file_name, logger=logger)

        if dict_str is not None:
            # Convert the JSON string back to a dictionary
            data_dict = json.loads(dict_str)
            logger.info(f'Dictionary read successfully from {hdfs_path}/{file_name}')
            return data_dict
        else:
            logger.warning(f'Failed to read dictionary from {hdfs_path}/{file_name}')
            return None

    except Exception as e:
        logger.error(f'Failed to read and convert dictionary: {e}')
        return None