import os
import json

def mkdir(path):
    """
    Creates a new directory at the specified path if it does not already exist.
    
    Parameters:
        path (str): The path where the directory will be created.
        
    Returns:
        None
    """
    # Check if the directory already exists
    if not os.path.exists(path):
        # If not, create the directory including any necessary parent directories
        os.makedirs(path)
        print("Create new directory:", path)
    else:
        print("Directory already exists:", path)

def read_json_file(file_path):
    """
    Read data from a JSON file and return it as a dictionary.

    Args:
        file_path (str): The path to the JSON file.

    Returns:
        dict: The data contained in the JSON file.
    """
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print("The specified file does not exist.")
    except json.JSONDecodeError as e:
        # 打印错误消息，包括出错的位置
        print(f"The file is not a valid JSON document. Error at line {e.lineno}, column {e.colno}: {e.msg}")

def write_json_file(data, file_path):
    """
    Write a dictionary to a JSON file.

    Args:
        data (dict): The data to write to the file.
        file_path (str): The path where the JSON file will be saved.

    Returns:
        None
    """
    try:
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)
    except IOError:
        print("An error occurred while writing to the file.")