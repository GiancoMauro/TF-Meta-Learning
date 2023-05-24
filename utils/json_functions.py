import json
from pathlib import Path


def ensure_file(file_name):
    """Ensures that the given file exists and converts it into a Path.

    Args:
        file_name: Path of the file to check.

    Returns:
        Path object of the file.

    Raises:
        FileNotFoundError: If the file does not exist.
        IsADirectoryError: If the file is a directory.
    """
    file = Path(file_name)
    if not file.exists():
        raise FileNotFoundError(f"{file} doesn't exist.")
    if file.is_dir():
        raise IsADirectoryError(f"{file} is a directory.")
    return file


def read_json(file_name):
    """Reads a json file from disc and returns the content.

    Args:
        file_name: Path to the file to read.

    Returns:
        Read content.
    """
    file = ensure_file(file_name)
    with file.open('r') as fp:
        try:
            return json.load(fp)
        except json.decoder.JSONDecodeError as exception:
            raise json.decoder.JSONDecodeError(f'File: {file}', exception.doc, exception.pos) from exception

def generate_config_to_save(config_file):
    """ Eliminate redundant information from the main experiment configuration, while making it more compact.

    :param config_file: main experiment configuration file to be processed
    :return: adjusted configuration file
    """
    config_file["dataset"] = config_file["train_dataset"].__dict__.copy()
    config_file["dataset"]["data_folder_train"] = config_file["dataset"]["data_folder"]
    config_file["dataset"].update(config_file["test_dataset"].__dict__)
    config_file["dataset"]["data_folder_test"] = config_file["dataset"]["data_folder"]
    config_file["dataset"].pop("data_folder")
    config_file.pop("train_dataset")
    config_file.pop("test_dataset")
    config_file["dataset"].pop("classes_tags")
    config_file["dataset"].pop("dirs")
    config_file["dataset"].pop("classes")

    return config_file