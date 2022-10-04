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
        except json.decoder.JSONDecodeError as excp:
            raise json.decoder.JSONDecodeError(f'File: {file}', excp.doc, excp.pos) from excp
