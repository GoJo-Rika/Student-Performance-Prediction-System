import pickle
import sys
from pathlib import Path

from src.exception import CustomError


def save_object(file_path: str, obj: object) -> None:
    try:
        dir_path = Path(file_path).parent

        dir_path.mkdir(parents=True, exist_ok=True)

        with Path(file_path).open("wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomError(e, sys)
