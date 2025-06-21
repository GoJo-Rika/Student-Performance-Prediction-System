import sys

from logger import logging


def error_message_detail(error: Exception, error_detail: sys) -> str:
    _, _, exc_tb = error_detail.exc_info()
    file_name = exc_tb.tb_frame.f_code.co_filename
    error_message = f"Error occured in python script name [{file_name}] line number [{exc_tb.tb_lineno}] error message[{error!s}]"

    return error_message


class CustomError(Exception):
    def __init__(self, error_message: str, error_detail: sys) -> None:
        super().__init__(error_message)
        self.error_message = error_message_detail(
            error_message,
            error_detail=error_detail,
        )

    def __str__(self) -> str:
        return self.error_message
