import sys
from types import TracebackType


class CustomException(Exception):
    """
    Custom exception class that provides detailed information
    about an exception for easier debugging.
    """

    def __init__(self, error: Exception, error_details: sys):
        super().__init__(str(error))
        self.error_message = self._build_error_message(error, error_details)

    @staticmethod
    def _build_error_message(
        error: Exception,
        error_details: sys,
    ) -> str:
        """
        Build a detailed exception message.
        """

        _, _, traceback = error_details.exc_info()

        if traceback is None:
            return str(error)

        tb: TracebackType = traceback

        file_name = tb.tb_frame.f_code.co_filename
        function_name = tb.tb_frame.f_code.co_name
        line_number = tb.tb_lineno
        exception_type = type(error).__name__

        return (
            "\n"
            "========== Exception Occurred ==========\n"
            f"Exception : {exception_type}\n"
            f"File      : {file_name}\n"
            f"Function  : {function_name}\n"
            f"Line      : {line_number}\n"
            f"Message   : {error}\n"
            "========================================"
        )

    def __str__(self) -> str:
        return self.error_message