import sys
from src.logger import logger

def message_detail(error):
    exc_type, exc_value, exc_traceback = sys.exc_info()
    if exc_traceback is not None:
        filename = exc_traceback.tb_frame.f_code.co_filename
        line_number = exc_traceback.tb_lineno
    else:
        filename = "Unknown"
        line_number = "Unknown"
    
    error_message = (
        f"Error Occurred -\n"
        f"  Python Script: [{filename}]\n"
        f"  Line Number: [{line_number}]\n"
        f"  Exception Type: [{type(error).__name__}]\n"
        f"  Error Message: [{error}]"
    )
    return error_message

class CustomException(Exception):
    def __init__(self, error_message):
        super().__init__(error_message)
        self.error_message = message_detail(error_message)

    def __str__(self):
        return self.error_message
    
if __name__=="__main__":
    try:
        x = 1 + "1"
    except Exception as e:
        logger.warning(CustomException(e))
