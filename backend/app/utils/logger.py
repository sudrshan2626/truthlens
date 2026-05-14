import logging
import sys
def get_logger(name: str) -> logging.Logger:
    """
    Creates and returns a configured logger instance. 

    Args:
        name: Usually __name__ of the calling module

    Returns: 
        confighred Logger object 
    """

    logger = logging.getLogger(name)

    #Avoid adding duplicate handlers if logger already exists
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.DEBUG)

    #Create console handler
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.DEBUG)

      # Create formatter — defines what each log line looks like

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"

    )

    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger