import json
import logging
import os
import shutil
import socket
import sys
from logging.handlers import TimedRotatingFileHandler
from pathlib import Path
from typing import Any


LOG_DIR = Path(__file__).resolve().parent.parent / "logs/"
LOG_DIR.mkdir(parents=True, exist_ok=True)
Path(LOG_DIR / "archive").mkdir(parents=True, exist_ok=True)


class CustomTimedRotatingFileHandler(TimedRotatingFileHandler):
    def __init__(self, dir_log, filename, when='h', interval=1, backupCount=0, encoding=None,
                 delay=False, utc=False, atTime=None):
        self.dir_log = dir_log
        self.dir_archive = os.path.join(dir_log, 'archive')
        self.filename = filename
        TimedRotatingFileHandler.__init__(self, os.path.join(dir_log, filename), when, interval, backupCount, encoding,
                                          delay, utc, atTime)

    def doRollover(self):
        current_log = self.baseFilename
        TimedRotatingFileHandler.doRollover(self)
        shutil.move(current_log, os.path.join(self.dir_archive, self.filename))


class JSONFormatter(logging.Formatter):
    """JSON log formatter."""

    def format(self, record):
        log_record = {
            "timestamp": self.formatTime(record, self.datefmt),
            "logging.googleapis.com/severity": record.levelname,
            "message": record.getMessage(),
            "name": record.name,
            "pathname": record.pathname,
            "lineno": record.lineno,
            "logging.googleapis.com/labels": {  # Add labels here
                "serial_number": getattr(record, "serial_number", None),
                "server_id": socket.gethostname(),
                "uuid": getattr(record, "uuid", None)
            },
        }
        return json.dumps(log_record)


def get_console_handler() -> logging.StreamHandler:
    """Return a console handler that logs INFO level and above."""
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(JSONFormatter())
    console_handler.setLevel(logging.DEBUG)
    return console_handler


def get_file_handler(name: str, level: str) -> CustomTimedRotatingFileHandler:
    """Return a file handler specific to a given log level."""
    file_handler = CustomTimedRotatingFileHandler(
        LOG_DIR, f"{name}_{level}.log", when="midnight", backupCount=7
    )
    file_handler.setFormatter(JSONFormatter())
    file_handler.setLevel(getattr(logging, level))
    return file_handler


def get_logger(name: str) -> Any:
    """Get a logger instance."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)  # Set logger level to DEBUG

    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    console_handler = get_console_handler()
    console_handler.setLevel(logging.DEBUG)  # Set console handler level to INFO
    logger.addHandler(console_handler)

    logger.propagate = False
    return logger


def close_logger(logger):
    if logger is None:  # If there are no handlers
        return False

    for handler in logger.handlers[:]:  # Iterate over a copy of the handlers list
        logger.removeHandler(handler)
    return True
