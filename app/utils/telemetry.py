import json
import logging
import os
import time
from typing import Any

# Global start time for processing duration metrics
_start_time = time.time()


class StructuredJsonFormatter(logging.Formatter):
    """
    Log formatter that outputs logs as a single line JSON object.
    Perfect for integration with cloud watch, Datadog, ELK, etc.
    """

    def format(self, record: logging.LogRecord) -> str:
        log_record: dict[str, Any] = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }

        # Include traceback details if an exception occurred
        if record.exc_info:
            log_record["exception"] = self.formatException(record.exc_info)

        # Merge standard log extra attributes if present
        # In Python logging, record.__dict__ contains custom fields passed in extra={...}
        # Standard LogRecord attributes to ignore when copying extra
        ignored_attrs = {
            "args", "asctime", "created", "exc_info", "exc_text", "filename",
            "funcName", "levelname", "levelno", "lineno", "module", "msecs",
            "message", "msg", "name", "pathname", "process", "processName",
            "relativeCreated", "stack_info", "thread", "threadName"
        }

        for key, value in record.__dict__.items():
            if key not in ignored_attrs:
                log_record[key] = value

        return json.dumps(log_record)


def setup_logging() -> None:
    """
    Configures the application-wide logging system.
    If the environment variable LOG_FORMAT is set to 'json', it enables
    structured JSON logging across the application.
    """
    log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    log_format = os.getenv("LOG_FORMAT", "text").strip().lower()

    root_logger = logging.getLogger()
    
    # Remove existing handlers to avoid duplicate log entries
    for handler in list(root_logger.handlers):
        root_logger.removeHandler(handler)

    if log_format == "json":
        handler = logging.StreamHandler()
        # Use our structured formatter
        handler.setFormatter(StructuredJsonFormatter())
        root_logger.addHandler(handler)
        root_logger.setLevel(log_level)
    else:
        # Standard color/text formatting
        logging.basicConfig(
            level=log_level,
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        )
