"""
Centralized logging setup for the crowd counting pipeline.

Usage:
    from logger import get_logger

    log = get_logger(__name__)
    log.info("Processing started")
    log.debug("Detection result: %s", detection_result)
    log.warning("Empty frame received")
    log.error("Model failed to load", exc_info=True)
"""

import logging
import sys
import time
from pathlib import Path

# ---------------------------------------------------------
# Config -- tweak these as needed
# ---------------------------------------------------------
LOG_DIR = Path("output/logs")
LOG_LEVEL = logging.DEBUG          # what gets captured overall
CONSOLE_LEVEL = logging.DEBUG       # what prints to terminal
FILE_LEVEL = logging.DEBUG         # what gets written to file
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

_configured = False


def _configure_root():
    """Set up the root logger once (console + rotating-by-run file)."""
    global _configured
    if _configured:
        return

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    log_file = LOG_DIR / f"run_{run_stamp}.log"

    root = logging.getLogger()
    root.setLevel(LOG_LEVEL)

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("tensorflow").setLevel(logging.WARNING)

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(CONSOLE_LEVEL)
    console_handler.setFormatter(formatter)
    root.addHandler(console_handler)

    # File handler -- one file per run, full detail
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(FILE_LEVEL)
    file_handler.setFormatter(formatter)
    root.addHandler(file_handler)

    _configured = True
    root.info(f"Logging initialized -> {log_file}")


def get_logger(name: str) -> logging.Logger:
    """Get a module-scoped logger. Call this once at the top of each module."""
    _configure_root()
    return logging.getLogger(name)


class StepTimer:
    """Context manager to log how long a pipeline step takes.

    Usage:
        with StepTimer(log, "Head detection"):
            detection_result = self.head_detector.detect(sparse_input)
    """

    def __init__(self, logger: logging.Logger, step_name: str):
        self.logger = logger
        self.step_name = step_name
        self.start = None

    def __enter__(self):
        self.start = time.time()
        self.logger.info(f"START | {self.step_name}")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed = time.time() - self.start
        if exc_type is None:
            self.logger.info(f"DONE  | {self.step_name} ({elapsed:.2f}s)")
        else:
            self.logger.error(
                f"FAILED | {self.step_name} ({elapsed:.2f}s) -> {exc_val}",
                exc_info=True,
            )
        # returning False re-raises the exception, which we want
        return False

def prompt_input(logger: logging.Logger, message: str) -> str:
    """
    Show `message` styled like a normal log line, wait for user input
    on the SAME line, and still record the prompt in the log file.
    """
    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)
    record = logger.makeRecord(logger.name, logging.INFO, "", 0, message, None, None)
    formatted = formatter.format(record)

    # Write the formatted line to the file handler(s) only (with newline),
    # so it's captured in the log exactly like log.info() would write it.
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler):
            handler.stream.write(formatted + "\n")
            handler.flush()

    # Console: use the same formatted text as the input() prompt itself,
    # so the cursor waits right after it on the same line.
    return input(formatted + " ")