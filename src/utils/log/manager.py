# Standard library imports
import os
import sys
import logging
import traceback
from typing import Annotated, TextIO, Optional
from logging.handlers import RotatingFileHandler

# Third-party imports
from pydantic import ValidationError

# Local imports
from src.utils.type.schema import LoggerType, SingleLineConsoleFormatterType, SingleLineFileFormatterType


class TeeStream:
    """
    A stream wrapper that duplicates stdout output to a file.

    Parameters
    ----------
    stream : TextIO
        The original stream (usually sys.stdout).
    filepath : str
        The file to which output is also written.
    """

    def __init__(
            self,
            stream: Annotated[TextIO, "Output stream like sys.stdout"],
            filepath: Annotated[str, "Path to file for writing stream"]
    ) -> None:
        self.stream: TextIO = stream
        self.file: TextIO = open(filepath, "a")

    def write(self, message: str) -> None:
        """
        Write message to both stream and file.
        """
        self.stream.write(message)
        self.file.write(message)

    def flush(self) -> None:
        """
        Flush both stream and file.
        """
        self.stream.flush()
        self.file.flush()

    def __del__(self):
        """
        Close the file if still open when the stream is deleted.
        """
        if not self.file.closed:
            try:
                self.file.close()
            except OSError as e:
                logging.getLogger(__name__).warning(f"Failed to close TeeStream file: {e}")


class SingleLineConsoleFormatter(logging.Formatter):
    """
    Formatter for console output with application context.

    Parameters
    ----------
    app : str, optional
        Application name shown in the logs.
    date_format : str, optional
        Date/time format for log timestamps.
    """

    def __init__(
            self,
            app: Annotated[str, "Application name"] = "Ayvaz",
            date_format: Annotated[Optional[str], "Date format"] = None,
    ) -> None:
        tmp_logger = logging.getLogger(__name__)
        try:
            settings = SingleLineConsoleFormatterType(app=app, date_format=date_format)
        except ValidationError as e:
            tmp_logger.error("Invalid parameters for SingleLineConsoleFormatter: %s", e.json())
            raise

        self.app = settings.app
        self.date_format = settings.date_format
        console_format = "%(asctime)s - %(app)s - %(levelname)s - %(name)s - %(message)s"
        super().__init__(fmt=console_format, datefmt=self.date_format)

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record.
        """
        record.app = self.app
        return super().format(record)


class SingleLineFileFormatter(logging.Formatter):
    """
    Formatter for file logging with application context.

    Parameters
    ----------
    app : str, optional
        Application name shown in the logs.
    date_format : str, optional
        Date/time format for log timestamps.
    """

    def __init__(
            self,
            app: Annotated[str, "Application name"] = "Ayvaz",
            date_format: Annotated[Optional[str], "Date format"] = None,
    ) -> None:
        tmp_logger = logging.getLogger(__name__)
        try:
            settings = SingleLineFileFormatterType(app=app, date_format=date_format)
        except ValidationError as e:
            tmp_logger.error("Invalid parameters for SingleLineFileFormatter: %s", e.json())
            raise

        self.app = settings.app
        self.date_format = settings.date_format
        file_format = "%(asctime)s - %(app)s - %(levelname)s - %(name)s - %(message)s"
        super().__init__(fmt=file_format, datefmt=self.date_format)

    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record.
        """
        record.app = self.app
        return super().format(record)


class Logger:
    """
    A configurable logging utility for applications.

    Parameters
    ----------
    name : str
        Logger name.
    app : str, optional
        Application name used in log messages.
    path : str, optional
        Directory where logs are saved.
    file : str, optional
        Log file name.
    console_level : int, optional
        Logging level for console output.
    file_level : int, optional
        Logging level for file output.
    max_bytes : int, optional
        Max size of a log file before rotation.
    backup_count : int, optional
        Number of rotated files to keep.
    verbose : bool, optional
        Whether to log to console in addition to file.

    Examples
    --------
    >>> logger = Logger(name="MyLogger").get()
    >>> logger.info("Logger is working.")
    """

    def __init__(
            self,
            name: Annotated[str, "Logger name"],
            app: Annotated[str, "Application name"] = "TAS",
            path: Annotated[str, "Folder to store log files"] = ".logs/terminal/",
            file: Annotated[str, "Log file name"] = "TAS.log",
            console_level: Annotated[int, "Console log level"] = logging.DEBUG,
            file_level: Annotated[int, "File log level"] = logging.DEBUG,
            max_bytes: Annotated[int, "Max file size for rotation"] = 5_000_000,
            backup_count: Annotated[int, "Number of old logs to keep"] = 11,
            verbose: Annotated[bool, "Whether logs are printed to console"] = True,
    ) -> None:
        tmp_logger = logging.getLogger(__name__)
        try:
            settings = LoggerType(
                name=name,
                app=app,
                path=path,
                file=file,
                console_level=console_level,
                file_level=file_level,
                max_bytes=max_bytes,
                backup_count=backup_count,
                verbose=verbose,
            )
        except ValidationError as e:
            tmp_logger.error("Invalid Logger parameters: %s", e.json())
            raise

        self.name = settings.name
        self.app = settings.app
        self.path = settings.path
        self.file = settings.file
        self.console_level = settings.console_level
        self.file_level = settings.file_level
        self.max_bytes = settings.max_bytes
        self.backup_count = settings.backup_count
        self.verbose = settings.verbose

        self.logger = logging.getLogger(self.name)
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

        if not self.logger.handlers:
            self._setup()

        self._setup_tee()

    def _setup(self) -> None:
        """
        Set up log file and console handlers with formatters.
        """
        os.makedirs(self.path, exist_ok=True)
        log_file_path = os.path.join(self.path, self.file)

        file_handler = RotatingFileHandler(
            filename=log_file_path,
            maxBytes=self.max_bytes,
            backupCount=self.backup_count,
        )
        file_handler.setLevel(self.file_level)
        file_formatter = SingleLineFileFormatter(date_format="%Y-%m-%d %H:%M:%S", app=self.app)
        file_handler.setFormatter(file_formatter)
        self.logger.addHandler(file_handler)

        if self.verbose:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(self.console_level)
            console_formatter = SingleLineConsoleFormatter(date_format="%Y-%m-%d %H:%M:%S", app=self.app)
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)

    @staticmethod
    def _setup_exception(log_path: Annotated[str, "Path to error log file"]) -> None:
        """
        Capture and log unhandled exceptions to a log file.
        """

        def hook(exc_type, exc_value, exc_traceback):
            """
            Custom exception hook to log uncaught exceptions.
            """
            with open(log_path, "a") as log_file:
                log_file.write(
                    "".join(traceback.format_exception(exc_type, exc_value, exc_traceback))
                )
            sys.__excepthook__(exc_type, exc_value, exc_traceback)

        sys.excepthook = hook

    def _setup_tee(self) -> None:
        """
        Redirect stdout to both console and a log file.
        """
        if not isinstance(sys.stdout, TeeStream):
            sys.stdout = TeeStream(sys.stdout, os.path.join(self.path, "output.log"))
        self._setup_exception(os.path.join(self.path, "error.log"))

    def get(self) -> Annotated[logging.Logger, "Returns the configured logger instance"]:
        """
        Get the logger instance.

        Returns
        -------
        logging.Logger
            The configured logger.
        """
        return self.logger


if __name__ == "__main__":
    log_manager = Logger(app="TestApp", name="TestLogger")
    test_logger = log_manager.get()

    test_logger.debug("Application debug log.")
    test_logger.info("Application is starting.")
    test_logger.warning("Warning! Something unexpected happened.")
    test_logger.error("An error has occurred!")
    test_logger.critical("This is a critical message.")
