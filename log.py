"""Logging helpers that capture stdout/stderr and tee output streams.

Provides simple stream wrappers used by the training scripts to duplicate
console output to a file while keeping the progress bar output clean.
"""

import sys
from collections.abc import Sequence
from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class Writable(Protocol):
    """Protocol representing a minimal writable stream.

    Implementations should provide ``write`` and ``flush`` methods so
    they can be used as sinks for text output (stdout/stderr).
    """

    def write(self, *args: Any, **kwargs: Any) -> object:
        """Write data to the stream.

        This method mirrors the standard file-like ``write`` signature and
        accepts positional and keyword arguments for compatibility.
        """
        ...

    def flush(self) -> None:
        """Flush any buffered output to the underlying storage."""
        ...


class OutputLogger(object):
    """Logger that captures stdout/stderr and optionally writes to a file.

    Parameters
    ----------
    file : str | None, optional
        Log file name. Defaults to None (no file logging initially).
    mode : str, optional
        File open mode. Defaults to "at" (append text).
    buffer : str | None, optional
        Buffer to store log data before file is set. Defaults to empty string.
    """

    def __init__(
        self, file: str | None = None, mode: str = "at", buffer: str | None = ""
    ) -> None:
        """Initialize the OutputLogger.

        Attributes
        ----------
        file : str | None
            The log file name.
        mode : str
            The mode in which to open the log file.
        buffer : str | None
            The buffer to store log data before writing to the file.
        """
        self.file = file
        self.mode = mode
        self.buffer = buffer

    def set_log_file(self, filename: str, mode: str = "at") -> None:
        """Set the log file and write any buffered data to it.

        Parameters
        ----------
        filename : str
            The name of the log file.
        mode : str, optional
            The mode in which to open the log file. Defaults to "at" (append text).
        """
        assert self.file is None
        self.file = filename
        self.mode = mode
        if self.buffer is not None:
            with open(self.file, self.mode) as f:
                f.write(self.buffer)
                self.buffer = None

    def write(self, data: str) -> None:
        r"""Write data to the log file and buffer.

        Skips lines starting with \r (typically emitted by progress bars)
        to avoid polluting log files with carriage-return-only updates.

        Parameters
        ----------
        data : str
            The data to write.
        """
        # Do not save tqdm print
        if data.startswith("\r"):
            return
        if self.file is not None:
            with open(self.file, self.mode) as f:
                f.write(data)
        if self.buffer is not None:
            self.buffer += data

    def flush(self) -> None:
        """Flush the log file buffer to disk."""
        if self.file is not None:
            with open(self.file, self.mode) as f:
                f.flush()


class TeeOutputStream(object):
    """Stream that writes to multiple child streams simultaneously.

    Similar to the Unix 'tee' command, duplicates output to multiple
    destinations.

    Parameters
    ----------
    child_streams : Sequence[Writable]
        List of output streams to write to.
    auto_flush : bool, optional
        Whether to automatically flush after each write. Defaults to False.
    """

    def __init__(
        self, child_streams: Sequence[Writable], auto_flush: bool = False
    ) -> None:
        """Initialize the TeeOutputStream.

        Attributes
        ----------
        child_streams : Sequence[Writable]
            List of child streams to write to.
        auto_flush : bool
            Whether to automatically flush after each write.
        buffer : str
            Internal buffer (currently unused).
        """
        self.child_streams = child_streams
        self.auto_flush = auto_flush
        self.buffer = ""

    def write(self, data: str) -> None:
        """Write data to all child streams.

        Parameters
        ----------
        data : str
            The data to write to each stream.
        """
        for stream in self.child_streams:
            stream.write(data)
        if self.auto_flush:
            self.flush()

    def flush(self) -> None:
        """Flush all child streams."""
        for stream in self.child_streams:
            stream.flush()


def init_output_logging(
    filename: str, mode: str = "at", output_logger: OutputLogger | None = None
) -> None:
    """Initialize output logging to redirect stdout/stderr to a file.

    Parameters
    ----------
    filename : str
        The name of the file to log output to.
    mode : str, optional
        The mode in which to open the log file. Defaults to "at" (append text).
    output_logger : OutputLogger | None, optional
        An existing OutputLogger instance to use. If None, creates a new one
        and redirects sys.stdout and sys.stderr. Defaults to None.
    """
    if output_logger is None:
        output_logger = OutputLogger()
        sys.stdout = TeeOutputStream([sys.stdout, output_logger], auto_flush=True)
        sys.stderr = TeeOutputStream([sys.stderr, output_logger], auto_flush=True)
    output_logger.set_log_file(filename, mode)


def format_time(seconds: int) -> str:
    """Format time duration in seconds to human-readable string.

    Parameters
    ----------
    seconds : int
        The time duration in seconds.

    Returns
    -------
    str
        Formatted time string in the format:
        - "Xs" for seconds < 60
        - "Xm YYs" for seconds < 3600
        - "Xh YYm ZZs" for seconds < 86400
        - "Xd YYh ZZm" for seconds >= 86400
    """
    s = int(round(seconds))
    if s < 60:
        return f"{s}s"
    elif s < 3600:
        return f"{s // 60}m {s % 60:02d}s"
    elif s < 86400:
        return f"{s // 3600}h {s // 60 % 60:02d}m {s % 60:02d}s"
    else:
        return f"{s // 86400}d {s // 3600 % 24:02d}h {s // 60 % 60:02d}m"
