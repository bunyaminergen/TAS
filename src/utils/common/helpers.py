# Standard library imports
import gc
import time
from typing import Annotated, Tuple

# Third-party imports
import torch


class Timer:
    """
    Timer is a utility class to measure elapsed time between events.

    This class allows you to start and stop a timer, and then calculate
    the elapsed time in hours, minutes, and seconds.

    Attributes
    ----------
    start_time : float or None
        The start time recorded by `start()`.
    end_time : float or None
        The end time recorded by `end()`.

    Methods
    -------
    start()
        Starts the timer by recording the current time.
    end()
        Stops the timer and records the current time.
    calculate()
        Calculates the elapsed time between start and end.

    Examples
    --------
    >>> timer = Timer()
    >>> timer.start()
    >>> time.sleep(1)
    >>> timer.end()
    >>> h, m, s = timer.calculate()
    >>> round(s) == 1
    True
    """

    def __init__(self):
        self.start_time = None
        self.end_time = None

    def start(self) -> None:
        """
        Starts the timer by recording the current time.
        """
        self.start_time = time.time()

    def end(self) -> None:
        """
        Stops the timer and records the current time.

        Raises
        ------
        ValueError
            If the timer was not started before calling this method.
        """
        if self.start_time is None:
            raise ValueError("Timer was not started.")
        self.end_time = time.time()

    def calculate(self) -> Annotated[Tuple[int, int, float], "Elapsed time as (hours, minutes, seconds)"]:
        """
        Calculates the time elapsed between start and end.

        Returns
        -------
        tuple of int, int, float
            A tuple containing hours, minutes, and seconds.

        Raises
        ------
        ValueError
            If start or end time is not recorded.

        Examples
        --------
        >>> timer = Timer()
        >>> timer.start()
        >>> time.sleep(0.1)
        >>> timer.end()
        >>> h, m, s = timer.calculate()
        >>> h == 0 and m == 0 and s > 0
        True
        """
        if self.start_time is None or self.end_time is None:
            raise ValueError("Timer has not been started or ended.")
        duration = self.end_time - self.start_time
        hours = int(duration // 3600)
        minutes = int((duration % 3600) // 60)
        seconds = duration % 60
        return hours, minutes, seconds


class GpuMemoryReleaser:
    """
    Utility class for releasing GPU memory.

    This class provides a static method to delete a model from memory
    and clear the CUDA cache.

    Methods
    -------
    release(obj)
        Releases the GPU memory by deleting the model attribute and
        clearing cache.

    Examples
    --------
    >>> class Dummy:
    ...     def __init__(self):
    ...         self.model = torch.nn.Linear(2, 2).to("cuda")
    >>> obj = Dummy()
    >>> GpuMemoryReleaser.release(obj)
    """

    @staticmethod
    def release(
            obj: Annotated[object, "Object containing a model attribute"]
    ) -> None:
        """
        Releases the GPU memory used by the object's model.

        Parameters
        ----------
        obj : object
            Object that may contain a 'model' attribute.

        Returns
        -------
        None
        """
        if hasattr(obj, 'model'):
            del obj.model
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    test_timer = Timer()
    test_timer.start()
    print("End-to-end process started.")

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")

        model = torch.nn.Linear(1024, 1024).to(device)
        input_data = torch.randn(64, 1024, device=device)

        output = model(input_data)
        print(f"Output shape: {output.shape}")

        GpuMemoryReleaser.release(model)

    except Exception as e:
        print(f"An error occurred: {e}")

    finally:
        test_timer.end()
        test_h, test_m, test_s = test_timer.calculate()
        print(f"Elapsed time: {test_h}h {test_m}m {test_s:.3f}s")
