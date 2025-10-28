# FILEPATH (visible): /Users/juanjuan1/Desktop/chase/auto/drivers/__init__.py
# filepath: /Users/juanjuan1/Desktop/chase/auto/drivers/__init__.py
import platform
from typing import Optional
from .base import SoftwareDriverBase
from .mock import MockDriver
from .mac import MacDriver
from .windows import WindowsDriver
from .linux import LinuxDriver

def create_driver(engine_time: float = 4.0, driver: Optional[str] = None, mock: bool = False) -> SoftwareDriverBase:
    if mock:
        return MockDriver(engine_time)

    name = (driver or "auto").lower()
    if name == "mac":
        return MacDriver(engine_time)
    if name in ("win", "windows"):
        return WindowsDriver(engine_time)
    if name in ("linux", "gnu/linux"):
        return LinuxDriver(engine_time)

    sysname = platform.system().lower()  # 'darwin' | 'windows' | 'linux'
    if sysname == "darwin":
        return MacDriver(engine_time)
    if sysname == "windows":
        return WindowsDriver(engine_time)
    if sysname == "linux":
        return LinuxDriver(engine_time)

    return MockDriver(engine_time)