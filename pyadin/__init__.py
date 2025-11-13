from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("pyadintool")
except PackageNotFoundError:
    __version__ = "unknown"

from .app import check_device as check_device
from .app import run_proclist as run_proclist
from .app import run_realtime as run_realtime
from .app import setup_config as setup_config
from .app import setup_pipeline as setup_pipeline
from .app import setup_logger as setup_logger
from .app import estimate_filter as estimate_filter
from .app import estimate_framepower as estimate_framepower
from .app import app_pyadintool as app_pyadintool
from .app import app_auxtool as app_auxtool
