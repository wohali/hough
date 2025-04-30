from .analyse import analyse_page, get_pages
from .cli import run
from .rotate import rotate
from .stats import histogram


try:
    from importlib.metadata import PackageNotFoundError, version  # type: ignore
except ImportError:  # pragma: no cover
    from importlib_metadata import PackageNotFoundError, version  # type: ignore


try:
    __version__ = version(__name__)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"

__all__ = ["analyse_page", "get_pages", "run", "rotate", "histogram"]
