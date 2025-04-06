import pathlib
import re

try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    from importlib_metadata import version, PackageNotFoundError  # Python < 3.8

try:
    __version__ = version("scitopt")
except PackageNotFoundError:
    # Fallback: Read from pyproject.toml when not installed
    def read_version_from_pyproject():
        root = pathlib.Path(__file__).resolve().parent.parent
        pyproject_path = root / "pyproject.toml"
        if pyproject_path.exists():
            content = pyproject_path.read_text()
            match = re.search(r'version\s*=\s*"(.*?)"', content)
            if match:
                return match.group(1)
        return "0.0.0"  # fallback version

    __version__ = read_version_from_pyproject()


from scitopt import mesh, core, fea, tools

__all__ = []
__all__.append("mesh")
__all__.append("core")
__all__.append("fea")
__all__.append("tools")