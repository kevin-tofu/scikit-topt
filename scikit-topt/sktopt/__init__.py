import pathlib
import re
# from sktopt import mesh, core, fea, tools
from . import mesh, core, fea, tools


try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:
    # old python
    from importlib_metadata import version, PackageNotFoundError


def read_version_from_pyproject():
    root = pathlib.Path(__file__).resolve().parent.parent
    # while root != root.parent and depth < max_depth:
    pyproject_path = root / "pyproject.toml"
    if pyproject_path.exists():
        content = pyproject_path.read_text()
        match = re.search(r'version\s*=\s*"(.*?)"', content)
        if match:
            return match.group(1)
    # root = root.parent
    #     depth += 1
    return "0.0.0"


def get_version(package_name):
    try:
        # print("version-", version(package_name))
        return version(package_name)
    except PackageNotFoundError:
        return read_version_from_pyproject()


# __version__ = get_version("sktopt")
__version__ = get_version("scikit-topt")
# print(f"sktopt version: {__version__}")


__all__ = [
    "__version__",
    "mesh",
    "core",
    "fea",
    "tools",
]
