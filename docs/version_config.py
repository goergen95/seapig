"""Retrieve version for docs."""

from importlib_metadata import version as _version

v = f"""VERSION={_version("seapig")}"""

f = open("_environment", "w")
f.write(v)
