from ..._mod_replace import replace_modname
from ._writer import TensorboardWriter, TextWriter, WriterBase

for _mod in (WriterBase, TensorboardWriter, TextWriter):
    replace_modname(_mod, __name__)

__all__ = ["TensorboardWriter", "TextWriter", "WriterBase"]
