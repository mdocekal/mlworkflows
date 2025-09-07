import importlib
import sys
from pathlib import Path
from types import ModuleType


def load_module(path: str) -> ModuleType:
    """
    Load module on given path and names it. It also imports all packages on given path to allow relative imports.

    :param path: path to the module
    :return: loaded module
    """

    p = Path(path)
    module_name_parts = [p.stem]
    search_location = None
    for parent in p.parents:
        if parent.name != "." and parent.name != ".." and parent.joinpath("__init__.py").is_file():
            # we don't want to use . and .. for module name
            module_name_parts.append(parent.name)
            search_location = str(parent.absolute())
            continue

        break

    module_name = ".".join(reversed(module_name_parts))

    if len(module_name_parts) > 1:
        spec = importlib.util.spec_from_file_location(module_name_parts[-1],
                                                      str(Path(search_location).joinpath("__init__.py")),
                                                      submodule_search_locations=[search_location])
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name_parts[-1]] = module
        spec.loader.exec_module(module)

    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module
