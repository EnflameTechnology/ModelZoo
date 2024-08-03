import importlib.util

def is_torch_dtu_available():
    if importlib.util.find_spec("torch_gcu") is None:
        return False
    if importlib.util.find_spec("torch_gcu.core") is None:
        return False
    return importlib.util.find_spec("torch_gcu.core.model") is not None