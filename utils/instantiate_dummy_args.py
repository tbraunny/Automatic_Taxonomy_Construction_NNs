import inspect

import inspect
import torch
import torch.nn as nn

def create_dummy_value(param):
    # create dummy vars if default values not listed
    if param.annotation in [int, inspect.Parameter.empty] and 'size' in param.name:
        return 784
    if param.annotation == int:
        return 128
    elif param.annotation == float:
        return 0.1
    elif param.annotation == bool:
        return True
    elif param.annotation == str:
        return "dummy"
    elif 'size' in param.name:
        return 784
    elif 'hidden' in param.name:
        return 128
    elif 'num' in param.name:
        return 1
    elif 'channels' in param.name:
        return 3
    elif 'dim' in param.name:
        return 64
    else:
        return 1  # fallback

def instantiate_with_dummy_args(cls):
    sig = inspect.signature(cls.__init__)
    dummy_args = {}

    for name, param in sig.parameters.items():
        if name == 'self':
            continue
        if param.default == inspect.Parameter.empty: #only required parameters get dummy values
            dummy_args[name] = create_dummy_value(param)
    print("DUMMY VARS" , cls(**dummy_args))
    try:
        return cls(**dummy_args)
    except Exception as e:
        print(f"[!] Could not instantiate {cls.__name__}: {e}")
        return None