import sys
import importlib

CONTRACTS = {
    "split": {
        "module": "src.split",
        "function": "create_splits",
        "owner": "Person 2"
    },
    "train": {
        "module": "src.train",
        "function": "train_model",
        "owner": "Person 3"
    },
    "evaluate": {
        "module": "src.evaluate",
        "function": "evaluate_predictions",
        "owner": "Person 4"
    },
    "eval_model": {
        "module": "src.evaluate",
        "function": "run_evaluation",
        "owner": "Person 4"
    },
    "inference": {
        "module": "inference",
        "function": "predict",
        "owner": "Person 5"
    }
}

def load_module(name):
    if name not in CONTRACTS:
        return None
    mod_name = CONTRACTS[name]["module"]
    try:
        return importlib.import_module(mod_name)
    except ModuleNotFoundError as e:
        if e.name == mod_name or mod_name.startswith(e.name + "."):
            return None
        raise
    except ImportError:
        return None

def is_available(name):
    mod = load_module(name)
    if mod is None:
        return False
    fn_name = CONTRACTS[name]["function"]
    return hasattr(mod, fn_name)

def get_function(name):
    mod = load_module(name)
    if mod is None:
        owner = CONTRACTS.get(name, {}).get("owner", "Contributor")
        mod_name = CONTRACTS.get(name, {}).get("module", name)
        sys.stderr.write(f"Error: {mod_name} from {owner} is not available.\n")
        return None
    fn_name = CONTRACTS[name]["function"]
    if not hasattr(mod, fn_name):
        owner = CONTRACTS.get(name, {}).get("owner", "Contributor")
        sys.stderr.write(f"Error: {fn_name} not found in {CONTRACTS[name]['module']} ({owner}).\n")
        return None
    return getattr(mod, fn_name)
