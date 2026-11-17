import sys
from src.registry import get_function

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.stderr.write("Usage: python -m src.run [split|train|eval]\n")
        sys.exit(1)
    cmd = sys.argv[1].strip().lower()
    if cmd == "split":
        fn = get_function("split")
        if fn is None:
            sys.exit(1)
        fn()
    elif cmd == "train":
        fn = get_function("train")
        if fn is None:
            sys.exit(1)
        fn()
    elif cmd in ("eval", "evaluate"):
        fn = get_function("eval_model")
        if fn is None:
            sys.exit(1)
        fn()
    else:
        sys.stderr.write(f"Unknown command: {cmd}\n")
        sys.exit(1)
