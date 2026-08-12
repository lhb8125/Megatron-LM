#!/usr/bin/env python3
# Convert a sync-free HybridEP recipe YAML (ARGS / ENV_VARS blocks) into shell.
#   usage: yaml_to_shell.py <recipe.yaml> {args|env}
#     args -> prints "--kebab-case value ..." for pretrain_gpt.py
#             (shell-quoted so values with (), [], spaces survive; list values
#              are expanded into multiple space-separated tokens)
#     env  -> prints "export KEY='VALUE'\n..." lines
import shlex, sys, yaml

def main():
    path, mode = sys.argv[1], sys.argv[2]
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if mode == "env":
        for k, v in (cfg.get("ENV_VARS") or {}).items():
            print(f"export {k}={str(v)!r}")
    elif mode == "args":
        out = []
        for k, v in (cfg.get("ARGS") or {}).items():
            flag = "--" + k.replace("_", "-")
            if isinstance(v, bool):
                if v:
                    out.append(flag)
            elif isinstance(v, (list, tuple)):
                # e.g. recompute_modules: [a, b] -> --recompute-modules a b
                out.append(flag)
                out.extend(shlex.quote(str(x)) for x in v)
            else:
                out.append(flag)
                out.append(shlex.quote(str(v)))
        print(" ".join(out))
    else:
        sys.exit(f"unknown mode {mode}")

if __name__ == "__main__":
    main()
