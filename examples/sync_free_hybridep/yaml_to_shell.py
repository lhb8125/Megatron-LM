#!/usr/bin/env python3
# Convert a sync-free HybridEP recipe YAML (ARGS / ENV_VARS blocks) into shell.
#   usage: yaml_to_shell.py <recipe.yaml> {args|env}
#     args -> prints "--kebab-case value ..." for pretrain_gpt.py.
#             The caller consumes this via an UNQUOTED ${TRAIN_ARGS} expansion,
#             which only word-splits (no shell quote-removal), so values must be
#             emitted WITHOUT shell quotes. Values containing whitespace are not
#             supported in args mode (none of pretrain_gpt.py's flags need it;
#             expressions like "([0]*4+[1]*24)" are whitespace-free).
#     env  -> prints "export KEY='VALUE'\n..." lines
import sys, yaml

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
                out.extend(str(x) for x in v)
            else:
                token = str(v)
                if any(c.isspace() for c in token):
                    sys.exit(
                        f"args mode cannot emit whitespace value for {flag!r}: "
                        f"{token!r} (consumed via unquoted shell expansion)"
                    )
                out.append(flag)
                out.append(token)
        print(" ".join(out))
    else:
        sys.exit(f"unknown mode {mode}")

if __name__ == "__main__":
    main()
