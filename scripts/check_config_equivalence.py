"""
One-shot safety net for the "config functions as data" conversion.

The old registry read the *source text* of every config function line by line and kept
only the values it could recognise as literals (see git history: _parse_config_params).
Anything else — an expression, a reference to another name — was dropped silently and
later replaced by _get_smart_defaults, so an attack could run with parameters that are
not the ones written in its config.

This script re-implements that old text parser (without importing the attack libraries,
which are optional) and compares it with what the converted config functions actually
return. Two kinds of differences:

  MISSING  the old parser had a value the new config no longer provides  -> regression
  GAINED   the new config provides a value the old parser silently dropped -> the bug

The "old" side is read from git (default: HEAD), the "new" side from the working tree,
so the comparison stays meaningful after the conversion is committed.

Usage:  python scripts/check_config_equivalence.py [git-ref-with-old-configs]
"""
import ast
import subprocess
import sys
from pathlib import Path

CONFIG_FILES = sorted(Path('attackbench/attacks').glob('*/configs.py'))


def source_at(ref: str, path: Path) -> str:
    """File content at a git ref ('' if the file did not exist there)."""
    result = subprocess.run(['git', 'show', f'{ref}:{path}'],
                            capture_output=True, text=True)
    return result.stdout if result.returncode == 0 else ''


def old_parse(source_lines):
    """Verbatim behaviour of the removed _parse_config_params, on source text."""
    params = {}
    for line in source_lines[1:]:  # Skip function definition
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        if '=' in line and not line.startswith('#'):
            try:
                var_name, var_value = line.split('=', 1)
                var_name = var_name.strip()
                var_value = var_value.strip()
                if '#' in var_value:
                    var_value = var_value.split('#')[0].strip()
                if (var_value.startswith("'") and var_value.endswith("'")) or \
                   (var_value.startswith('"') and var_value.endswith('"')):
                    params[var_name] = var_value[1:-1]
                elif var_value == 'True':
                    params[var_name] = True
                elif var_value == 'False':
                    params[var_name] = False
                elif var_value == 'None':
                    params[var_name] = None
                else:
                    try:
                        if var_value.isdigit():
                            params[var_name] = int(var_value)
                        elif '.' in var_value and var_value.replace('.', '').isdigit():
                            params[var_name] = float(var_value)
                        elif '/' in var_value:
                            parts = var_value.split('/')
                            if len(parts) == 2 and all(p.strip().isdigit() for p in parts):
                                params[var_name] = float(parts[0]) / float(parts[1])
                    except Exception:
                        continue
            except Exception:
                continue
    return params


def module_namespace(path: Path):
    """Module-level constants a config value may legitimately refer to."""
    ns = {'minimal_search_steps': 20,
          'minimal_init_eps': {'l0': 100, 'l1': 10, 'l2': 1, 'linf': 1 / 255}}
    return ns


def new_values(func: ast.FunctionDef, ns: dict):
    """Values returned by a converted config function (a single `return dict(...)`)."""
    ret = [n for n in func.body if isinstance(n, ast.Return)]
    if len(ret) != 1 or not isinstance(ret[0].value, ast.Call):
        return None
    call = ret[0].value
    if not (isinstance(call.func, ast.Name) and call.func.id == 'dict'):
        return None
    out = {}
    for kw in call.keywords:
        try:
            out[kw.arg] = eval(compile(ast.Expression(kw.value), '<cfg>', 'eval'), {}, ns)
        except Exception as e:  # noqa: BLE001 - reported, not raised
            out[kw.arg] = f'<unevaluable: {e}>'
    return out


def config_functions(source: str):
    """(name, function node, prefix) for every config function in a configs.py source."""
    tree = ast.parse(source)
    prefix = next((n.value.value for n in tree.body
                   if isinstance(n, ast.Assign)
                   and getattr(n.targets[0], 'id', None) == '_prefix'), None)
    if prefix is None:
        return {}
    return {n.name: n for n in tree.body
            if isinstance(n, ast.FunctionDef)
            and n.name.startswith(prefix + '_') and not n.name.startswith('get_')}


def main() -> int:
    ref = sys.argv[1] if len(sys.argv) > 1 else 'HEAD'
    regressions, gains, checked = [], [], 0

    for path in CONFIG_FILES:
        old_source = source_at(ref, path)
        if not old_source:
            print(f'note: {path} does not exist at {ref}, skipping')
            continue
        old_lines = old_source.splitlines()
        old_funcs = config_functions(old_source)
        ns = module_namespace(path)

        for name, func in config_functions(path.read_text()).items():
            if name not in old_funcs:
                continue  # new config, nothing to compare against

            old_func = old_funcs[name]
            # A few configs were already written as `return dict(...)`; for those the
            # honest comparison is dict-to-dict, not against the source-text parser.
            old = (new_values(old_func, ns)
                   if any(isinstance(s, ast.Return) for s in old_func.body)
                   else old_parse(old_lines[old_func.lineno - 1:old_func.end_lineno]))
            new = new_values(func, ns)
            if new is None:
                regressions.append(f'{path}:{name}: not converted (no `return dict(...)`)')
                continue

            checked += 1
            for key, value in old.items():
                if key not in new:
                    regressions.append(f'{path}:{name}: lost {key}={value!r}')
                elif new[key] != value:
                    regressions.append(
                        f'{path}:{name}: {key} changed {value!r} -> {new[key]!r}')
            for key, value in new.items():
                if key not in old:
                    gains.append(f'{path}:{name}: {key}={value!r} '
                                 f'(the old parser dropped this)')

    print(f'checked {checked} config functions in {len(CONFIG_FILES)} files')
    if gains:
        print(f'\n{len(gains)} value(s) recovered — these were silently replaced by '
              f'_get_smart_defaults before:')
        for g in gains:
            print('  GAINED  ' + g)
    if regressions:
        print(f'\n{len(regressions)} REGRESSION(S):')
        for r in regressions:
            print('  MISSING ' + r)
        return 1
    print('\nno regressions: every value the old parser understood is preserved')
    return 0


if __name__ == '__main__':
    sys.exit(main())
