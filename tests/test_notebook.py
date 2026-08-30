"""Keep the published Colab tutorial aligned with the supported 2.x API."""

import ast
import json
from pathlib import Path


NOTEBOOK = Path(__file__).parents[1] / "examples" / "AttackBenchLib.ipynb"


def _load_notebook():
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def test_colab_uses_supported_install_and_attack_paths():
    notebook = _load_notebook()
    source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )

    assert "attackbenchlib[models,attacks]>=2.0.2,<2.1" in source
    assert "attackbenchlib[adv_lib]" not in source
    assert "attackbenchlib[all]" not in source
    assert "fra31/auto-attack" not in source
    assert "adv-lib @ git+https://github.com/jeromerony/adversarial-library" in source
    assert "from attackbench.attacks import apgd as apgd_l2" in source
    assert "get_attack(lib='original', attack='apgd', threat_model='l2'" not in source
    assert "protocol_version" in source
    assert "distance_semantics" in source


def test_colab_python_cells_parse():
    notebook = _load_notebook()

    for index, cell in enumerate(notebook["cells"]):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        python_lines = [
            "pass" if line.lstrip().startswith(("!", "%")) else line
            for line in source.splitlines()
        ]
        ast.parse("\n".join(python_lines), filename=f"cell-{index}")
