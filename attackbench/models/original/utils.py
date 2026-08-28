from typing import Optional

from .external import load_stutz2020, load_xiao2020
from .zhang2020.crown import load_crown_model

_available_defenses = {
    "Stutz2020CCAT": load_stutz2020,
    "Xiao2020KWTA": load_xiao2020,
    "Zhang2020CrownLarge": load_crown_model,
    "Zhang2020CrownSmall": load_crown_model,
}

_external_defenses = {"Stutz2020CCAT", "Xiao2020KWTA"}


def load_original_model(
    model_name: str,
    dataset: str,
    threat_model: str,
    checkpoint_path: Optional[str] = None,
    accept_license: bool = False,
):
    try:
        loader = _available_defenses[model_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown bundled model {model_name!r}. Available: "
            f"{sorted(_available_defenses)}"
        ) from exc
    if model_name in _external_defenses:
        return loader(
            model=model_name,
            dataset=dataset,
            threat_model=threat_model,
            checkpoint_path=checkpoint_path,
            accept_license=accept_license,
        )
    if checkpoint_path is not None:
        raise ValueError(f"{model_name} does not accept checkpoint_path")
    return loader(model=model_name, dataset=dataset, threat_model=threat_model)
