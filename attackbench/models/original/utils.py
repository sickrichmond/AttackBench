from .zhang2020.crown import load_crown_model

_available_defenses = {
    "Zhang2020CrownLarge": load_crown_model,
    "Zhang2020CrownSmall": load_crown_model,
}


def load_original_model(model_name: str, dataset: str, threat_model: str):
    try:
        loader = _available_defenses[model_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown bundled model {model_name!r}. Available: "
            f"{sorted(_available_defenses)}"
        ) from exc
    return loader(model=model_name, dataset=dataset, threat_model=threat_model)
