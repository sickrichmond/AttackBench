"""Release acceptance-runner regressions."""

from types import SimpleNamespace

import attackbench.preconfigured as preconfigured
from attackbench import acceptance


def test_minimum_asr_is_opt_in():
    args = acceptance.parse_args(
        [
            "--model",
            "mnist_smallcnn",
            "--dataset",
            "mnist",
            "--threat-model",
            "l2",
            "--attacks",
            "preconfigured:fmn",
        ]
    )

    assert args.minimum_asr is None


def test_fmn_uses_norm_specific_step_size():
    recorded = []

    def fake_getter(**kwargs):
        recorded.append(kwargs)
        return lambda **attack_kwargs: attack_kwargs["inputs"]
    # Replace the getter captured by the public wrapper with an equivalent wrapper
    # whose configuration can be inspected without running Foolbox.
    configured = preconfigured._minimum_norm(
        fake_getter,
        "fmn",
        norm_params={"linf": {"max_step_size": 10}},
        num_steps=1000,
        max_step_size=1,
        gamma=0.05,
    )

    marker = object()
    configured(None, marker, None, threat_model="l2")
    configured(None, marker, None, threat_model="linf")

    assert recorded[0]["max_step_size"] == 1
    assert recorded[1]["max_step_size"] == 10


def test_minimum_asr_failure_is_reported(monkeypatch, capsys):
    args = SimpleNamespace(
        model="toy",
        dataset="mnist",
        threat_model="l2",
        attacks=["preconfigured:fmn"],
        n_samples=1,
        batch_size=1,
        seed=0,
        query_budget=2000,
        reference="ensemble",
        minimum_asr=90.0,
        device="cpu",
        data_root="data",
        cache_dir="cache",
    )
    result = {
        "distances": {"l2": [float("inf")]},
        "final_distances": {"l2": [float("inf")]},
        "adv_success": [False],
        "ori_success": [False],
        "correct": [True],
        "num_forwards": [1],
        "num_backwards": [0],
        "times": [0.0],
        "batch_failures": [False],
        "box_failures": [False],
        "metadata": {"model_name": "toy"},
    }

    monkeypatch.setattr(acceptance, "parse_args", lambda argv=None: args)
    monkeypatch.setattr(acceptance.attackbench, "get_loader", lambda **kwargs: object())
    monkeypatch.setattr(acceptance.attackbench, "get_model", lambda name: object())
    monkeypatch.setattr(
        acceptance, "build_attack", lambda spec, threat_model: ("fmn-original", object())
    )
    monkeypatch.setattr(acceptance.attackbench, "run_attack", lambda *args, **kwargs: result)

    assert acceptance.main([]) == 1
    assert "ASR 0.0% < required 90.0%" in capsys.readouterr().out
