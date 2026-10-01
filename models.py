"""Model architectures, selected by name (train.py --arch) and rebuilt from checkpoints.

Checkpoints store "arch" next to "config"; checkpoints from before the GPT
baseline have no "arch" and are BDH models.
"""

import bdh
import gpt

ARCHS = {
    "bdh": (bdh.BDHConfig, bdh.BDH),
    "gpt": (gpt.GPTConfig, gpt.GPT),
}


def build(arch, config):
    """A new model of architecture arch from a config dict."""
    config_cls, model_cls = ARCHS[arch]
    return model_cls(config_cls(**config))


def arch_of(checkpoint):
    return checkpoint.get("arch", "bdh")


def from_checkpoint(checkpoint, **overrides):
    """The model saved in checkpoint, with weights loaded. overrides replace config
    entries, e.g. dropout=0.1."""
    model = build(arch_of(checkpoint), {**checkpoint["config"], **overrides})
    model.load_state_dict(checkpoint["model"])
    return model
