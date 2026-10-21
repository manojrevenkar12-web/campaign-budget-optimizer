"""Creative-fatigue classifier: trained in PyTorch, served in NumPy.

Data scientists train and iterate in PyTorch; the API should not carry a 700 MB framework to run a
two-layer network. The trained weights and the feature scaling are exported as JSON and evaluated
with NumPy; a parity test checks both give the same probabilities.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cbo.fatigue.data import FEATURES


@dataclass
class FatigueModel:
    mean: np.ndarray
    std: np.ndarray
    layers: list[tuple[np.ndarray, np.ndarray]]  # (W, b) per layer; ReLU between, sigmoid at the end
    temperature: float = 1.0
    version: str = "dev"

    def predict(self, x: np.ndarray) -> np.ndarray:
        h = (np.asarray(x, float) - self.mean) / self.std
        for i, (w, b) in enumerate(self.layers):
            h = h @ w.T + b
            if i < len(self.layers) - 1:
                h = np.maximum(h, 0.0)
        z = h[:, 0] / self.temperature
        return 1.0 / (1.0 + np.exp(-z))

    def to_json(self) -> dict:
        return {
            "features": FEATURES,
            "mean": self.mean.tolist(),
            "std": self.std.tolist(),
            "layers": [{"w": w.tolist(), "b": b.tolist()} for w, b in self.layers],
            "temperature": self.temperature,
            "version": self.version,
        }

    @classmethod
    def from_json(cls, d: dict) -> FatigueModel:
        if d["features"] != FEATURES:
            raise ValueError("model was trained on a different feature list")
        return cls(
            np.asarray(d["mean"]),
            np.asarray(d["std"]),
            [(np.asarray(layer["w"]), np.asarray(layer["b"])) for layer in d["layers"]],
            float(d.get("temperature", 1.0)),
            str(d.get("version", "dev")),
        )


def train_torch(
    x: np.ndarray,
    y: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    hidden: int = 32,
    epochs: int = 60,
    seed: int = 0,
) -> FatigueModel:
    import torch
    from torch import nn

    torch.manual_seed(seed)
    mean, std = x.mean(0), x.std(0) + 1e-6
    xt = torch.tensor((x - mean) / std, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    xv = torch.tensor((x_val - mean) / std, dtype=torch.float32)
    yv = torch.tensor(y_val, dtype=torch.float32)
    net = nn.Sequential(
        nn.Linear(x.shape[1], hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1)
    )
    opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    best, best_state, patience = float("inf"), None, 0
    gen = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        net.train()
        perm = torch.randperm(len(xt), generator=gen)
        for i in range(0, len(xt), 512):
            idx = perm[i : i + 512]
            opt.zero_grad()
            loss = loss_fn(net(xt[idx])[:, 0], yt[idx])
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            v = float(loss_fn(net(xv)[:, 0], yv))
        if v < best - 1e-4:
            best, best_state, patience = v, {k: t.clone() for k, t in net.state_dict().items()}, 0
        else:
            patience += 1
            if patience >= 6:
                break
    net.load_state_dict(best_state)
    # temperature scaling on the validation set: probabilities that mean what they say
    with torch.no_grad():
        logits = net(xv)[:, 0]
    t = torch.ones(1, requires_grad=True)
    topt = torch.optim.LBFGS([t], lr=0.1, max_iter=100)

    def closure():
        topt.zero_grad()
        lv = loss_fn(logits / t, yv)
        lv.backward()
        return lv

    topt.step(closure)
    layers = [
        (m.weight.detach().numpy().astype(float), m.bias.detach().numpy().astype(float))
        for m in net
        if isinstance(m, nn.Linear)
    ]
    model = FatigueModel(mean, std, layers, float(t.detach()[0]))
    model._torch = (net, mean, std)  # kept for the parity check, never serialised
    return model


def torch_predict(model: FatigueModel, x: np.ndarray) -> np.ndarray:
    import torch

    net, mean, std = model._torch
    with torch.no_grad():
        z = net(torch.tensor((x - mean) / std, dtype=torch.float32))[:, 0] / model.temperature
    return torch.sigmoid(z).numpy()


def load_champion(out_dir: str = "data") -> FatigueModel | None:
    """The serving side loads the exported file (copied from the registry's champion at deploy
    time), so the API and worker images need neither PyTorch nor MLflow."""
    p = Path(os.environ.get("CBO_FATIGUE_MODEL", f"{out_dir}/fatigue_model.json"))
    return FatigueModel.from_json(json.loads(p.read_text())) if p.exists() else None
