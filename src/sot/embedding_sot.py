"""Sentence-embedding variant of the State-of-Thought signal.

An immutable PCA projection maps sentence embeddings to a compact space.  The
same geometric quantities used by the hidden-state implementation are then
computed in that space, keeping the rest of the SoT runtime unchanged.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor

from .config import DebtConfig, NormalizationConfig
from .logic_state import LogicSignature, build_logic_state, compute_entropy
from .state_extractor import SoTState

# ---------------------------------------------------------------------------
# PCA + OpenAI
# ---------------------------------------------------------------------------


@dataclass
class PcaModel:
    mean: np.ndarray  # (D,)
    components: np.ndarray  # (K, D)
    openai_model: str = "text-embedding-3-small"
    n_components: int = 32
    dim_in: int = 1536

    def transform(self, x: np.ndarray) -> np.ndarray:
        """x: (D,) or (N,D) -> (K,) or (N,K)"""
        one = x.ndim == 1
        if one:
            x = x[None, :]
        z = (x - self.mean[None, :]) @ self.components.T
        return z[0] if one else z

    def to_dict(self) -> Dict[str, Any]:
        return {
            "kind": "mot_openai_pca_v1",
            "openai_model": self.openai_model,
            "n_components": int(self.n_components),
            "dim_in": int(self.dim_in),
            "mean": self.mean.astype(np.float64).tolist(),
            "components": self.components.astype(np.float64).tolist(),
        }

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "PcaModel":
        mean = np.asarray(d["mean"], dtype=np.float64)
        comp = np.asarray(d["components"], dtype=np.float64)
        return PcaModel(
            mean=mean,
            components=comp,
            openai_model=str(d.get("openai_model") or "text-embedding-3-small"),
            n_components=int(comp.shape[0]),
            dim_in=int(mean.shape[0]),
        )

    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> "PcaModel":
        p = Path(path)
        return PcaModel.from_dict(json.loads(p.read_text(encoding="utf-8")))


def fit_pca_on_matrix(x: np.ndarray, n_components: int = 32) -> PcaModel:
    """x: (N, D) full embedding matrix."""
    from sklearn.decomposition import PCA  # type: ignore

    n = int(x.shape[0])
    k = min(int(n_components), n, int(x.shape[1]))
    k = max(1, k)
    p = PCA(n_components=k, svd_solver="randomized" if n > 4096 else "auto", random_state=0)
    p.fit(x)
    return PcaModel(
        mean=p.mean_.astype(np.float64),
        components=p.components_.astype(np.float64),
        openai_model="",
        n_components=k,
        dim_in=int(x.shape[1]),
    )


def _chunk_pieces(text: str) -> List[str]:
    t = (text or "").strip()
    if not t:
        return [" "]
    words = t.split()
    if len(words) >= 2:
        mid = max(1, len(words) // 2)
        a = " ".join(words[:mid])
        b = " ".join(words[mid:])
        return [a, b] if a and b else [t]
    n = len(t)
    if n < 2:
        return [t]
    mid = n // 2
    return [t[:mid], t[mid:]]


def _np_pad_vec(v: np.ndarray, target: int, *, pad_value: float = 0.0) -> np.ndarray:
    v = v.astype(np.float64).ravel()
    t = max(1, int(target))
    if int(v.size) >= t:
        return v[:t]
    pad = np.full((t - int(v.size),), float(pad_value), dtype=np.float64)
    return np.concatenate([v, pad], axis=0)


def _spectral_entropy(z: np.ndarray) -> float:
    z = z.astype(np.float64).ravel()
    a = np.abs(z)
    s = float(a.sum() + 1e-8)
    p = a / s + 1e-12
    return float(-(p * np.log(p + 1e-15)).sum())


def compute_logic_signature_pca_lite(
    *,
    z_t: np.ndarray,
    z_prev: Optional[np.ndarray],
    z_prev_prev: Optional[np.ndarray],
    delta: float,
    h_alt: Optional[float] = None,
) -> LogicSignature:
    H = float(h_alt) if h_alt is not None else _spectral_entropy(z_t)
    if z_prev is None:
        return LogicSignature(delta=float(delta), v=0.0, c=1.0, H=H)
    dz = z_t - z_prev
    v = float(np.linalg.norm(dz) + 1e-8)
    if z_prev_prev is None:
        return LogicSignature(delta=float(delta), v=v, c=1.0, H=H)
    dz_prev = z_prev - z_prev_prev
    num = float(np.dot(dz, dz_prev))
    den = float((np.linalg.norm(dz) * np.linalg.norm(dz_prev)) + 1e-8)
    c = num / den if den > 0 else 1.0
    return LogicSignature(delta=float(delta), v=v, c=float(c), H=H)


class _EmbedRuntime:
    def __init__(
        self,
        *,
        pca: PcaModel,
        openai_model: str,
        h_source: str = "spectral",
        m_source: str = "geometry4",
    ) -> None:
        self.pca = pca
        self.openai_model = str(openai_model or pca.openai_model or "text-embedding-3-small").strip() or "text-embedding-3-small"
        self.h_source = str(h_source or "spectral").strip().lower()
        self.m_source = str(m_source or "geometry4").strip().lower()
        self._cache: Dict[str, np.ndarray] = {}
        self._lock = threading.Lock()
        self._client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            try:
                from openai import OpenAI  # type: ignore
            except Exception as e:  # noqa: BLE001
                raise RuntimeError("openai package required: pip install openai") from e
            self._client = OpenAI()
        return self._client

    def embed_text(self, text: str) -> np.ndarray:
        key = (text or "")[:20000]
        with self._lock:
            hit = self._cache.get(key)
        if hit is not None:
            return hit
        v = self.embed_batch([key])[0]
        with self._lock:
            if len(self._cache) < 100_000:
                self._cache[key] = v
        return v

    def embed_batch(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.pca.dim_in), dtype=np.float64)
        client = self._get_client()
        out_rows: List[List[float]] = []
        batch = 64
        for i in range(0, len(texts), batch):
            chunk = [str(t or "") for t in texts[i : i + batch]]
            resp = client.embeddings.create(model=self.openai_model, input=chunk)
            for j in range(len(chunk)):
                out_rows.append(list(resp.data[j].embedding))
        return np.asarray(out_rows, dtype=np.float64)

    def z_and_delta(
        self,
        sentence: str,
    ) -> Tuple[np.ndarray, float, np.ndarray]:
        """Returns z_pca (K,), delta scalar, full embedding of sentence (D,) for cache bookkeeping."""
        full = self.embed_text(sentence)
        pieces = _chunk_pieces(sentence)
        if len(pieces) >= 2:
            raw = self.embed_batch(pieces)  # (2+, D)
            pca_pieces = self.pca.transform(raw)  # (2+, K)
            ctr = pca_pieces.mean(axis=0)
            dlt = float(np.mean((pca_pieces - ctr[None, :]) ** 2))
        else:
            dlt = 0.0
        zt = self.pca.transform(full)
        return zt.astype(np.float64), dlt, full.astype(np.float64)

    def to_sot_state(
        self,
        *,
        sentence: str,
        logits_t: Optional[Tensor],
        z_prev: Optional[Tensor],
        z_prev_prev: Optional[Tensor],
        prev_e: float,
        norm_cfg: NormalizationConfig,
        debt_cfg: DebtConfig,
    ) -> SoTState:
        z_np, delta, _full = self.z_and_delta(sentence)
        zp = z_np
        zpv = z_prev.detach().float().cpu().numpy() if z_prev is not None else None
        zppv = z_prev_prev.detach().float().cpu().numpy() if z_prev_prev is not None else None

        dev = (
            z_prev.device
            if z_prev is not None
            else (logits_t.device if logits_t is not None else torch.device("cpu"))
        )
        m_src = self.m_source
        if m_src in {"raw_pca", "raw32", "pca32"}:
            # Use the K-dimensional PCA embedding directly for gate/stop state;
            # it is distinct from the four-dimensional geometric signature.
            mu = _np_pad_vec(np.asarray(norm_cfg.mu, dtype=np.float64), int(zp.size))
            sigm = _np_pad_vec(np.asarray(norm_cfg.sigma, dtype=np.float64), int(zp.size), pad_value=1.0)
            m_np = (zp - mu) / (sigm + float(norm_cfg.eps))
            h_mode = self.h_source
            h_alt: Optional[float] = None
            if h_mode == "logits" and logits_t is not None and logits_t.dim() == 2 and logits_t.shape[0] > 0:
                h_alt = compute_entropy(logits_t[-1, :])
            sig = compute_logic_signature_pca_lite(
                z_t=zp, z_prev=zpv, z_prev_prev=zppv, delta=delta, h_alt=h_alt
            )
            z_t_t = torch.tensor(zp, dtype=torch.float32, device=dev)
            m_t = torch.tensor(m_np, dtype=torch.float32, device=dev)
            # Calibrate the geometric signature from its own four coordinates.
            # Do not reuse the higher-dimensional embedding-state tensor.
            st = build_logic_state(
                sig=sig, prev_e=prev_e, norm_cfg=NormalizationConfig(), debt_cfg=debt_cfg, device=dev
            )
            return SoTState(z=z_t_t, signature=sig, m=m_t, e=st.e, e_bar=st.e_bar)

        h_mode = self.h_source
        h_alt: Optional[float] = None
        if h_mode == "logits" and logits_t is not None and logits_t.dim() == 2 and logits_t.shape[0] > 0:
            h_alt = compute_entropy(logits_t[-1, :])
        # Spectral and unrecognized modes use the default entropy branch.

        sig = compute_logic_signature_pca_lite(
            z_t=zp, z_prev=zpv, z_prev_prev=zppv, delta=delta, h_alt=h_alt
        )
        st = build_logic_state(sig=sig, prev_e=prev_e, norm_cfg=norm_cfg, debt_cfg=debt_cfg, device=dev)
        z_t_t = torch.tensor(zp, dtype=torch.float32, device=dev)
        return SoTState(z=z_t_t, signature=sig, m=st.m, e=st.e, e_bar=st.e_bar)


_RUNTIME_LOCK = threading.Lock()
_RUNTIME_CACHE: Dict[str, _EmbedRuntime] = {}


def get_embed_runtime(
    pca_path: str,
    openai_model: str,
    *,
    h_source: str = "spectral",
    m_source: str = "geometry4",
) -> _EmbedRuntime:
    key = f"{Path(pca_path).resolve()}|{openai_model}|{h_source}|{m_source}"
    with _RUNTIME_LOCK:
        if key in _RUNTIME_CACHE:
            return _RUNTIME_CACHE[key]
        pca = PcaModel.load(pca_path)
        m = str(openai_model or "").strip() or pca.openai_model or "text-embedding-3-small"
        rt = _EmbedRuntime(
            pca=pca,
            openai_model=m,
            h_source=h_source,
            m_source=m_source,
        )
        _RUNTIME_CACHE[key] = rt
        return rt


def clear_embed_runtime_cache() -> None:
    with _RUNTIME_LOCK:
        _RUNTIME_CACHE.clear()


def embed_texts_for_pca_fit(texts: List[str], openai_model: str) -> np.ndarray:
    """Fit PCA from one batched embedding matrix without using the PCA cache."""
    if not texts:
        return np.zeros((0, 0), dtype=np.float64)
    try:
        from openai import OpenAI  # type: ignore
    except Exception as e:  # noqa: BLE001
        raise RuntimeError("openai package required: pip install openai") from e
    client = OpenAI()
    model = str(openai_model or "text-embedding-3-small").strip()
    out_rows: List[List[float]] = []
    n = 64
    for i in range(0, len(texts), n):
        chunk = [str(t or "") for t in texts[i : i + n]]
        resp = client.embeddings.create(model=model, input=chunk)
        for j in range(len(chunk)):
            out_rows.append(list(resp.data[j].embedding))
    return np.asarray(out_rows, dtype=np.float64)


__all__ = [
    "PcaModel",
    "fit_pca_on_matrix",
    "embed_texts_for_pca_fit",
    "get_embed_runtime",
    "clear_embed_runtime_cache",
]
