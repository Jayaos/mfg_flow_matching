import os
import numpy as np
import torch
from torch.utils.data import Dataset


class _H5Images(Dataset):
    """Minimal HDF5 image reader (expects images under key 'imgs')."""
    def __init__(self, hdf_path: str, indices=None):
        import h5py
        self._path = os.fspath(hdf_path)
        self._file = None
        self._ds = None

        with h5py.File(self._path, "r") as f:
            if "imgs" not in f:
                raise KeyError(f"Key 'imgs' not found in {self._path}. Available: {list(f.keys())}")
            base_len = len(f["imgs"])

        if indices is None:
            self._indices = None
            self._length = base_len
        else:
            self._indices = np.asarray(indices, dtype=np.int64)
            self._length = int(self._indices.shape[0])

    def _valid(self, h5obj) -> bool:
        return hasattr(h5obj, "id") and getattr(h5obj.id, "valid", False)

    def _ensure_open(self):
        import h5py
        if self._file is None or not self._valid(self._file):
            self._file = h5py.File(self._path, "r", swmr=True, libver="latest")
            self._ds = None
        if self._ds is None or not self._valid(self._ds):
            self._ds = self._file["imgs"]

    def __len__(self):
        return self._length

    def __getitem__(self, idx: int):
        self._ensure_open()
        j = self._indices[idx] if self._indices is not None else idx
        arr = self._ds[j]  # uint8, (H,W,C) or (C,H,W)

        if arr.ndim != 3:
            raise ValueError(f"Expected 3D image, got {arr.shape} at idx={idx}")
        if arr.shape[0] in (1, 3):  # (C,H,W) -> (H,W,C)
            arr = arr.transpose(1, 2, 0)

        img = torch.from_numpy(arr.astype("uint8")).permute(2, 0, 1).float() / 255.0
        img = img * 2.0 - 1.0  # [-1,1]
        return {"image": img}


class ShoesBags(Dataset):
    """
    Concatenate two HDF5 sets (shoes + bags).
    Use `seed` + `*_frac` to pick a deterministic subset; set `complement=True`
    to get the remainder (with the same seed and frac).

    Args:
      shoes_path: str
      bags_path:  str
      shoes_frac: float in (0,1]
      bags_frac:  float in (0,1]
      seed:       int
      complement: bool  # False => take first k; True => take the rest n-k
    """
    def __init__(
        self,
        shoes_path: str,
        bags_path: str,
        shoes_frac: float = 1.0,
        bags_frac: float = 1.0,
        seed: int = 0,
        complement: bool = False,
    ):
        import h5py

        # probe sizes
        with h5py.File(os.fspath(shoes_path), "r") as f:
            if "imgs" not in f: raise KeyError(f"'imgs' not in {shoes_path}. Keys={list(f.keys())}")
            n_shoes = len(f["imgs"])
        with h5py.File(os.fspath(bags_path), "r") as f:
            if "imgs" not in f: raise KeyError(f"'imgs' not in {bags_path}. Keys={list(f.keys())}")
            n_bags = len(f["imgs"])

        # deterministic permutation per source (seed offsets keep streams distinct)
        def split_indices(n, frac, seed_plus, complement_flag):
            if not (0.0 < frac <= 1.0):
                raise ValueError(f"fraction must be in (0,1], got {frac}")
            k = max(1, int(np.floor(n * frac)))
            rng = np.random.RandomState(seed_plus)
            perm = rng.permutation(n)
            chosen = perm[:k]
            rest = perm[k:]
            return rest if complement_flag else chosen

        shoes_idx = split_indices(n_shoes, shoes_frac, seed + 17, complement)
        bags_idx  = split_indices(n_bags,  bags_frac,  seed + 71, complement)

        # stash for reproducibility/debug if you want to inspect later
        self.selected_indices_shoes = shoes_idx
        self.selected_indices_bags  = bags_idx

        self.shoes = _H5Images(shoes_path, indices=shoes_idx)
        self.bags  = _H5Images(bags_path,  indices=bags_idx)

        self._len_shoes = len(self.shoes)
        self._len_bags = len(self.bags)
        self._length = self._len_shoes + self._len_bags

    def __len__(self):
        return self._length

    def __getitem__(self, idx: int):
        if idx < self._len_shoes:
            return self.shoes[idx]
        return self.bags[idx - self._len_shoes]
    

class Shoes(Dataset):
    """
    Concatenate two HDF5 sets (shoes + bags).
    Use `seed` + `*_frac` to pick a deterministic subset; set `complement=True`
    to get the remainder (with the same seed and frac).

    Args:
      shoes_path: str
      shoes_frac: float in (0,1]
      bags_frac:  float in (0,1]
      seed:       int
      complement: bool  # False => take first k; True => take the rest n-k
    """
    def __init__(
        self,
        shoes_path: str,
        shoes_frac: float = 1.0,
        seed: int = 0,
        complement: bool = False,):
        import h5py

        # probe sizes
        with h5py.File(os.fspath(shoes_path), "r") as f:
            if "imgs" not in f: raise KeyError(f"'imgs' not in {shoes_path}. Keys={list(f.keys())}")
            n_shoes = len(f["imgs"])

        # deterministic permutation per source (seed offsets keep streams distinct)
        def split_indices(n, frac, seed_plus, complement_flag):
            if not (0.0 < frac <= 1.0):
                raise ValueError(f"fraction must be in (0,1], got {frac}")
            k = max(1, int(np.floor(n * frac)))
            rng = np.random.RandomState(seed_plus)
            perm = rng.permutation(n)
            chosen = perm[:k]
            rest = perm[k:]
            return rest if complement_flag else chosen

        shoes_idx = split_indices(n_shoes, shoes_frac, seed + 17, complement)

        # stash for reproducibility/debug if you want to inspect later
        self.selected_indices_shoes = shoes_idx

        self.shoes = _H5Images(shoes_path, indices=shoes_idx)

        self._len_shoes = len(self.shoes)
        self._length = self._len_shoes

    def __len__(self):
        return self._length

    def __getitem__(self, idx: int):
        return self.shoes[idx]


class Bags(Dataset):
    """
    Concatenate two HDF5 sets (shoes + bags).
    Use `seed` + `*_frac` to pick a deterministic subset; set `complement=True`
    to get the remainder (with the same seed and frac).

    Args:
      bags_path:  str
      bags_frac:  float in (0,1]
      seed:       int
      complement: bool  # False => take first k; True => take the rest n-k
    """
    def __init__(
        self,
        bags_path: str,
        bags_frac: float = 1.0,
        seed: int = 0,
        complement: bool = False,
    ):
        import h5py

        # probe sizes
        with h5py.File(os.fspath(bags_path), "r") as f:
            if "imgs" not in f: raise KeyError(f"'imgs' not in {bags_path}. Keys={list(f.keys())}")
            n_bags = len(f["imgs"])

        # deterministic permutation per source (seed offsets keep streams distinct)
        def split_indices(n, frac, seed_plus, complement_flag):
            if not (0.0 < frac <= 1.0):
                raise ValueError(f"fraction must be in (0,1], got {frac}")
            k = max(1, int(np.floor(n * frac)))
            rng = np.random.RandomState(seed_plus)
            perm = rng.permutation(n)
            chosen = perm[:k]
            rest = perm[k:]
            return rest if complement_flag else chosen

        bags_idx  = split_indices(n_bags,  bags_frac,  seed + 71, complement)

        # stash for reproducibility/debug if you want to inspect later
        self.selected_indices_bags  = bags_idx
        self.bags  = _H5Images(bags_path,  indices=bags_idx)

        self._len_bags = len(self.bags)
        self._length = self._len_bags

    def __len__(self):
        return self._length

    def __getitem__(self, idx: int):
        return self.bags[idx]
