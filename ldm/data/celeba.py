import os
from typing import Optional, Sequence, Dict, Any
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from PIL import Image

__all__ = [
    "CelebAAll", "CelebATrain", "CelebAValid", "CelebATest",
    "CelebATrainMale", "CelebATrainFemale",
    "CelebAValidMale", "CelebAValidFemale",
    "CelebATestMale", "CelebATestFemale",
]


def _build_transform(
    size: int = 256,
    center_crop: bool = True,
    random_resized_crop: bool = False,
) -> transforms.Compose:
    """
    Build transforms to produce CHW tensors normalized to [-1, 1].
    CelebA original size is 178x218; we typically center-crop to a square first.
    """
    t = []
    if center_crop:
        t.append(transforms.CenterCrop(178))  # 178x178

    if random_resized_crop:
        # small jitter; you can widen scale/ratio if you want stronger aug
        t.append(
            transforms.RandomResizedCrop(
                size, scale=(0.9, 1.0), ratio=(0.95, 1.05), interpolation=Image.BICUBIC
            )
        )
    else:
        t.append(transforms.Resize(size, interpolation=Image.BICUBIC))

    t.extend(
        [
            transforms.ToTensor(),                              # [0, 1]
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # [-1, 1]
        ]
    )
    return transforms.Compose(t)


class _CelebABase(Dataset):
    """
    Minimal wrapper around torchvision.datasets.CelebA.

    Returns:
        dict(image=tensor in [-1, 1], CxHxW)

    Notes:
      - Set `download=True` if you want torchvision to fetch the files.
      - If your model expects a different key (e.g. 'jpg' or 'x'), change the return dict below.
    """

    def __init__(
        self,
        root: str = "data/celeba",
        split: str = "train",                 # 'train' | 'valid' | 'test' | 'all'
        size: int = 256,
        center_crop: bool = True,
        random_resized_crop: bool = False,
        download: bool = False,
        subset_indices: Optional[Sequence[int]] = None,
        return_key: str = "image",
    ):
        super().__init__()
        self.return_key = return_key
        self.transform = _build_transform(
            size=size,
            center_crop=center_crop,
            random_resized_crop=random_resized_crop,
        )

        # target_type='attr' keeps attributes available if you decide to expose them later.
        self.ds = datasets.CelebA(
            root=root,
            split=split,
            target_type="attr",
            transform=self.transform,
            download=download,
        )

        self.indices = list(subset_indices) if subset_indices is not None else None

    def __len__(self) -> int:
        return len(self.indices) if self.indices is not None else len(self.ds)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.indices is not None:
            idx = self.indices[idx]
        img, attr = self.ds[idx]   # ignore attrs by default
        return {self.return_key: img}


class CelebAAll(_CelebABase):
    """All CelebA images combined (train+valid+test)."""
    def __init__(self, **kwargs):
        kwargs.setdefault("split", "all")
        super().__init__(**kwargs)


class CelebATrain(_CelebABase):
    """Train split of CelebA."""
    def __init__(self, **kwargs):
        kwargs.setdefault("split", "train")
        super().__init__(**kwargs)


class CelebAValid(_CelebABase):
    """Validation split of CelebA (official 'valid' partition)."""
    def __init__(self, **kwargs):
        kwargs.setdefault("split", "valid")
        super().__init__(**kwargs)


class CelebATest(_CelebABase):
    """Test split of CelebA."""
    def __init__(self, **kwargs):
        kwargs.setdefault("split", "test")
        super().__init__(**kwargs)


class CelebATrainMale(_CelebABase):
    """Train split of CelebA restricted to male images only."""
    MALE_IDX = 20

    def __init__(self, **kwargs):
        kwargs.setdefault("split", "train")
        super().__init__(**kwargs)

        # build male-only indices
        if hasattr(self.ds, "attr") and isinstance(self.ds.attr, torch.Tensor):
            mask = self.ds.attr[:, self.MALE_IDX] > 0   # +1 = male, -1 = female
            all_indices = torch.arange(len(self.ds))[mask].tolist()
        else:
            # fallback: iterate
            all_indices = []
            for i in range(len(self.ds)):
                _, attr = self.ds[i]
                if attr[self.MALE_IDX] > 0:
                    all_indices.append(i)

        # if a subset_indices was passed, intersect with male-only
        if self.indices is not None:
            self.indices = [i for i in self.indices if i in set(all_indices)]
        else:
            self.indices = all_indices


class CelebATrainFemale(_CelebABase):
    """Train split of CelebA restricted to female images only."""
    MALE_IDX = 20

    def __init__(self, **kwargs):
        kwargs.setdefault("split", "train")
        super().__init__(**kwargs)

        if hasattr(self.ds, "attr") and isinstance(self.ds.attr, torch.Tensor):
            mask = self.ds.attr[:, self.MALE_IDX] <= 0   # -1 = female
            all_indices = torch.arange(len(self.ds))[mask].tolist()
        else:
            all_indices = []
            for i in range(len(self.ds)):
                _, attr = self.ds[i]
                if attr[self.MALE_IDX] <= 0:
                    all_indices.append(i)

        if self.indices is not None:
            self.indices = [i for i in self.indices if i in set(all_indices)]
        else:
            self.indices = all_indices


class CelebAValidMale(_CelebABase):
    """Validation split of CelebA restricted to male images only."""
    MALE_IDX = 20

    def __init__(self, **kwargs):
        kwargs.setdefault("split", "valid")
        super().__init__(**kwargs)

        if hasattr(self.ds, "attr") and isinstance(self.ds.attr, torch.Tensor):
            mask = self.ds.attr[:, self.MALE_IDX] > 0
            all_indices = torch.arange(len(self.ds))[mask].tolist()
        else:
            all_indices = []
            for i in range(len(self.ds)):
                _, attr = self.ds[i]
                if attr[self.MALE_IDX] > 0:
                    all_indices.append(i)

        if self.indices is not None:
            self.indices = [i for i in self.indices if i in set(all_indices)]
        else:
            self.indices = all_indices


class CelebAValidFemale(_CelebABase):
    """Validation split of CelebA restricted to female images only."""
    MALE_IDX = 20

    def __init__(self, **kwargs):
        kwargs.setdefault("split", "valid")
        super().__init__(**kwargs)

        if hasattr(self.ds, "attr") and isinstance(self.ds.attr, torch.Tensor):
            mask = self.ds.attr[:, self.MALE_IDX] <= 0
            all_indices = torch.arange(len(self.ds))[mask].tolist()
        else:
            all_indices = []
            for i in range(len(self.ds)):
                _, attr = self.ds[i]
                if attr[self.MALE_IDX] <= 0:
                    all_indices.append(i)

        if self.indices is not None:
            self.indices = [i for i in self.indices if i in set(all_indices)]
        else:
            self.indices = all_indices


class CelebATestMale(_CelebABase):
    """Test split of CelebA restricted to male images only."""
    MALE_IDX = 20

    def __init__(self, **kwargs):
        kwargs.setdefault("split", "test")
        super().__init__(**kwargs)

        if hasattr(self.ds, "attr") and isinstance(self.ds.attr, torch.Tensor):
            mask = self.ds.attr[:, self.MALE_IDX] > 0
            all_indices = torch.arange(len(self.ds))[mask].tolist()
        else:
            all_indices = []
            for i in range(len(self.ds)):
                _, attr = self.ds[i]
                if attr[self.MALE_IDX] > 0:
                    all_indices.append(i)

        if self.indices is not None:
            self.indices = [i for i in self.indices if i in set(all_indices)]
        else:
            self.indices = all_indices


class CelebATestFemale(_CelebABase):
    """Test split of CelebA restricted to female images only."""
    MALE_IDX = 20

    def __init__(self, **kwargs):
        kwargs.setdefault("split", "test")
        super().__init__(**kwargs)

        if hasattr(self.ds, "attr") and isinstance(self.ds.attr, torch.Tensor):
            mask = self.ds.attr[:, self.MALE_IDX] <= 0
            all_indices = torch.arange(len(self.ds))[mask].tolist()
        else:
            all_indices = []
            for i in range(len(self.ds)):
                _, attr = self.ds[i]
                if attr[self.MALE_IDX] <= 0:
                    all_indices.append(i)

        if self.indices is not None:
            self.indices = [i for i in self.indices if i in set(all_indices)]
        else:
            self.indices = all_indices