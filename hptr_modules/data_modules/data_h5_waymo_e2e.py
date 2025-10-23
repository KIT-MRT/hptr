# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
from typing import Optional, Dict, Any, Tuple
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
import numpy as np
import h5py


class DatasetBase(Dataset[Dict[str, np.ndarray]]):
    def __init__(self, filepath: str, tensor_size: Dict[str, Tuple]) -> None:
        super().__init__()
        self.tensor_size = tensor_size
        self.filepath = filepath
        with h5py.File(self.filepath, "r", libver="latest", swmr=True) as hf:
            self.dataset_len = int(hf.attrs["data_len"])

    def __len__(self) -> int:
        return self.dataset_len


class DatasetTrain(DatasetBase):

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        idx = np.random.randint(self.dataset_len)
        idx_key = str(idx)
        out_dict = {"episode_idx": idx}
        with h5py.File(self.filepath, "r", libver="latest", swmr=True) as hf:
            for k in self.tensor_size.keys():
                out_dict[k] = np.ascontiguousarray(hf[idx_key][k])
        return out_dict


class DatasetVal(DatasetBase):
    # for validation.h5 and testing.h5
    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        idx_key = str(idx)
        with h5py.File(self.filepath, "r", libver="latest", swmr=True) as hf:
            out_dict = {
                "episode_idx": idx,
                "scenario_id": hf[idx_key].attrs["scenario_id"],
            }
            for k, _size in self.tensor_size.items():
                out_dict[k] = np.ascontiguousarray(hf[idx_key][k])
                if out_dict[k].shape != _size:
                    assert "agent" in k
                    out_dict[k] = np.ones(_size, dtype=out_dict[k].dtype)
        return out_dict


class DataH5womd(LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        filename_train: str = "training",
        filename_val: str = "validation",
        filename_test: str = "testing",
        batch_size: int = 3,
        num_workers: int = 4,
    ) -> None:
        super().__init__()
        self.interactive_challenge = (
            "interactive" in filename_val or "interactive" in filename_test
        )

        self.path_train_h5 = f"{data_dir}/{filename_train}.h5"
        self.path_val_h5 = f"{data_dir}/{filename_val}.h5"
        self.path_test_h5 = f"{data_dir}/{filename_test}.h5"
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.tensor_size_train = {
            # front left camera
            "agent/front_left/img": (1079, 972, 3),  # float32
            "agent/front_left/calib/intr": (9,),  # float32
            "agent/front_left/calib/extr": (16,), # float32
            # front camera
            "agent/front/img": (1079, 972, 3),  # float32
            "agent/front/calib/intr": (9,),  # float32
            "agent/front/calib/extr": (16,), # float32
            # front right camera
            "agent/front_right/img": (1079, 972, 3),  # float32
            "agent/front_right/calib/intr": (9,),  # float32
            "agent/front_right/calib/extr": (16,), # float32
            # side left camera
            "agent/side_left/img": (1079, 972, 3),  # float32
            "agent/side_left/calib/intr": (9,),  # float32
            "agent/side_left/calib/extr": (16,), # float32
            # side right camera
            "agent/side_right/img": (1079, 972, 3),  # float32
            "agent/side_right/calib/intr": (9,),  # float32
            "agent/side_right/calib/extr": (16,), # float32
            # rear left camera
            "agent/rear_left/img": (587, 972, 3),  # float32
            "agent/rear_left/calib/intr": (9,),  # float32
            "agent/rear_left/calib/extr": (16,), # float32
            # rear camera
            "agent/rear/img": (551, 972, 3),  # float32
            "agent/rear/calib/intr": (9,),  # float32
            "agent/rear/calib/extr": (16,), # float32
            # rear right camera
            "agent/rear_right/img": (587, 972, 3),  # float32
            "agent/rear_right/calib/intr": (9,),  # float32
            "agent/rear_right/calib/extr": (16,), # float32
            # intent
            "agent/intent": (1,), # float32
            # current velocity
            "agent/vel": (6,), # float32
            # history
            "history/agent/pose": (16,), # float32
            "history/agent/pos": (16, 2), # float32
            "history/agent/vel": (16, 2), # float32
            "history/agent/acc": (16, 2), # float32
            # total pos
            "agent/pos": (36, 2),
            # ground truth pos
            "gt/pos": (20, 3),
        }

        self.tensor_size_test = {
            # front left camera
            "agent/front_left/img": (1079, 972, 3),  # float32
            "agent/front_left/calib/intr": (9,),  # float32
            "agent/front_left/calib/extr": (16,), # float32
            # front camera
            "agent/front/img": (1079, 972, 3),  # float32
            "agent/front/calib/intr": (9,),  # float32
            "agent/front/calib/extr": (16,), # float32
            # front right camera
            "agent/front_right/img": (1079, 972, 3),  # float32
            "agent/front_right/calib/intr": (9,),  # float32
            "agent/front_right/calib/extr": (16,), # float32
            # side left camera
            "agent/side_left/img": (1079, 972, 3),  # float32
            "agent/side_left/calib/intr": (9,),  # float32
            "agent/side_left/calib/extr": (16,), # float32
            # side right camera
            "agent/side_right/img": (1079, 972, 3),  # float32
            "agent/side_right/calib/intr": (9,),  # float32
            "agent/side_right/calib/extr": (16,), # float32
            # rear left camera
            "agent/rear_left/img": (587, 972, 3),  # float32
            "agent/rear_left/calib/intr": (9,),  # float32
            "agent/rear_left/calib/extr": (16,), # float32
            # rear camera
            "agent/rear/img": (551, 972, 3),  # float32
            "agent/rear/calib/intr": (9,),  # float32
            "agent/rear/calib/extr": (16,), # float32
            # rear right camera
            "agent/rear_right/img": (587, 972, 3),  # float32
            "agent/rear_right/calib/intr": (9,),  # float32
            "agent/rear_right/calib/extr": (16,), # float32
            # intent
            "agent/intent": (1,), # float32
            # current velocity
            "agent/vel": (6,), # float32
            # history
            "history/agent/pose": (16,), # float32
            "history/agent/pos": (16, 2), # float32
            "history/agent/vel": (16, 2), # float32
            "history/agent/acc": (16, 2), # float32
        }

        self.tensor_size_val = {
            "gt/preference_scores": (3,),
        }

        self.tensor_size_val = (
            self.tensor_size_val | self.tensor_size_train | self.tensor_size_test
        )

    def setup(self, stage: Optional[str] = None) -> None:
        if stage == "fit" or stage is None:
            self.train_dataset = DatasetTrain(
                self.path_train_h5, self.tensor_size_train
            )
            self.val_dataset = DatasetVal(self.path_val_h5, self.tensor_size_val)
        elif stage == "validate":
            self.val_dataset = DatasetVal(self.path_val_h5, self.tensor_size_val)
        elif stage == "test":
            self.test_dataset = DatasetVal(self.path_test_h5, self.tensor_size_test)
        elif stage == "predict":
            self.val_dataset = DatasetVal(self.path_val_h5, self.tensor_size_val)

    def train_dataloader(self) -> DataLoader[Any]:
        return self._get_dataloader(
            self.train_dataset, self.batch_size, self.num_workers
        )

    def val_dataloader(self) -> DataLoader[Any]:
        return self._get_dataloader(self.val_dataset, self.batch_size, self.num_workers)

    def test_dataloader(self) -> DataLoader[Any]:
        return self._get_dataloader(
            self.test_dataset, self.batch_size, self.num_workers
        )

    def predict_dataloader(self) -> DataLoader[Any]:
        return self._get_dataloader(self.val_dataset, self.batch_size, self.num_workers)

    @staticmethod
    def _get_dataloader(
        ds: Dataset, batch_size: int, num_workers: int
    ) -> DataLoader[Any]:
        return DataLoader(
            ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True,
            shuffle=False,
            drop_last=False,
            persistent_workers=True,
        )
