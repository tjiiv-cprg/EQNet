# Copyright (c) 2023, Zikang Zhou. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# MODIFICATION NOTES:
#  - add NO_TEST_DATA flag to control whether the test data should be loaded,
#    this is only because I didn't upload the full data to some machines.

from typing import Callable, Optional

import os
import pytorch_lightning as pl
from torch.utils.data import DataLoader as PyTorchDataLoader
from torch_geometric.loader import DataLoader

from datasets import ArgoverseV2Dataset
from transforms import TargetBuilder


NO_TEST_DATA:bool = eval(os.environ.get("NO_TEST_DATA", "False"))


def bypassed(data_list):
    return data_list


class ArgoverseV2DataModule(pl.LightningDataModule):

    def __init__(self,
                 root: str,
                 train_batch_size: int,
                 val_batch_size: int,
                 test_batch_size: int,
                 shuffle: bool = True,
                 num_workers: int = 0,
                 pin_memory: bool = True,
                 persistent_workers: bool = True,
                 train_raw_dir: Optional[str] = None,
                 val_raw_dir: Optional[str] = None,
                 test_raw_dir: Optional[str] = None,
                 train_processed_dir: Optional[str] = None,
                 val_processed_dir: Optional[str] = None,
                 test_processed_dir: Optional[str] = None,
                 train_transform: Optional[Callable] = TargetBuilder(50, 60),
                 val_transform: Optional[Callable] = TargetBuilder(50, 60),
                 test_transform: Optional[Callable] = None,
                 **kwargs) -> None:
        super(ArgoverseV2DataModule, self).__init__()
        self.root = root
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.train_raw_dir = train_raw_dir
        self.val_raw_dir = val_raw_dir
        self.test_raw_dir = test_raw_dir
        self.train_processed_dir = train_processed_dir
        self.val_processed_dir = val_processed_dir
        self.test_processed_dir = test_processed_dir
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

    def prepare_data(self) -> None:
        ArgoverseV2Dataset(self.root, 'train', self.train_raw_dir, self.train_processed_dir, self.train_transform)
        ArgoverseV2Dataset(self.root, 'val', self.val_raw_dir, self.val_processed_dir, self.val_transform)
        if not NO_TEST_DATA:
            ArgoverseV2Dataset(self.root, 'test', self.test_raw_dir, self.test_processed_dir, self.test_transform)

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = ArgoverseV2Dataset(self.root, 'train', self.train_raw_dir, self.train_processed_dir,
                                                self.train_transform)
        self.val_dataset = ArgoverseV2Dataset(self.root, 'val', self.val_raw_dir, self.val_processed_dir,
                                              self.val_transform)
        if not NO_TEST_DATA:
            self.test_dataset = ArgoverseV2Dataset(self.root, 'test', self.test_raw_dir, self.test_processed_dir,
                                                   self.test_transform)

    def train_dataloader(self):
        return PyTorchDataLoader(self.train_dataset, batch_size=self.train_batch_size, shuffle=self.shuffle,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers, collate_fn=bypassed)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.val_batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          persistent_workers=self.persistent_workers)

    def test_dataloader(self):
        if NO_TEST_DATA:
            raise NotImplementedError()
        else:
            return DataLoader(self.test_dataset, batch_size=self.test_batch_size, shuffle=False,
                              num_workers=self.num_workers, pin_memory=self.pin_memory,
                              persistent_workers=self.persistent_workers)
