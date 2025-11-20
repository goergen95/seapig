"""TorchGeo based example data."""

import os
import tempfile

from torch.utils.data import DataLoader
from torchgeo.datasets import CDL, Landsat7, Landsat8, stack_samples
from torchgeo.datasets.utils import download_and_extract_archive
from torchgeo.samplers import GridGeoSampler, RandomGeoSampler

landsat_root = os.path.join(tempfile.gettempdir(), "landsat")

url = "https://hf.co/datasets/torchgeo/tutorials/resolve/ff30b729e3cbf906148d69a4441cc68023898924/"
landsat7_url = url + "LE07_L2SP_022032_20230725_20230820_02_T1.tar.gz"
landsat8_url = url + "LC08_L2SP_023032_20230831_20230911_02_T1.tar.gz"

download_and_extract_archive(landsat7_url, landsat_root)
download_and_extract_archive(landsat8_url, landsat_root)

landsat7_bands = ["SR_B1", "SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B7"]
landsat8_bands = ["SR_B2", "SR_B3", "SR_B4", "SR_B5", "SR_B6", "SR_B7"]

landsat7 = Landsat7(paths=landsat_root, bands=landsat7_bands)
landsat8 = Landsat8(paths=landsat_root, bands=landsat8_bands)

cdl_root = os.path.join(tempfile.gettempdir(), "cdl")
cdl_url = url + "2023_30m_cdls.zip"
download_and_extract_archive(cdl_url, cdl_root)
cdl = CDL(paths=cdl_root)

landsat = landsat7 | landsat8
dataset = landsat & cdl


size = 12
train_sampler = RandomGeoSampler(dataset, size=size, length=1000)
test_sampler = GridGeoSampler(dataset, size=size, stride=size)

train_dataloader = DataLoader(
    dataset, batch_size=4, sampler=train_sampler, collate_fn=stack_samples
)
test_dataloader = DataLoader(
    dataset, batch_size=4, sampler=test_sampler, collate_fn=stack_samples
)


batch = next(iter(test_dataloader))

query = next(iter(train_sampler))
x = landsat7[query]
landsat7.plot(x)
