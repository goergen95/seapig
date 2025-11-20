import pytest
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from seapig import (
    CosineScore,
    EuclideanScore,
    MahalanobisScore,
    PNormScore,
    RandomScore,
)
from seapig.mockups import MockupCNN, MockupDataset


class TestScores:
    @pytest.fixture
    def dataset(self) -> MockupDataset:
        return MockupDataset()

    @pytest.fixture
    def model(self) -> MockupCNN:
        return MockupCNN()

    @pytest.fixture
    def dataloaders(self, dataset: MockupDataset):
        return {
            "train": DataLoader(
                dataset=dataset.get_train_split(), batch_size=8
            ),
            "val": DataLoader(dataset=dataset.get_val_split(), batch_size=8),
            "test": DataLoader(dataset=dataset.get_test_split(), batch_size=8),
        }

    def test_RandomScore(self, score=RandomScore()):
        with pytest.raises(TypeError):
            RandomScore(arg=1)
        inputs = torch.rand(4)
        dists = score.score(batch=inputs)
        dists = score.score(batch={"inputs": inputs})
        assert isinstance(dists, torch.Tensor)
        assert dists.size()[0] == 4

    def test_EuclideanScore(self, dataloaders, model, tmp_path):
        score = EuclideanScore()

        with pytest.raises(AssertionError):
            score.train(model=model, loader=[1, 2, 3])

        with pytest.raises(AssertionError):
            score.train(model=[1, 2, 3], loader=dataloaders["train"])

        assert score.embeddings is None
        score.train(model, loader=dataloaders["train"])
        assert isinstance(score.embeddings, Tensor)

        new_path = tmp_path / "new_path"
        score.train(
            model, loader=dataloaders["train"], outdir=new_path, prefix="test"
        )
        assert new_path.is_dir()
        assert isinstance(score.embeddings, Tensor)
        assert (new_path / "test.parquet").is_file()
        assert score.embeddings.size() == (333, 32)

        batch = next(iter(dataloaders["train"]))
        dists = score.score(batch=batch, model=model)
        assert dists.shape[0] == batch["inputs"].shape[0]
        score = EuclideanScore(k=2)
        score.train(
            model, loader=dataloaders["train"], outdir=new_path, prefix="test"
        )
        dists2 = score.score(batch=batch, model=model)
        assert dists is not dists2

    def test_CosineScore(self, dataloaders, model, tmp_path):
        score = CosineScore()

        with pytest.raises(AssertionError):
            score.train(model=model, loader=[1, 2, 3])

        with pytest.raises(AssertionError):
            score.train(model=[1, 2, 3], loader=dataloaders["train"])

        assert score.embeddings is None
        score.train(model, loader=dataloaders["train"])
        assert isinstance(score.embeddings, Tensor)

        new_path = tmp_path / "new_path"
        score.train(
            model, loader=dataloaders["train"], outdir=new_path, prefix="test"
        )
        assert new_path.is_dir()
        assert isinstance(score.embeddings, Tensor)
        assert (new_path / "test.parquet").is_file()
        assert score.embeddings.size() == (333, 32)

        batch = next(iter(dataloaders["train"]))
        dists = score.score(batch=batch, model=model)
        assert dists.shape[0] == batch["inputs"].shape[0]
        assert sum(dists > 0) == batch["inputs"].shape[0]
        score = CosineScore(k=2, abs=False)
        score.train(
            model, loader=dataloaders["train"], outdir=new_path, prefix="test"
        )
        dists2 = score.score(batch=batch, model=model)
        assert dists is not dists2

    def test_PNormScore(self, dataloaders, model, tmp_path):
        score = PNormScore()

        with pytest.raises(AssertionError):
            score.train(model=model, loader=[1, 2, 3])

        with pytest.raises(AssertionError):
            score.train(model=[1, 2, 3], loader=dataloaders["train"])

        assert score.embeddings is None
        score.train(model, loader=dataloaders["train"])
        assert isinstance(score.embeddings, Tensor)

        new_path = tmp_path / "new_path"
        score.train(
            model, loader=dataloaders["train"], outdir=new_path, prefix="test"
        )
        assert new_path.is_dir()
        assert isinstance(score.embeddings, Tensor)
        assert (new_path / "test.parquet").is_file()
        assert score.embeddings.size() == (333, 32)

        batch = next(iter(dataloaders["train"]))
        dists = score.score(batch=batch, model=model)
        score = PNormScore(p=3)
        score.train(
            model, loader=dataloaders["train"], outdir=new_path, prefix="test"
        )
        dists2 = score.score(batch=batch, model=model)
        assert dists is not dists2

    def test_MahalanobisScore(self, dataloaders, model, tmp_path):
        score = MahalanobisScore()

        with pytest.raises(AssertionError):
            score.train(model=model, loader=[1, 2, 3])

        with pytest.raises(AssertionError):
            score.train(model=[1, 2, 3], loader=dataloaders["train"])

        assert score.embeddings is None
        score.train(model, loader=dataloaders["train"])
        assert isinstance(score.embeddings, Tensor)

        new_path = tmp_path / "new_path"
        score.train(
            model, loader=dataloaders["train"], outdir=new_path, prefix="test"
        )
        assert new_path.is_dir()
        assert isinstance(score.embeddings, Tensor)
        assert (new_path / "test.parquet").is_file()
        assert score.embeddings.size() == (333, 32)

        batch = next(iter(dataloaders["train"]))
        dists = score.score(batch=batch, model=model)
        assert isinstance(dists, torch.Tensor)
