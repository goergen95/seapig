import pytest
import torch
from torch.utils.data import DataLoader

from seapig.mockups import MockupCNN, MockupDataset
from seapig.scores.utils import check_model, get_embeddings


class TestScores:
    @pytest.fixture
    def dataset_dict(self) -> MockupDataset:
        return MockupDataset(return_dict=True)

    @pytest.fixture
    def dataset_tensor(self) -> MockupDataset:
        return MockupDataset(return_dict=False)

    @pytest.fixture
    def model_flat(self) -> MockupCNN:
        return MockupCNN(avg=True)

    @pytest.fixture
    def model_full(self) -> MockupCNN:
        return MockupCNN(avg=False)

    @pytest.fixture
    def dataloaders1(self, dataset_dict):
        return DataLoader(dataset=dataset_dict.get_train_split(), batch_size=8)

    @pytest.fixture
    def dataloaders2(self, dataset_tensor):
        return DataLoader(
            dataset=dataset_tensor.get_train_split(), batch_size=8
        )

    def test_get_embeddings(
        self, dataloaders1, dataloaders2, model_flat, model_full, tmp_path
    ):
        embs = get_embeddings(model=model_flat, loader=dataloaders1)
        assert isinstance(embs, torch.Tensor)
        assert embs.shape == (333, 32)
        embs = get_embeddings(model=model_flat, loader=dataloaders2)
        assert isinstance(embs, torch.Tensor)
        assert embs.shape == (333, 32)
        embs = get_embeddings(model=model_flat, loader=dataloaders2)
        outfile = tmp_path / "test.parquet"
        assert not outfile.is_file()
        embs = get_embeddings(
            model=model_flat, loader=dataloaders2, path=outfile
        )
        assert outfile.is_file()
        embs = get_embeddings(
            model=model_flat, loader=dataloaders1, path=outfile
        )

        with pytest.warns(UserWarning):
            embs = get_embeddings(model=model_full, loader=dataloaders1)

    def test_check_model(self):
        with pytest.raises(Exception) as err:
            check_model(torch.nn.Module())
            assert "no attribute 'embed'" in err.value.message

        class mockup(torch.nn.Module):
            def embed(self, y):
                pass

        model = mockup()
        with pytest.raises(Exception) as err:
            check_model(model)
            assert "required to except x as argument." in err.value.message
