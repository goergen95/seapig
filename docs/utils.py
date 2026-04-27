from typing import Tuple, List
from shapely.geometry import box, Polygon
from torchgeo.trainers import SemanticSegmentationTask
import torch
from torchmetrics import MetricCollection, Accuracy, Precision, Recall, F1Score 

def make_split_polygons(bounds: tuple, size_props: List[float], margin: float = 80.0, res: float = 30.0) -> Tuple[Polygon, ...]:
    """
    Produce rectangular Shapely polygons splitting the x-axis of `bounds`
    according to `size_props`. `bounds` is (x_slice, y_slice, t_slice).
    """
    x_slice, y_slice, _ = bounds
    xmin, xmax = float(x_slice.start), float(x_slice.stop)
    ymin, ymax = float(y_slice.start), float(y_slice.stop)

    # we omit one chip on the edges to avoid issues with sampling near the boundaries
    xmin += margin * res
    xmax -= margin * res
    ymin += margin * res
    ymax -= margin * res

    if len(size_props) < 1:
        raise ValueError("size_props must have at least one proportion")
    if abs(sum(size_props) - 1.0) > 1e-6:
        raise ValueError("size_props must sum to 1.0")

    width = xmax - xmin
    cum = 0.0
    polys: List[Polygon] = []
    for p in size_props:
        x0 = xmin + cum * width
        cum += p
        x1 = xmin + cum * width
        polys.append(box(x0, ymin, x1, ymax))

    return tuple(polys)

class MySegmentationTask(SemanticSegmentationTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    def embed(self, x: torch.Tensor) -> torch.Tensor:
        """Embed an input tensor."""
        # x to model device
        x = x.to(next(self.model.parameters()).device)
        embs = self.model.encoder(x)[-1]
        embs = torch.mean(embs, dim=(-2, -1)) + torch.amax(embs, dim=(-2, -1))
        return embs
    def configure_metrics(self) -> None:
        kwargs = {
            "task": "multiclass",
            "num_classes": self.hparams["num_classes"],
            "ignore_index": int(self.hparams.get("nodata", 255)),
        }
        metrics = MetricCollection(
            {
                "AverageAccuracy": Accuracy(average="micro", **kwargs),
            }
        )
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")
