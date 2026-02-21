import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from seapig.risk_coverage import risk_coverage
from seapig.scores import EuclideanScore


class MockupCNN(nn.Module):
    """A simple mockup CNN model for demonstration purposes."""

    def __init__(self):
        super(MockupCNN, self).__init__()
        self.conv = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, 1)

    def forward(self, x):
        x = self.conv(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def embed(self, x):
        x = self.conv(x)
        x = self.pool(x).squeeze(dim=(-2, -1))
        return x


class MockupDataset(Dataset):
    """A mockup dataset for demonstration purposes."""

    def __init__(self, size=100, image_shape=(3, 32, 32)):
        self.size = size
        self.image_shape = image_shape
        self.data = torch.randn(size, *image_shape)
        self.labels = torch.randn(size)
        self.masks = torch.randn(size, *image_shape)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return {
            "image": self.data[idx],
            "label": self.labels[idx],
            "mask": self.masks[idx],
        }

    def get_train_split(self):
        return self

    def get_val_split(self):
        return self

    def get_test_split(self):
        return self



def main() -> None:
    """Run the risk-coverage example."""
    print("=" * 60)
    print("Risk-Coverage Analysis Example")
    print("=" * 60)

    # Setup
    model = MockupCNN()
    dataset = MockupDataset()

    train_loader = DataLoader(dataset=dataset.get_train_split(), batch_size=8)
    val_loader = DataLoader(dataset=dataset.get_val_split(), batch_size=8)
    test_loader = DataLoader(dataset=dataset.get_test_split(), batch_size=8)

    # Train confidence score
    print("\n1. Training Euclidean confidence score...")
    score = EuclideanScore(k=1)
    score.fit(model=model, loaders={"train": train_loader, "val": val_loader})
    score.set_threshold(q=0.75)
    print(f"   Threshold set at: {score.get_threshold():.4f}")

    # Calculate confidence scores and residuals on validation set
    print("\n2. Calculating confidence scores and residuals...")
    all_residuals = []

    # Get confidence scores using score
    all_scores = score.score(
        model=model, loader=val_loader, outdir=None, prefix=None
    )

    # Calculate residuals on validation set
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            # Get predictions and calculate residuals
            pred = model.forward(batch["image"])
            # Average pool predictions to get a scalar value per sample
            pred_pooled = pred.view(pred.size(0), -1).mean(1)
            # Average the mask to get a scalar value per sample
            mask_pooled = batch["mask"].view(batch["mask"].size(0), -1).mean(1)
            residuals = torch.abs(pred_pooled - mask_pooled)
            all_residuals.append(residuals)

    all_residuals = torch.cat(all_residuals)

    print(f"   Collected {len(all_scores)} samples")
    print(f"   Mean score: {all_scores.mean():.4f}")
    print(f"   Mean residual: {all_residuals.mean():.4f}")

    # Calculate risk-coverage curves
    print("\n3. Calculating risk-coverage curves...")

    # Generalized risk
    rc_gen = risk_coverage(all_scores, all_residuals, risk="generalized")
    print("\n   Generalized Risk:")
    print(f"   - AUC Empirical: {rc_gen.auc_empirical:.4f}")
    print(f"   - AUC Reference: {rc_gen.auc_reference:.4f}")
    print(f"   - E-AURC:        {rc_gen.auc_excess:.4f}")

    # Selective risk
    rc_sel = risk_coverage(all_scores, all_residuals, risk="selective")
    print("\n   Selective Risk:")
    print(f"   - AUC Empirical: {rc_sel.auc_empirical:.4f}")
    print(f"   - AUC Reference: {rc_sel.auc_reference:.4f}")
    print(f"   - E-AURC:        {rc_sel.auc_excess:.4f}")

    # Plot the curves (if matplotlib is available)
    print("\n4. Creating visualizations...")
    try:
        fig1 = rc_gen.plot()
        fig1.savefig(
            "/tmp/risk_coverage_generalized.png", dpi=150, bbox_inches="tight"
        )
        print(
            "   ✓ Saved generalized risk plot to /tmp/risk_coverage_generalized.png"
        )

        fig2 = rc_sel.plot()
        fig2.savefig(
            "/tmp/risk_coverage_selective.png", dpi=150, bbox_inches="tight"
        )
        print(
            "   ✓ Saved selective risk plot to /tmp/risk_coverage_selective.png"
        )
    except ImportError:
        print("   ⚠ Matplotlib not available, skipping plots")

    # Analyze at specific coverage levels
    print("\n5. Analysis at specific coverage levels:")
    coverage_levels = [0.5, 0.75, 0.9, 1.0]

    for target_cov in coverage_levels:
        # Find closest coverage level
        idx = (torch.abs(rc_gen.coverage - target_cov)).argmin()
        actual_cov = rc_gen.coverage[idx].item()
        emp_risk = rc_gen.risk[idx].item()
        ref_risk = rc_gen.reference[idx].item()
        threshold = rc_gen.threshold[idx].item()

        print(f"\n   Coverage {target_cov:.0%} (actual: {actual_cov:.2%}):")
        print(f"   - Threshold:      {threshold:.4f}")
        print(f"   - Empirical Risk: {emp_risk:.4f}")
        print(f"   - Reference Risk: {ref_risk:.4f}")
        print(f"   - Excess Risk:    {emp_risk - ref_risk:.4f}")

    print("\n" + "=" * 60)
    print("Analysis complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
