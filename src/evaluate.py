import os
import torch
from typing import Dict
from tqdm import tqdm
from models.model_capsulenet import CapsuleNet
from data_loader import load_mnist_data
from safetensors.torch import load_file


class ModelEvaluator:
    """
    Evaluator class for CapsuleNet model inference and evaluation.
    """

    def __init__(
            self,
            checkpoint_path: str = "src/results/checkpoint-11600/model.safetensors",
            device: str = None,
            batch_size: int = 64
    ):
        self.checkpoint_path = checkpoint_path
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        self.batch_size = batch_size
        self.model = self._load_model()

    def _load_model(self) -> CapsuleNet:
        """
        Load the model and its weights.

        Returns:
            CapsuleNet: Loaded model
        """
        print(f"Loading model from {self.checkpoint_path}")
        model = CapsuleNet()

        # Verify checkpoint exists
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {self.checkpoint_path}")

        # Load weights
        state_dict = load_file(self.checkpoint_path)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()

        return model

    def calculate_metrics(self, preds: torch.Tensor, targets: torch.Tensor) -> Dict[str, float]:
        """
        Calculate evaluation metrics using PyTorch.

        Args:
            preds: Model predictions
            targets: Ground truth labels

        Returns:
            dict: Dictionary of metrics
        """
        # Calculate accuracy
        correct = (preds == targets).float()
        accuracy = correct.mean().item()

        # Calculate per-class accuracy
        class_correct = torch.zeros(10, device=self.device)
        class_total = torch.zeros(10, device=self.device)
        for i in range(10):
            mask = targets == i
            if mask.any():
                class_correct[i] = correct[mask].sum()
                class_total[i] = mask.sum()

        class_accuracies = (class_correct / class_total).cpu().tolist()

        return {
            "accuracy": accuracy,
            "class_accuracies": class_accuracies
        }

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the model on the test set.

        Returns:
            dict: Dictionary containing evaluation metrics
        """
        # Load test data
        _, test_loader = load_mnist_data(batch_size=self.batch_size)

        total_loss = 0.0
        num_batches = 0
        all_preds = []
        all_targets = []

        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                # Move data to device
                images = batch["pixel_values"].to(self.device)
                labels = batch["labels"].to(self.device)

                # Forward pass
                probs, weighted_probs = self.model(images)
                preds = torch.argmax(probs, dim=1)

                # Calculate loss
                loss = torch.tensor(0., device=self.device)
                for weight in weighted_probs:
                    branch_loss = torch.nn.functional.cross_entropy(probs, labels)
                    loss = loss + weight * branch_loss

                # Accumulate results
                total_loss += loss.item()
                num_batches += 1
                all_preds.append(preds)
                all_targets.append(labels)

        # Concatenate all predictions and targets
        all_preds = torch.cat(all_preds)
        all_targets = torch.cat(all_targets)

        # Calculate metrics
        metrics = self.calculate_metrics(all_preds, all_targets)
        metrics["loss"] = total_loss / num_batches

        return metrics


def main():
    """
    Main function to run evaluation.
    """
    # Initialize evaluator
    evaluator = ModelEvaluator()

    # Run evaluation
    print("Starting evaluation...")
    metrics = evaluator.evaluate()

    # Print results
    print("\nEvaluation Results:")
    print("-" * 50)
    print(f"Average Loss: {metrics['loss']:.4f}")
    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    print("\nPer-class Accuracies:")
    for i, acc in enumerate(metrics['class_accuracies']):
        print(f"Class {i}: {acc:.4f}")
    print("-" * 50)


if __name__ == "__main__":
    main()