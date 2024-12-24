import sys
import os

# 添加项目根目录到 sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.model_capsulenet import CapsuleNet
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import wandb
from datetime import datetime

from transformers import Trainer, TrainingArguments, PrinterCallback

os.environ["WANDB_PROJECT"] = "mnist-capsulenet"  # name your W&B project
os.environ["WANDB_LOG_MODEL"] = "checkpoint"  # log all model checkpoints


# 自定义数据集包装类
class MNISTDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return {
            "pixel_values": image,
            "labels": label
        }


# 自定义模型包装类
class CapsuleNetForTrainer(CapsuleNet):
    def forward(self, pixel_values, labels=None):
        probs, weighted_probs = super().forward(pixel_values)

        loss = None
        if labels is not None:
            loss = torch.tensor(0., device=probs.device)
            for i, weight in enumerate(weighted_probs):
                branch_loss = torch.nn.functional.cross_entropy(probs, labels)
                loss = loss + weight * branch_loss

        return {"loss": loss, "logits": probs} if loss is not None else probs


def main():

    wandb.login()

    # 初始化 wandb 运行
    run = wandb.init(
        project="mnist-capsulenet",
        name=f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        config={
            "learning_rate": 1e-3,
            "batch_size": 32,
            "epochs": 50,
            "weight_decay": 1e-5,
        }
    )



    # 数据预处理
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    # 加载数据集
    train_dataset = datasets.MNIST('data', train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST('data', train=False, transform=transform)

    # 包装数据集
    train_dataset = MNISTDataset(train_dataset)
    val_dataset = MNISTDataset(val_dataset)

    # 创建模型
    model = CapsuleNetForTrainer()

    # 定义训练参数
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=50,
        per_device_train_batch_size=60,
        per_device_eval_batch_size=32,
        learning_rate=1e-3,
        weight_decay=1e-5,
        logging_dir="./logs",
        logging_strategy="steps",  # 按步数记录
        eval_strategy="steps",  # 按步数评估
        eval_steps=500,
        save_strategy="steps",
        save_steps=1000,  # 每1000步保存一次模型
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        report_to="wandb",  # 启用W&B日志记录
        logging_steps=1,  # 每1步记录一次
    )

    # 定义评估函数
    def compute_metrics(eval_pred):
        predictions = eval_pred.predictions.argmax(-1)
        labels = eval_pred.label_ids
        accuracy = (predictions == labels).mean()
        return {
            "accuracy": accuracy
        }

    # 初始化回调函数
    callbacks = [PrinterCallback()]

    # 初始化 Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[PrinterCallback()]
    )

    # 开始训练
    print("Starting training...")
    trainer.train()
    print("Training completed!")

    # 保存最佳模型
    trainer.save_model("./best_model")
    print("Model saved to ./best_model")


if __name__ == "__main__":
    main()
