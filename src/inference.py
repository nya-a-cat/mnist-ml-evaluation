import torch
from torchvision import transforms
from PIL import Image
from models.model_capsulenet import CapsuleNet


class Predictor:
    def __init__(self, checkpoint_path="src/results/checkpoint-11600/model.safetensors"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {self.device}")
        self.model = self._load_model(checkpoint_path)
        self.transform = self._get_transforms()

    def _load_model(self, checkpoint_path):
        """加载模型和权重"""
        print(f"Loading model from {checkpoint_path}")
        model = CapsuleNet()

        try:
            # 尝试加载 safetensors 格式
            from safetensors.torch import load_file
            state_dict = load_file(checkpoint_path)
        except:
            # 如果失败，尝试加载 PyTorch 格式
            print("Failed to load safetensors format, trying PyTorch format...")
            if checkpoint_path.endswith('.safetensors'):
                checkpoint_path = checkpoint_path.replace('.safetensors', '.pt')
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            # 如果加载的是完整的检查点而不是状态字典
            if isinstance(state_dict, dict) and 'state_dict' in state_dict:
                state_dict = state_dict['state_dict']

        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        print("Model loaded successfully")
        return model

    def _get_transforms(self):
        """获取图像预处理转换"""
        return transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def predict_single_image(self, image_path):
        """预测单张图片"""
        # 加载和预处理图像
        image = Image.open(image_path).convert('L')
        image = self.transform(image).unsqueeze(0)
        image = image.to(self.device)

        # 推理
        with torch.no_grad():
            probs, _ = self.model(image)
            prediction = torch.argmax(probs, dim=1).item()
            confidence = torch.max(probs).item()

        return {
            'prediction': prediction,
            'confidence': confidence,
            'probabilities': probs[0].cpu().numpy()  # 所有类别的概率
        }

    def predict_batch(self, image_tensor):
        """预测一批图像"""
        image_tensor = image_tensor.to(self.device)

        with torch.no_grad():
            probs, _ = self.model(image_tensor)
            predictions = torch.argmax(probs, dim=1)
            confidences = torch.max(probs, dim=1)[0]

        return {
            'predictions': predictions.cpu().numpy(),
            'confidences': confidences.cpu().numpy(),
            'probabilities': probs.cpu().numpy()  # 所有类别的概率
        }


def main():
    # 修改为实际的检查点路径
    checkpoint_path = "src/results/checkpoint-11600/pytorch_model.bin"  # 或 .pt 文件
    predictor = Predictor(checkpoint_path)

    # 批量预测示例（使用随机生成的测试数据）
    print("\nTesting with random data:")
    batch_size = 4
    test_batch = torch.randn(batch_size, 1, 28, 28)
    results = predictor.predict_batch(test_batch)

    for i in range(batch_size):
        print(f"\nImage {i + 1}:")
        print(f"  Prediction: {results['predictions'][i]}")
        print(f"  Confidence: {results['confidences'][i]:.4f}")
        print(f"  Class probabilities: {results['probabilities'][i].round(3)}")


if __name__ == "__main__":
    main()