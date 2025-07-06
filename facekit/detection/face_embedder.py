import torch
import numpy as np
from torchvision import transforms
from PIL import Image

class FaceEmbedder:
    def __init__(self, model_path="models/mobilefacenet.pth", device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = torch.jit.load(model_path, map_location=self.device).eval()

        self.preprocess = transforms.Compose([
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5])
        ])

    def extract(self, image: np.ndarray) -> np.ndarray:
        img = Image.fromarray(image).convert("RGB")
        tensor = self.preprocess(img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            embedding = self.model(tensor).squeeze().cpu().numpy()
        return embedding / np.linalg.norm(embedding)
