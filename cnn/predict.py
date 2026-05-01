import torch
import cv2
from torchvision import transforms

from cnn.model import GestureCNN
from cnn.classes import CLASS_NAMES


class GesturePredictor:
    def __init__(self, model_path="models/hand_gesture_cnn.pth"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.class_names = CLASS_NAMES

        self.model = GestureCNN(num_classes=len(self.class_names)).to(self.device)
        self.model.load_state_dict(
            torch.load(model_path, map_location=self.device, weights_only=True)
        )
        self.model.eval()

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.5, 0.5, 0.5),
                std=(0.5, 0.5, 0.5)
            )
        ])

    def predict(self, hand_crop):
        hand_crop = cv2.cvtColor(hand_crop, cv2.COLOR_BGR2RGB)

        image = self.transform(hand_crop)
        image = image.unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.model(image)
            probs = torch.softmax(logits, dim=1)

            confidence, pred_idx = torch.max(probs, dim=1)

        predicted_class = self.class_names[pred_idx.item()]
        confidence = confidence.item()

        return predicted_class, confidence