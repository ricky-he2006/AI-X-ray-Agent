# ai_models.py
# ======================================
# Universal X-ray Analyzer – FIXED VERSION
# ======================================

import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import numpy as np
import os


# =====================================================
# Utility Functions
# =====================================================

def _clean_state_dict(state_dict):
    """Strip 'module.' prefix if model was saved with DataParallel."""
    if any(k.startswith("module.") for k in state_dict.keys()):
        new_state = {}
        for k, v in state_dict.items():
            new_state[k.replace("module.", "")] = v
        return new_state
    return state_dict


def _ensure_image(image):
    """Ensure input is a PIL image."""
    if isinstance(image, np.ndarray):
        if image.dtype != np.uint8:
            raise ValueError(f"Expected uint8 numpy array, got {image.dtype}")
        image = Image.fromarray(image)
    elif not isinstance(image, Image.Image):
        raise TypeError("Input must be a PIL Image or numpy array.")
    return image


def _debug_device(model, tensor):
    """Print device info for debugging."""
    print("Model device:", next(model.parameters()).device)
    print("Tensor device:", tensor.device, "| shape:", tensor.shape)


# =====================================================
# 1️⃣ Body Region Classifier
# =====================================================

class BodyRegionClassifier:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = ["Chest", "Extremity", "Abdomen", "Other"]
        self.model = None
        self.load_model()

    def load_model(self):
        """Load pretrained ResNet-50 model for region classification."""
        self.model = models.resnet50(pretrained=True)
        self.model.fc = nn.Linear(self.model.fc.in_features, len(self.classes))

        weight_path = "models/region_classifier_resnet50.pth"
        if os.path.exists(weight_path):
            print(f"✓ Loading region classifier weights from {weight_path}")
            state = torch.load(weight_path, map_location=self.device)
            state = _clean_state_dict(state)
            self.model.load_state_dict(state, strict=False)
        else:
            raise FileNotFoundError(
                f"Region classifier weights not found at {weight_path}. Please add them."
            )

        self.model.to(self.device)
        self.model.eval()

    def predict(self, image):
        image = _ensure_image(image)
        transform = transforms.Compose([
            transforms.Resize((320, 320)),
            transforms.CenterCrop(320),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform(image).unsqueeze(0).to(self.device)
        _debug_device(self.model, img_tensor)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            pred_idx = probabilities.argmax(dim=1).item()

        probs = probabilities.cpu().numpy().squeeze()
        print("Top region probs:", [(self.classes[i], float(probs[i])) for i in np.argsort(probs)[::-1]])

        return self.classes[pred_idx], float(probabilities[0, pred_idx])


# =====================================================
# 2️⃣ Chest X-ray Analyzer
# =====================================================

class ChestXrayAnalyzer:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = [
            "Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Effusion",
            "Emphysema", "Fibrosis", "Fracture", "Hernia", "Infiltration",
            "Lung Lesion", "Mass", "Nodule", "Pleural Thickening", "Pneumonia",
            "Pneumothorax"
        ]
        self.model = None
        self.load_model()

    def load_model(self):
        self.model = models.densenet121(pretrained=True)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, len(self.classes))

        weight_path = "models/chexpert_densenet121.pth"
        if os.path.exists(weight_path):
            print(f"✓ Loading CheXpert DenseNet weights from {weight_path}")
            state = torch.load(weight_path, map_location=self.device)
            state = _clean_state_dict(state)
            self.model.load_state_dict(state, strict=False)
        else:
            raise FileNotFoundError(
                f"Chest X-ray weights not found at {weight_path}. Please add them."
            )

        self.model.to(self.device)
        self.model.eval()

    def predict(self, image):
        image = _ensure_image(image)
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((320, 320)),
            transforms.CenterCrop(320),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform(image).unsqueeze(0).to(self.device)
        _debug_device(self.model, img_tensor)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.sigmoid(outputs).cpu().numpy().squeeze()

        print("Chest probs:")
        for i, name in enumerate(self.classes):
            print(f"  {name:<20} {probs[i]:.3f}")

        findings = [
            {"condition": name, "confidence": float(prob)}
            for name, prob in zip(self.classes, probs)
            if prob >= 0.5
        ]

        return findings if findings else [{"condition": "No major abnormalities", "confidence": 0.0}]


# =====================================================
# 3️⃣ Bone X-ray Analyzer
# =====================================================

class BoneXrayAnalyzer:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = ["Normal", "Abnormal"]
        self.model = None
        self.load_model()

    def load_model(self):
        self.model = models.densenet169(pretrained=True)
        self.model.classifier = nn.Linear(self.model.classifier.in_features, len(self.classes))

        weight_path = "models/mura_densenet169.pth"
        if os.path.exists(weight_path):
            print(f"✓ Loading MURA DenseNet weights from {weight_path}")
            state = torch.load(weight_path, map_location=self.device)
            state = _clean_state_dict(state)
            self.model.load_state_dict(state, strict=False)
        else:
            raise FileNotFoundError(
                f"MURA bone weights not found at {weight_path}. Please add them."
            )

        self.model.to(self.device)
        self.model.eval()

    def predict(self, image):
        image = _ensure_image(image)
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((320, 320)),
            transforms.CenterCrop(320),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform(image).unsqueeze(0).to(self.device)
        _debug_device(self.model, img_tensor)

        with torch.no_grad():
            outputs = self.model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)[0].cpu().numpy()

        print("Bone probs:", [(self.classes[i], float(probs[i])) for i in np.argsort(probs)[::-1]])

        pred_idx = int(np.argmax(probs))
        return self.classes[pred_idx], float(probs[pred_idx])


# =====================================================
# 4️⃣ Universal X-ray Analyzer
# =====================================================

class UniversalXrayAnalyzer:
    def __init__(self, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🧠 Using device: {self.device}")
        self.region_classifier = BodyRegionClassifier(self.device)
        self.chest_analyzer = ChestXrayAnalyzer(self.device)
        self.bone_analyzer = BoneXrayAnalyzer(self.device)

    def analyze(self, image):
        image = _ensure_image(image)
        print("Running Body Region Detection...")
        region, conf = self.region_classifier.predict(image)
        print(f"Region detected: {region} (conf={conf:.3f})")

        if region == "Chest":
            print("Running Chest X-ray Analysis...")
            findings = self.chest_analyzer.predict(image)
        elif region == "Extremity":
            print("Running Bone X-ray Analysis...")
            pred, score = self.bone_analyzer.predict(image)
            findings = [{"condition": pred, "confidence": score}]
        else:
            findings = [{"condition": "Unsupported region", "confidence": 0.0}]

        return {"body_region": region, "region_confidence": conf, "findings": findings}
