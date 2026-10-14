"""Visual embedding models for jewellery retrieval."""

from abc import ABC, abstractmethod
from typing import List, Union, Optional
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as T
import timm
import open_clip

class BaseEncoder(ABC):
    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        pass

    @abstractmethod
    def encode_image(self, image: Image.Image) -> np.ndarray:
        """Encode a single PIL Image to a 1D unit-normalized float32 numpy array."""
        pass

    @abstractmethod
    def encode_batch(self, images: List[Image.Image], batch_size: int = 32) -> np.ndarray:
        """Encode a list of PIL Images to an (N, D) unit-normalized float32 numpy array."""
        pass


class DINOv2Encoder(BaseEncoder):
    def __init__(self, model_name: str = "vit_small_patch14_dinov2", img_size: int = 224, device: str = "cpu"):
        self._model_name = model_name
        self.device = torch.device(device)
        self.img_size = img_size
        
        # Load vision transformer with dynamic image size interpolation
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0,
            dynamic_img_size=True
        )
        self.model.to(self.device)
        self.model.eval()

        self.transform = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @property
    def embedding_dim(self) -> int:
        return 384

    @property
    def model_name(self) -> str:
        return self._model_name

    def _prepare_image(self, image: Image.Image) -> torch.Tensor:
        if image.mode != "RGB":
            image = image.convert("RGB")
        return self.transform(image)

    def encode_image(self, image: Image.Image) -> np.ndarray:
        tensor = self._prepare_image(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(tensor)
            feat = torch.nn.functional.normalize(feat, p=2, dim=-1)
        return feat.cpu().numpy()[0].astype(np.float32)

    def encode_batch(self, images: List[Image.Image], batch_size: int = 32) -> np.ndarray:
        all_feats = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            tensors = torch.stack([self._prepare_image(img) for img in batch]).to(self.device)
            with torch.no_grad():
                feat = self.model(tensors)
                feat = torch.nn.functional.normalize(feat, p=2, dim=-1)
            all_feats.append(feat.cpu().numpy())
        return np.vstack(all_feats).astype(np.float32)

    def get_patch_tokens(self, image: Image.Image) -> np.ndarray:
        """Extract spatial patch tokens (excluding CLS token) for local geometric matching."""
        tensor = self._prepare_image(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.forward_features(tensor) # Shape (1, 257, 384)
            patch_tokens = features[:, 1:, :] # Shape (1, 256, 384)
            patch_tokens = torch.nn.functional.normalize(patch_tokens, p=2, dim=-1)
        return patch_tokens.cpu().numpy()[0].astype(np.float32)

    def get_patch_tokens_batch(self, images: List[Image.Image], batch_size: int = 32) -> np.ndarray:
        all_patches = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            tensors = torch.stack([self._prepare_image(img) for img in batch]).to(self.device)
            with torch.no_grad():
                features = self.model.forward_features(tensors)
                patch_tokens = features[:, 1:, :]
                patch_tokens = torch.nn.functional.normalize(patch_tokens, p=2, dim=-1)
            all_patches.append(patch_tokens.cpu().numpy())
        return np.vstack(all_patches).astype(np.float32)


class CLIPEncoder(BaseEncoder):
    def __init__(self, model_name: str = "ViT-B-32", pretrained: str = "openai", device: str = "cpu"):
        self._model_name = f"clip_{model_name}"
        self.device = torch.device(device)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained
        )
        self.model.to(self.device)
        self.model.eval()

    @property
    def embedding_dim(self) -> int:
        return 512

    @property
    def model_name(self) -> str:
        return self._model_name

    def encode_image(self, image: Image.Image) -> np.ndarray:
        if image.mode != "RGB":
            image = image.convert("RGB")
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model.encode_image(tensor)
            feat = torch.nn.functional.normalize(feat, p=2, dim=-1)
        return feat.cpu().numpy()[0].astype(np.float32)

    def encode_batch(self, images: List[Image.Image], batch_size: int = 32) -> np.ndarray:
        all_feats = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            tensors = torch.stack([
                self.preprocess(img.convert("RGB") if img.mode != "RGB" else img)
                for img in batch
            ]).to(self.device)
            with torch.no_grad():
                feat = self.model.encode_image(tensors)
                feat = torch.nn.functional.normalize(feat, p=2, dim=-1)
            all_feats.append(feat.cpu().numpy())
        return np.vstack(all_feats).astype(np.float32)

class DINOv3Encoder(BaseEncoder):
    def __init__(self, model_name: str = "vit_small_patch16_dinov3.lvd1689m", img_size: int = 256, device: str = "cpu"):
        self._model_name = "dinov3"
        self.backbone_tag = model_name
        self.device = torch.device(device)
        self.img_size = img_size
        
        self.model = timm.create_model(
            model_name,
            pretrained=True,
            num_classes=0
        )
        self.model.to(self.device)
        self.model.eval()

        self.transform = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @property
    def embedding_dim(self) -> int:
        return 384

    @property
    def model_name(self) -> str:
        return self._model_name

    def _prepare_image(self, image: Image.Image) -> torch.Tensor:
        if image.mode != "RGB":
            image = image.convert("RGB")
        return self.transform(image)

    def encode_image(self, image: Image.Image) -> np.ndarray:
        tensor = self._prepare_image(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            feat = self.model(tensor)
            feat = torch.nn.functional.normalize(feat, p=2, dim=-1)
        return feat.cpu().numpy()[0].astype(np.float32)

    def encode_batch(self, images: List[Image.Image], batch_size: int = 32) -> np.ndarray:
        all_feats = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            tensors = torch.stack([self._prepare_image(img) for img in batch]).to(self.device)
            with torch.no_grad():
                feat = self.model(tensors)
                feat = torch.nn.functional.normalize(feat, p=2, dim=-1)
            all_feats.append(feat.cpu().numpy())
        return np.vstack(all_feats).astype(np.float32)

    def get_patch_tokens(self, image: Image.Image) -> np.ndarray:
        """Extract spatial patch tokens (excluding prefix registers) for local geometric matching."""
        tensor = self._prepare_image(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.model.forward_features(tensor)  # Shape (1, 261, 384)
            # Patch tokens start after the 5 prefix/register tokens
            patch_tokens = features[:, 5:, :]  # Shape (1, 256, 384)
            patch_tokens = torch.nn.functional.normalize(patch_tokens, p=2, dim=-1)
        return patch_tokens.cpu().numpy()[0].astype(np.float32)

    def get_patch_tokens_batch(self, images: List[Image.Image], batch_size: int = 32) -> np.ndarray:
        all_patches = []
        for i in range(0, len(images), batch_size):
            batch = images[i : i + batch_size]
            tensors = torch.stack([self._prepare_image(img) for img in batch]).to(self.device)
            with torch.no_grad():
                features = self.model.forward_features(tensors)
                patch_tokens = features[:, 5:, :]
                patch_tokens = torch.nn.functional.normalize(patch_tokens, p=2, dim=-1)
            all_patches.append(patch_tokens.cpu().numpy())
        return np.vstack(all_patches).astype(np.float32)


class MobileNetV4Encoder(BaseEncoder):
    """Ultra-fast lightweight vision backbone for high-throughput and low-latency inference."""
    def __init__(
        self,
        model_name: str = "mobilenetv4_conv_small",
        img_size: int = 224,
        device: str = "cpu",
        onnx_path: Optional[str] = "data/models/mobilenetv4_conv_small_fp32.onnx"
    ):
        self._model_name = "mobilenetv4"
        self.img_size = img_size
        self.onnx_path = Path(onnx_path) if onnx_path else None
        self.session = None
        self.transform = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Prefer ONNX Runtime engine if export exists
        if self.onnx_path and self.onnx_path.exists():
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(str(self.onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
        else:
            self.device = torch.device(device)
            self.model = timm.create_model(model_name, pretrained=True, num_classes=0)
            self.model.to(self.device).eval()

    @property
    def embedding_dim(self) -> int:
        return 1280

    @property
    def model_name(self) -> str:
        return self._model_name

    def _prepare_image(self, image: Image.Image) -> torch.Tensor:
        if image.mode != "RGB":
            image = image.convert("RGB")
        return self.transform(image)

    def encode_image(self, image: Image.Image) -> np.ndarray:
        tensor = self._prepare_image(image)
        if self.session is not None:
            inp = tensor.unsqueeze(0).numpy()
            feat = self.session.run(None, {"input": inp})[0][0]
            norm = np.linalg.norm(feat)
            return (feat / (norm + 1e-8)).astype(np.float32)
        else:
            with torch.no_grad():
                feat = self.model(tensor.unsqueeze(0).to(self.device))
                feat = torch.nn.functional.normalize(feat, p=2, dim=-1)
            return feat.cpu().numpy()[0].astype(np.float32)

    def encode_batch(self, images: List[Image.Image], batch_size: int = 32) -> np.ndarray:
        return np.vstack([self.encode_image(img) for img in images]).astype(np.float32)


class DINOv3ONNXEncoder(BaseEncoder):
    """ONNX Runtime accelerated DINOv3 encoder (INT8 quantized or FP32)."""
    def __init__(self, onnx_path: str = "data/models/dinov3_vits16_int8.onnx", img_size: int = 256):
        self._model_name = "dinov3_int8" if "int8" in onnx_path else "dinov3_onnx"
        self.img_size = img_size
        self.onnx_path = Path(onnx_path)
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 2
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        self.session = ort.InferenceSession(str(self.onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
        self.transform = T.Compose([
            T.Resize((img_size, img_size), interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    @property
    def embedding_dim(self) -> int:
        return 384

    @property
    def model_name(self) -> str:
        return self._model_name

    def encode_image(self, image: Image.Image) -> np.ndarray:
        if image.mode != "RGB":
            image = image.convert("RGB")
        tensor = self.transform(image).unsqueeze(0).numpy()
        feat = self.session.run(None, {"input": tensor})[0][0]
        norm = np.linalg.norm(feat)
        return (feat / (norm + 1e-8)).astype(np.float32)

    def encode_batch(self, images: List[Image.Image], batch_size: int = 32) -> np.ndarray:
        return np.vstack([self.encode_image(img) for img in images]).astype(np.float32)


def get_encoder(name: str = "dinov2", device: str = "cpu") -> BaseEncoder:
    """Factory for embedding models."""
    name_clean = name.lower()
    if "mobilenet" in name_clean:
        return MobileNetV4Encoder(device=device)
    elif "dinov3_int8" in name_clean or "dinov3_onnx" in name_clean:
        return DINOv3ONNXEncoder("data/models/dinov3_vits16_int8.onnx")
    elif "dinov3_static" in name_clean:
        return DINOv3ONNXEncoder("data/models/dinov3_vits16_static_int8.onnx")
    elif "dinov3" in name_clean:
        # shortcut: default to local INT8 ONNX if available to prevent HF downloads, use dinov3_pytorch to force timm
        onnx_file = Path("data/models/dinov3_vits16_int8.onnx")
        if onnx_file.exists() and "pytorch" not in name_clean:
            return DINOv3ONNXEncoder(str(onnx_file))
        return DINOv3Encoder(device=device)
    elif "dino" in name_clean:
        return DINOv2Encoder(device=device)
    elif "clip" in name_clean:
        return CLIPEncoder(device=device)
    else:
        raise ValueError(f"Unknown model encoder: {name}. Supported: 'dinov2', 'dinov3', 'dinov3_int8', 'mobilenetv4', 'clip'.")
