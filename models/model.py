import os
import time
import torch
import numpy as np
import cv2
from PIL import Image
from utils.util import prepare_img_and_mask, download_model
import platform

LAMA_MODEL_URL = os.environ.get(
    "LAMA_MODEL_URL",
    "https://github.com/enesmsahin/simple-lama-inpainting/releases/download/v0.1.0/big-lama.pt",  # noqa
)


class SimpleLama:
    def __init__(
        self,
        device: torch.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ) -> None:
        if os.environ.get("LAMA_MODEL"):
            model_path = os.environ.get("LAMA_MODEL")
            if not os.path.exists(model_path):
                raise FileNotFoundError(
                    f"lama torchscript model not found: {model_path}"
                )
        else:
            model_path = download_model(LAMA_MODEL_URL)

        # 使用 lazy 加载以节省时间
        self.model = torch.jit.load(model_path, map_location=device)
        self.model = torch.jit.optimize_for_inference(self.model)  # 优化模型
        self.model.eval()
        self.model.to(device)
        self.device = device

    def __call__(self, image: Image.Image | np.ndarray, mask: Image.Image | np.ndarray):
        # 准备输入图像和掩码
        image, mask = prepare_img_and_mask(
            image, mask, self.device, pad_out_to_modulo=16
        )

        # 打印输入张量形状
        print(f"Image shape: {image.shape}, Mask shape: {mask.shape}")

        # 开始计时
        start_time = time.time()

        # 优化推理过程
        with torch.inference_mode():  # 使用混合精度推理
            inpainted = self.model(image, mask)

        # 后处理：减少 CPU 和 GPU 间的数据传输次数
        cur_res = (inpainted[0].permute(1, 2, 0).detach().cpu().numpy() * 255).astype(
            np.uint8
        )

        # 使用 PIL 优化生成
        cur_res = Image.fromarray(cur_res)

        # 结束计时
        elapsed_time = time.time() - start_time
        print(f"Inference completed in {elapsed_time:.4f} seconds")

        return cur_res
