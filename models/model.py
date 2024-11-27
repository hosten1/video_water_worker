import time
import os
import torch
import numpy as np
from PIL import Image
from utils.util import prepare_img_and_mask, download_model

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

        self.model = torch.jit.load(model_path, map_location=device)
        self.model.eval()
        self.model.to(device)
        self.device = device

    def __call__(self, image: Image.Image | np.ndarray, mask: Image.Image | np.ndarray):
        # 准备输入图像和掩码
        image, mask = prepare_img_and_mask(image, mask, self.device)
        print("Preparing input image and mask")
        # 开始计时
        start_time = time.time()

        # 使用 torch.jit.optimized_execution(False) 替代 torch.no_grad()
        with torch.jit.optimized_execution(False):
            inpainted = self.model(image, mask)

            cur_res = inpainted[0].permute(1, 2, 0).detach().cpu().numpy()
            cur_res = np.clip(cur_res * 255, 0, 255).astype(np.uint8)

            cur_res = Image.fromarray(cur_res)

        # 结束计时
        end_time = time.time()

        # 打印耗时结果
        elapsed_time = end_time - start_time
        print(f"Inference completed in {elapsed_time:.4f} seconds")

        return cur_res
