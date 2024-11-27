import cv2
import numpy as np
from matplotlib import pyplot as plt


def analyze_mask(mask):
    """
    分析掩码图像以检测连通区域的数量和位置。

    参数：
        mask (numpy.ndarray): 掩码图像。

    返回：
        num_labels (int): 连通区域的数量。
        labels (numpy.ndarray): 每个像素的连通区域标签。
        stats (numpy.ndarray): 每个连通区域的统计信息（包括边界框和像素数）。
        centroids (numpy.ndarray): 每个连通区域的质心。
    """
    # 使用 connectedComponentsWithStats 检测连通区域
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    return num_labels, labels, stats, centroids


def inpaint_image(
    input_path: str,
    mask_path: str,
    output_path: str,
    inpaint_radius: int = 3,
    method: str = "telea",
):
    """
    使用 OpenCV 的 inpaint 方法修复图像。

    参数：
        input_path (str): 输入图像的文件路径。
        mask_path (str): 掩码图像的文件路径。
        output_path (str): 修复后图像保存的文件路径。
        inpaint_radius (int): 修复半径，默认为 3。
        method (str): 修复算法，"telea" 或 "ns"。默认为 "telea"。

    返回：
        restored_image (numpy.ndarray): 修复后的图像。
    """
    # 读取输入图像
    input_image = cv2.imread(input_path)
    if input_image is None:
        raise ValueError(f"无法读取输入图像，请检查路径：{input_path}")

    # 读取掩码图像
    mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask_image is None:
        raise ValueError(f"无法读取掩码图像，请检查路径：{mask_path}")

    # 确保掩码是二值化的 (0 和 255)
    _, mask_image = cv2.threshold(mask_image, 127, 255, cv2.THRESH_BINARY)

    # 分析掩码图像
    num_labels, labels, stats, centroids = analyze_mask(mask_image)
    print(f"检测到 {num_labels - 1} 个连通区域（不包括背景）。")

    # 修复图像
    restored_image = input_image.copy()

    for i in range(1, num_labels):  # 从 1 开始，跳过背景
        # 提取单独连通区域的掩码
        single_mask = np.zeros_like(mask_image)
        single_mask[labels == i] = 255  # 将当前连通区域设为白色

        # 动态调整修复半径（根据当前区域大小）
        area = stats[i, cv2.CC_STAT_AREA]
        adjusted_radius = max(min(inpaint_radius + int(np.log1p(area) / 2), 8), 1)
        print(f"修复区域 {i}，面积：{area}，修复半径：{adjusted_radius}")

        # 选择修复算法
        if method.lower() == "telea":
            flags = cv2.INPAINT_TELEA
        elif method.lower() == "ns":
            flags = cv2.INPAINT_NS
        else:
            raise ValueError(f"未知的修复方法：{method}，可选值为 'telea' 或 'ns'")

        # 执行单次修复
        restored_image = cv2.inpaint(
            restored_image, single_mask, inpaintRadius=adjusted_radius, flags=flags
        )

    # 保存修复后的图像
    cv2.imwrite(output_path, restored_image)

    return restored_image


def display_images(original, mask, restored):
    """
    显示原始图像、掩码图像和修复后的图像。

    参数：
        original (numpy.ndarray): 原始图像。
        mask (numpy.ndarray): 掩码图像。
        restored (numpy.ndarray): 修复后的图像。
    """
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.title("Input Image")
    plt.imshow(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.title("Mask Image")
    plt.imshow(mask, cmap="gray")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.title("Restored Image")
    plt.imshow(cv2.cvtColor(restored, cv2.COLOR_BGR2RGB))
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 文件路径
    input_path = (
        "/Users/luoyongmeng/Documents/lym/python-woork/watermaker/jpg/WechatIMG190.jpg"
    )
    mask_path = "/Users/luoyongmeng/Documents/lym/python-woork/watermaker/jpg/WechatIMG190_mask.jpg"
    output_path = "/Users/luoyongmeng/Documents/lym/python-woork/watermaker/jpg/restored_image_.jpg"

    # 调用封装函数
    restored_image = inpaint_image(
        input_path, mask_path, output_path, inpaint_radius=1, method="telea"
    )

    # 可视化结果
    original_image = cv2.imread(input_path)
    mask_image = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    display_images(original_image, mask_image, restored_image)
