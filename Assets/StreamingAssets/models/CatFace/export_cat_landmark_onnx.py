"""
将 best_cat_model.pth 导出为 ONNX 格式
=====================================================
模型来源: https://github.com/chelsea23311/Cat-Face-Landmark-Detection
网络结构: ResNet-50 backbone + Dropout(0.5) + Linear(2048, 18) + Sigmoid
输入:  (1, 3, 224, 224)  float32  RGB  已做 ImageNet 归一化
输出:  (1, 18)            float32  9个关键点坐标，已经 sigmoid 到 [0,1]
       reshape → (9, 2) 后乘以 224 得到像素坐标

用法:
    pip install torch torchvision onnx onnxsim onnxruntime
    python export_cat_landmark_onnx.py --weights best_cat_model.pth --output cat_landmark.onnx --simplify
"""

import argparse
import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torchvision.models as models


# ─────────────────────────────────────────────────────────────────────────────
# 模型定义 —— 与 model.py 完全一致，无需依赖原始仓库
# ─────────────────────────────────────────────────────────────────────────────
class ResNet50(nn.Module):
    """
    ResNet-50 猫脸关键点回归网络
    输出 9 个关键点坐标，经 Sigmoid 归一化到 [0, 1]
    顺序: 左眼, 右眼, 嘴, 左耳1, 左耳2, 左耳3, 右耳1, 右耳2, 右耳3
    """
    def __init__(self, num_landmarks: int = 9):
        super().__init__()
        self.backbone = models.resnet50(weights=None)   # 导出时不需要 pretrained 权重
        num_ftrs = self.backbone.fc.in_features         # 2048
        self.backbone.fc = nn.Identity()
        self.dropout = nn.Dropout(p=0.5)
        self.fc = nn.Linear(num_ftrs, num_landmarks * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone(x)   # (B, 2048)
        x = self.dropout(x)
        x = self.fc(x)         # (B, 18)
        return torch.sigmoid(x)


# ─────────────────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────────────────
def load_weights(model: nn.Module, weights_path: str, device: torch.device) -> nn.Module:
    """加载 .pth 权重，自动处理 DataParallel 的 'module.' 前缀"""
    print(f"  加载权重: {weights_path}")
    state_dict = torch.load(weights_path, map_location=device)

    # 兼容 DataParallel 保存的 state_dict
    if list(state_dict.keys())[0].startswith('module.'):
        new_sd = OrderedDict()
        for k, v in state_dict.items():
            new_sd[k[7:]] = v          # 去掉 'module.' 前缀
        state_dict = new_sd

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


# ─────────────────────────────────────────────────────────────────────────────
# 导出主函数
# ─────────────────────────────────────────────────────────────────────────────
def export(weights_path: str, output_path: str, simplify: bool = True, opset: int = 12):
    device = torch.device('cpu')

    # 1. 建模 & 加载权重
    print("[1/4] 构建模型并加载权重 ...")
    model = ResNet50(num_landmarks=9).to(device)
    model = load_weights(model, weights_path, device)

    # 2. 构造 dummy 输入
    #    预处理与 predict.py 完全一致:
    #      - resize to 224×224
    #      - ToTensor  (值域 0~1)
    #      - Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    dummy = torch.zeros(1, 3, 224, 224, dtype=torch.float32)

    print("[2/4] 验证前向推理 ...")
    with torch.no_grad():
        out = model(dummy)
    print(f"  输出 shape: {out.shape}  (期望: [1, 18])")
    assert out.shape == (1, 18), f"输出形状异常: {out.shape}"

    # 3. 导出 ONNX
    print(f"[3/4] 导出 ONNX (opset={opset}) → {output_path}")
    torch.onnx.export(
        model,
        dummy,
        output_path,
        export_params=True,
        opset_version=opset,
        do_constant_folding=True,
        input_names=['input'],          # (1, 3, 224, 224)
        output_names=['landmarks'],     # (1, 18)  [x0,y0, x1,y1, ..., x8,y8]
        dynamic_axes=None,              # 静态 shape，方便 Unity 使用
        verbose=False,
    )
    print("  ONNX 文件已写入。")

    # 4. 可选简化
    if simplify:
        try:
            import onnx, onnxsim
            print("[4/4] 使用 onnx-simplifier 简化计算图 ...")
            model_onnx = onnx.load(output_path)
            model_sim, ok = onnxsim.simplify(model_onnx)
            if ok:
                onnx.save(model_sim, output_path)
                print("  简化成功。")
            else:
                print("  简化失败，保留原始 ONNX。")
        except ImportError:
            print("[4/4] 未安装 onnx/onnxsim，跳过简化。")
            print("      安装: pip install onnx onnx-simplifier")
    else:
        print("[4/4] 已跳过简化 (去掉 --simplify 参数)。")

    # 5. ONNXRuntime 正确性验证
    try:
        import onnxruntime as ort
        import numpy as np
        sess = ort.InferenceSession(output_path, providers=['CPUExecutionProvider'])
        inp  = dummy.numpy()
        outs = sess.run(None, {'input': inp})
        print(f"\n[OK] ONNXRuntime 验证通过。输出 shape: {outs[0].shape}")
        # 输出坐标范围应在 [0, 1]
        print(f"     坐标值范围: [{outs[0].min():.4f}, {outs[0].max():.4f}]  (期望 0~1)")
    except ImportError:
        print("\n[INFO] 未安装 onnxruntime，跳过验证。")

    file_size = os.path.getsize(output_path) / 1024 / 1024
    print(f"\n完成！模型已保存: {output_path}  ({file_size:.1f} MB)")
    print("\n预处理合同 (Unity 端必须保持一致)")
    print("  1. 将输入图像 resize 到 224×224 (RGB)")
    print("  2. 各通道除以 255 归一化到 [0,1]")
    print("  3. 减均值: R-=0.485, G-=0.456, B-=0.406")
    print("  4. 除标准差: R/=0.229, G/=0.224, B/=0.225")
    print("  5. 排列为 NCHW float32: shape (1, 3, 224, 224)")
    print("  输出: (1, 18) → reshape 为 (9, 2) → 每个值乘以 224 得像素坐标")
    print("  关键点顺序: 左眼 右眼 嘴 左耳1 左耳2 左耳3 右耳1 右耳2 右耳3")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='导出猫脸关键点模型为 ONNX')
    parser.add_argument('--weights', default='best_cat_model.pth', help='.pth 权重路径')
    parser.add_argument('--output',  default='cat_landmark.onnx',  help='输出 .onnx 路径')
    parser.add_argument('--opset',   default=12, type=int,         help='ONNX opset 版本 (默认 12)')
    parser.add_argument('--simplify', action='store_true',         help='使用 onnx-simplifier 简化')
    args = parser.parse_args()

    export(args.weights, args.output, args.simplify, args.opset)