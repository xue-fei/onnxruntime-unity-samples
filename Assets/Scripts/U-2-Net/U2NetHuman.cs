using System;
using System.Collections.Generic;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using UnityEngine;

/// <summary>
/// U2Net 人体分割推理器
/// 对应 Python u2net_human_seg.py 的完整推理流程：
///   预处理(RescaleT 320 + ToTensorLab) → 推理 → normPRED → 还原尺寸
/// </summary>
public partial class U2NetHuman : IDisposable
{
    // ── 模型输入尺寸（与 Python RescaleT(320) 对应）─────────────────
    private const int INPUT_SIZE = 320;

    // ── ImageNet 归一化均值/标准差（与 ToTensorLab 对应）─────────────
    private static readonly float[] MEAN = { 0.485f, 0.456f, 0.406f };
    private static readonly float[] STD = { 0.229f, 0.224f, 0.225f };

    private InferenceSession _session;
    private bool _disposed;

    // ── 输入/输出节点名（u2net_human_seg.onnx 标准命名）─────────────
    private string _inputName;
    private string _outputName;

    // ────────────────────────────────────────────────────────────────
    #region 初始化

    /// <summary>
    /// 从文件路径加载 ONNX 模型
    /// </summary>
    public U2NetHuman(string onnxModelPath)
    {
        if (string.IsNullOrEmpty(onnxModelPath))
            throw new ArgumentNullException(nameof(onnxModelPath));

        var options = new SessionOptions();
        options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        options.AppendExecutionProvider_CUDA(0);
        _session = new InferenceSession(onnxModelPath, options);
        _inputName = _session.InputNames[0];
        _outputName = _session.OutputNames[0];  // d1 — 第一个（最精细）输出

        Debug.Log($"[U2Net] 模型加载成功\n"
                + $"  输入: {_inputName}  {string.Join("x", _session.InputMetadata[_inputName].Dimensions)}\n"
                + $"  输出: {_outputName} {string.Join("x", _session.OutputMetadata[_outputName].Dimensions)}");
    }

    /// <summary>
    /// 从字节数组加载模型（适合从 Resources/StreamingAssets 读取后传入）
    /// </summary>
    public U2NetHuman(byte[] modelBytes)
    {
        if (modelBytes == null || modelBytes.Length == 0)
            throw new ArgumentNullException(nameof(modelBytes));

        var options = new SessionOptions();
        options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        options.AppendExecutionProvider_CUDA(0);
        _session = new InferenceSession(modelBytes, options);
        _inputName = _session.InputNames[0];
        _outputName = _session.OutputNames[0];
    }

    #endregion

    // ────────────────────────────────────────────────────────────────
    #region 主推理接口

    /// <summary>
    /// 对输入 Texture2D 进行人体分割。
    /// 返回与原图同尺寸的灰度 mask（0=背景，255=人体）。
    /// </summary>
    /// <param name="sourceTexture">原始图像（任意尺寸，RGBA32 或 RGB24）</param>
    /// <returns>与 sourceTexture 等尺寸的 Texture2D（R8 格式灰度 mask）</returns>
    public Texture2D Predict(Texture2D sourceTexture)
    {
        if (sourceTexture == null) throw new ArgumentNullException(nameof(sourceTexture));

        int origW = sourceTexture.width;
        int origH = sourceTexture.height;

        // 1. 预处理：RescaleT(320) + ToTensorLab
        float[] inputTensor = Preprocess(sourceTexture);

        // 2. 推理
        float[] rawPred = RunInference(inputTensor);

        // 3. normPRED（对应 Python normPRED）
        NormalizePrediction(rawPred);

        // 4. 还原为原始尺寸，生成 mask Texture2D
        return BuildMaskTexture(rawPred, INPUT_SIZE, INPUT_SIZE, origW, origH);
    }

    /// <summary>
    /// 返回归一化后的 float[] mask（值域 [0,1]，尺寸 INPUT_SIZE×INPUT_SIZE）
    /// 适合需要进一步处理的场景（如抠图合成）
    /// </summary>
    public float[] PredictRaw(Texture2D sourceTexture)
    {
        if (sourceTexture == null) throw new ArgumentNullException(nameof(sourceTexture));
        float[] inputTensor = Preprocess(sourceTexture);
        float[] rawPred = RunInference(inputTensor);
        NormalizePrediction(rawPred);
        return rawPred;
    }

    #endregion

    // ────────────────────────────────────────────────────────────────
    #region 预处理（对应 Python RescaleT + ToTensorLab）

    /// <summary>
    /// RescaleT(320)：双线性缩放到 320×320
    /// ToTensorLab(flag=0)：归一化到 [0,1] 再减均值除标准差，转 CHW float32
    /// 返回 float[1, 3, 320, 320]（展开为一维数组）
    /// </summary>
    private float[] Preprocess(Texture2D src)
    {
        // ── 双线性缩放到 320×320 ─────────────────────────────────────
        // Unity Texture2D 直接用 GPU 拉伸：创建 RenderTexture，blit，再 ReadPixels
        RenderTexture rt = RenderTexture.GetTemporary(INPUT_SIZE, INPUT_SIZE, 0,
                                                      RenderTextureFormat.ARGB32,
                                                      RenderTextureReadWrite.Linear);
        Graphics.Blit(src, rt);
        RenderTexture prev = RenderTexture.active;
        RenderTexture.active = rt;

        Texture2D scaled = new Texture2D(INPUT_SIZE, INPUT_SIZE, TextureFormat.RGB24, false);
        scaled.ReadPixels(new Rect(0, 0, INPUT_SIZE, INPUT_SIZE), 0, 0);
        scaled.Apply();

        RenderTexture.active = prev;
        RenderTexture.ReleaseTemporary(rt);

        Color32[] pixels = scaled.GetPixels32();
        UnityEngine.Object.Destroy(scaled);

        // ── ToTensorLab: pixels → CHW float32, 归一化 ────────────────
        // Python:
        //   tmpImg = tmpImg / np.max(tmpImg)         ← 先归到[0,1]
        //   tmpImg -= mean; tmpImg /= std             ← ImageNet 标准化
        //   转 CHW

        int hw = INPUT_SIZE * INPUT_SIZE;
        float[] tensor = new float[3 * hw];   // [C, H, W] 展开

        // Unity GetPixels32 返回顺序：左下角起、逐行向上（与图像 Y 轴相反）
        // U2Net 训练时图像 Y 从上到下，所以这里需要垂直翻转
        for (int row = 0; row < INPUT_SIZE; row++)
        {
            // Unity row 0 = 图像底部，需要翻转
            int srcRow = INPUT_SIZE - 1 - row;

            for (int col = 0; col < INPUT_SIZE; col++)
            {
                Color32 c = pixels[srcRow * INPUT_SIZE + col];

                float r = c.r / 255.0f;
                float g = c.g / 255.0f;
                float b = c.b / 255.0f;

                // ImageNet 标准化
                r = (r - MEAN[0]) / STD[0];
                g = (g - MEAN[1]) / STD[1];
                b = (b - MEAN[2]) / STD[2];

                int pixIdx = row * INPUT_SIZE + col;
                tensor[0 * hw + pixIdx] = r;   // R channel
                tensor[1 * hw + pixIdx] = g;   // G channel
                tensor[2 * hw + pixIdx] = b;   // B channel
            }
        }

        return tensor;
    }

    #endregion

    private const int INPUT_SIZE_EXT = 320;
    /// <summary>
    /// 从已预处理的 float[] tensor（shape [1,3,320,320]）直接推理。
    /// 此方法线程安全，可在非主线程调用。
    /// 返回归一化后的 float[] mask，shape [320*320]，值域 [0,1]。
    /// </summary>
    public float[] PredictRaw_FromTensor(float[] inputTensor)
    {
        var dims = new int[] { 1, 3, INPUT_SIZE_EXT, INPUT_SIZE_EXT };
        var tensor = new DenseTensor<float>(inputTensor, dims);

        var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(_inputName, tensor)
            };

        using var results = _session.Run(inputs);
        var output = results.First(r => r.Name == _outputName);
        float[] rawPred = output.AsTensor<float>().ToArray();

        NormalizePrediction(rawPred);
        return rawPred;
    }

    // ────────────────────────────────────────────────────────────────
    #region 推理

    private float[] RunInference(float[] inputData)
    {
        // 创建 ONNX 输入 tensor：shape [1, 3, 320, 320]
        var dims = new int[] { 1, 3, INPUT_SIZE, INPUT_SIZE };
        var tensor = new DenseTensor<float>(inputData, dims);

        var inputs = new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor(_inputName, tensor)
            };

        // 推理，只取第一个输出（d1，最精细）
        using var results = _session.Run(inputs);
        var output = results.First(r => r.Name == _outputName);
        var outTensor = output.AsTensor<float>();

        // 输出 shape: [1, 1, 320, 320]，取内容
        return outTensor.ToArray();
    }

    #endregion

    // ────────────────────────────────────────────────────────────────
    #region 后处理（对应 Python normPRED + resize + save_output）

    /// <summary>
    /// 对应 Python normPRED：(d - min) / (max - min)
    /// 就地归一化到 [0, 1]
    /// </summary>
    private static void NormalizePrediction(float[] pred)
    {
        float max = float.MinValue;
        float min = float.MaxValue;

        foreach (float v in pred)
        {
            if (v > max) max = v;
            if (v < min) min = v;
        }

        float range = max - min;
        if (range < 1e-8f) range = 1e-8f;   // 防除零

        for (int i = 0; i < pred.Length; i++)
            pred[i] = (pred[i] - min) / range;
    }

    /// <summary>
    /// 将 320×320 的 float mask 双线性缩放回原始尺寸，
    /// 返回 Texture2D（RGBA32）：
    ///   RGB = 原图像素（透明区域置黑）
    ///   A   = mask 值（0=背景透明，255=人体不透明）
    ///
    /// 与 Python save_output 的行为对应（BILINEAR resize）
    /// </summary>
    private static Texture2D BuildMaskTexture(float[] pred,
                                               int predW, int predH,
                                               int origW, int origH)
    {
        // 双线性上采样 pred 到 origW × origH
        Texture2D maskTex = new Texture2D(origW, origH, TextureFormat.RGBA32, false);
        Color32[] outPixels = new Color32[origW * origH];

        for (int y = 0; y < origH; y++)
        {
            // Unity 纹理 Y 轴从下到上，pred 数组从上到下
            // → 翻转 y
            float srcYf = (origH - 1 - y) / (float)(origH - 1) * (predH - 1);
            int srcY0 = Mathf.FloorToInt(srcYf);
            int srcY1 = Mathf.Min(srcY0 + 1, predH - 1);
            float fy = srcYf - srcY0;

            for (int x = 0; x < origW; x++)
            {
                float srcXf = x / (float)(origW - 1) * (predW - 1);
                int srcX0 = Mathf.FloorToInt(srcXf);
                int srcX1 = Mathf.Min(srcX0 + 1, predW - 1);
                float fx = srcXf - srcX0;

                // 双线性插值
                float v00 = pred[srcY0 * predW + srcX0];
                float v01 = pred[srcY0 * predW + srcX1];
                float v10 = pred[srcY1 * predW + srcX0];
                float v11 = pred[srcY1 * predW + srcX1];

                float val = v00 * (1 - fx) * (1 - fy)
                          + v01 * fx * (1 - fy)
                          + v10 * (1 - fx) * fy
                          + v11 * fx * fy;

                byte alpha = (byte)(Mathf.Clamp01(val) * 255f);
                outPixels[y * origW + x] = new Color32(alpha, alpha, alpha, 255);
            }
        }

        maskTex.SetPixels32(outPixels);
        maskTex.Apply();
        return maskTex;
    }

    #endregion

    // ────────────────────────────────────────────────────────────────
    #region IDisposable

    public void Dispose()
    {
        if (!_disposed)
        {
            _session?.Dispose();
            _disposed = true;
        }
    }

    #endregion
}