using System;
using System.Collections.Generic;
using UnityEngine;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

/// <summary>
/// NanoSAM / MobileSAM Mask Decoder 推理封装
/// 输入: image embedding + 提示点 → 输出: 分割Mask
/// </summary>
public class NanoSAM : IDisposable
{
    // ─── 模型常量（与MobileSAM保持一致）───
    private const int IMAGE_SIZE = 1024;  // SAM原始图像尺寸
    private const int EMBED_DIM = 256;   // 图像embedding通道数
    private const int EMBED_H = 64;    // embedding空间高度 (1024/16)
    private const int EMBED_W = 64;    // embedding空间宽度
    private const int MASK_INPUT_SIZE = 256;   // low-res mask输入尺寸
    private const int NUM_MASK_TOKENS = 4;     // SAM mask token数量

    private InferenceSession _decoderSession;
    private bool _disposed;

    // ─── 构造 ───────────────────────────────────────────
    public NanoSAM(string decoderModelPath)
    {
        var options = new SessionOptions();
        options.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL;
        // 如需GPU: options.AppendExecutionProvider_CUDA(0);

        _decoderSession = new InferenceSession(decoderModelPath, options);

        Debug.Log("[NanoSAM] Decoder loaded.");
        LogSessionInfo(_decoderSession, "Decoder");
    }

    // ─── 主推理接口 ──────────────────────────────────────
    /// <summary>
    /// 运行Mask Decoder推理
    /// </summary>
    /// <param name="imageEmbedding">图像embedding float数组 [1,256,64,64]</param>
    /// <param name="promptPoints">归一化坐标点列表 (x,y) ∈ [0,1]</param>
    /// <param name="promptLabels">点标签: 1=前景, 0=背景</param>
    /// <param name="outputWidth">输出mask目标宽度</param>
    /// <param name="outputHeight">输出mask目标高度</param>
    /// <returns>二值化Mask纹理</returns>
    public Texture2D RunDecoder(
        float[] imageEmbedding,
        List<Vector2> promptPoints,
        List<int> promptLabels,
        int outputWidth,
        int outputHeight)
    {
        if (promptPoints.Count == 0)
            throw new ArgumentException("至少需要一个提示点");

        // —— 构建输入Tensors ——
        var inputs = BuildDecoderInputs(imageEmbedding, promptPoints, promptLabels);

        // —— 运行推理 ——
        using var results = _decoderSession.Run(inputs);

        // —— 解析输出 ——
        // MobileSAM decoder输出: masks [1,4,256,256], iou_predictions [1,4]
        float[] masks = null;
        float[] iouPreds = null;
        int[] masksShape = null;

        foreach (var r in results)
        {
            if (r.Name.Contains("mask") && !r.Name.Contains("iou"))
            {
                var t = r.Value as DenseTensor<float>;
                masks = t.Buffer.ToArray();
                masksShape = new int[] {
                    (int)t.Dimensions[0],
                    (int)t.Dimensions[1],
                    (int)t.Dimensions[2],
                    (int)t.Dimensions[3]
                };
            }
            else if (r.Name.Contains("iou"))
            {
                iouPreds = (r.Value as DenseTensor<float>).Buffer.ToArray();
            }
        }

        if (masks == null)
            throw new Exception("未找到masks输出，请检查模型输出节点名称");

        // —— 选择最高IOU的Mask ——
        int bestIdx = SelectBestMask(iouPreds);
        Debug.Log($"[NanoSAM] Best mask index: {bestIdx}, IOU: {iouPreds[bestIdx]:F3}");

        // —— 提取并上采样Mask ——
        int mH = masksShape[2];
        int mW = masksShape[3];
        float[] bestMask = ExtractMask(masks, bestIdx, mH, mW);

        return BuildMaskTexture(bestMask, mW, mH, outputWidth, outputHeight);
    }

    // ─── 构建Decoder输入 ────────────────────────────────
    private List<NamedOnnxValue> BuildDecoderInputs(
        float[] imageEmbedding,
        List<Vector2> points,
        List<int> labels)
    {
        int numPoints = points.Count;

        // MobileSAM decoder期望的输入节点(请根据实际模型调整节点名):
        // 1. image_embeddings     [1, 256, 64, 64]
        // 2. point_coords         [1, N+1, 2]    (多一个padding点)
        // 3. point_labels         [1, N+1]       (padding标签=-1)
        // 4. mask_input           [1, 1, 256, 256]
        // 5. has_mask_input       [1]

        // —— 1. image_embeddings ——
        var embTensor = new DenseTensor<float>(
            imageEmbedding,
            new[] { 1, EMBED_DIM, EMBED_H, EMBED_W });

        // —— 2. point_coords: SAM坐标系(原图像素坐标，非归一化) ——
        int totalPts = numPoints + 1; // +1 padding
        float[] coordData = new float[1 * totalPts * 2];
        for (int i = 0; i < numPoints; i++)
        {
            // 将归一化坐标转为SAM图像坐标 [0, IMAGE_SIZE]
            coordData[i * 2] = points[i].x * IMAGE_SIZE;
            coordData[i * 2 + 1] = points[i].y * IMAGE_SIZE;
        }
        // padding点放在左上角
        coordData[numPoints * 2] = 0f;
        coordData[numPoints * 2 + 1] = 0f;

        var coordTensor = new DenseTensor<float>(
            coordData,
            new[] { 1, totalPts, 2 });

        // —— 3. point_labels ——
        float[] labelData = new float[1 * totalPts];
        for (int i = 0; i < numPoints; i++)
            labelData[i] = labels[i];
        labelData[numPoints] = -1f; // padding

        var labelTensor = new DenseTensor<float>(
            labelData,
            new[] { 1, totalPts });

        // —— 4. mask_input (全零，表示无先验mask) ——
        float[] maskInputData = new float[1 * 1 * MASK_INPUT_SIZE * MASK_INPUT_SIZE];
        var maskInputTensor = new DenseTensor<float>(
            maskInputData,
            new[] { 1, 1, MASK_INPUT_SIZE, MASK_INPUT_SIZE });

        // —— 5. has_mask_input ——
        var hasMaskTensor = new DenseTensor<float>(
            new float[] { 0f },
            new[] { 1 });

        // ⚠️ 以下节点名基于标准MobileSAM导出，如不匹配请用LogSessionInfo打印实际名称
        return new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("image_embeddings", embTensor),
            NamedOnnxValue.CreateFromTensor("point_coords",     coordTensor),
            NamedOnnxValue.CreateFromTensor("point_labels",     labelTensor),
            NamedOnnxValue.CreateFromTensor("mask_input",       maskInputTensor),
            NamedOnnxValue.CreateFromTensor("has_mask_input",   hasMaskTensor),
        };
    }

    // ─── 工具方法 ────────────────────────────────────────
    private int SelectBestMask(float[] iouPreds)
    {
        int best = 0;
        for (int i = 1; i < iouPreds.Length; i++)
            if (iouPreds[i] > iouPreds[best]) best = i;
        return best;
    }

    private float[] ExtractMask(float[] allMasks, int maskIdx, int h, int w)
    {
        float[] result = new float[h * w];
        int offset = maskIdx * h * w;
        Array.Copy(allMasks, offset, result, 0, h * w);
        return result;
    }

    /// <summary>
    /// 将logits mask转为Texture2D，双线性上采样到目标尺寸
    /// </summary>
    private Texture2D BuildMaskTexture(float[] mask, int srcW, int srcH,
                                        int dstW, int dstH)
    {
        var tex = new Texture2D(dstW, dstH, TextureFormat.RGBA32, false);
        Color[] pixels = new Color[dstW * dstH];

        float scaleX = (float)srcW / dstW;
        float scaleY = (float)srcH / dstH;

        for (int y = 0; y < dstH; y++)
        {
            for (int x = 0; x < dstW; x++)
            {
                // 双线性采样
                float sx = (x + 0.5f) * scaleX - 0.5f;
                float sy = (y + 0.5f) * scaleY - 0.5f;
                float val = BilinearSample(mask, srcW, srcH, sx, sy);

                // logits > 0 视为前景
                float alpha = val > 0f ? 1f : 0f;
                // Unity纹理Y轴翻转
                pixels[(dstH - 1 - y) * dstW + x] =
                    new Color(0.2f, 0.8f, 1f, alpha * 0.6f);
            }
        }

        tex.SetPixels(pixels);
        tex.Apply();
        return tex;
    }

    private float BilinearSample(float[] data, int w, int h, float x, float y)
    {
        int x0 = Mathf.Clamp(Mathf.FloorToInt(x), 0, w - 1);
        int y0 = Mathf.Clamp(Mathf.FloorToInt(y), 0, h - 1);
        int x1 = Mathf.Clamp(x0 + 1, 0, w - 1);
        int y1 = Mathf.Clamp(y0 + 1, 0, h - 1);

        float fx = x - x0;
        float fy = y - y0;

        float v00 = data[y0 * w + x0];
        float v10 = data[y0 * w + x1];
        float v01 = data[y1 * w + x0];
        float v11 = data[y1 * w + x1];

        return Mathf.Lerp(
            Mathf.Lerp(v00, v10, fx),
            Mathf.Lerp(v01, v11, fx), fy);
    }

    private void LogSessionInfo(InferenceSession session, string name)
    {
        Debug.Log($"[{name}] ── 输入节点 ──");
        foreach (var kv in session.InputMetadata)
            Debug.Log($"  {kv.Key}: [{string.Join(",", kv.Value.Dimensions)}] {kv.Value.ElementType}");

        Debug.Log($"[{name}] ── 输出节点 ──");
        foreach (var kv in session.OutputMetadata)
            Debug.Log($"  {kv.Key}: [{string.Join(",", kv.Value.Dimensions)}] {kv.Value.ElementType}");
    }

    public void Dispose()
    {
        if (!_disposed)
        {
            _decoderSession?.Dispose();
            _disposed = true;
        }
    }
}