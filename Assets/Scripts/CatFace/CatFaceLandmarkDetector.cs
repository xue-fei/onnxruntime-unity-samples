// =====================================================================
// Unity 猫脸关键点检测推理脚本
// 模型: cat_landmark.onnx (ResNet-50, 9 关键点)
// 依赖: Microsoft.ML.OnnxRuntime >= 1.17
// 输出关键点顺序:
//   0=左眼  1=右眼  2=嘴
//   3=左耳1 4=左耳2 5=左耳3
//   6=右耳1 7=右耳2 8=右耳3
// =====================================================================

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using UnityEngine;

public class CatFaceLandmarkDetector : MonoBehaviour
{
    // ── Inspector 配置 ────────────────────────────────────────────────
    [Header("模型文件")]
    [Tooltip("放在 StreamingAssets/ 下的 onnx 文件名")]
    public string modelFileName = "models/cat_landmark.onnx";

    [Header("可视化 (可选)")]
    [Tooltip("挂载后会在 OnGUI 中绘制关键点，调试用")]
    public bool showDebugGUI = true;

    // ── 常量：与 Python 预处理完全一致 ───────────────────────────────
    private const int INPUT_SIZE = 224;

    // ImageNet 均值 / 标准差，RGB 顺序
    private static readonly float[] MEAN = { 0.485f, 0.456f, 0.406f };
    private static readonly float[] STD = { 0.229f, 0.224f, 0.225f };

    private static readonly string[] LANDMARK_NAMES =
    {
        "左眼", "右眼", "嘴",
        "左耳1", "左耳2", "左耳3",
        "右耳1", "右耳2", "右耳3"
    };

    // ── 私有成员 ──────────────────────────────────────────────────────
    private InferenceSession _session;
    private Vector2[] _lastLandmarks;
    private int _lastOrigW, _lastOrigH;

    // ─────────────────────────────────────────────────────────────────
    // 生命周期
    // ─────────────────────────────────────────────────────────────────
    private void Awake()
    {
        LoadModel();
    }

    private void OnDestroy()
    {
        _session?.Dispose();
    }

    // ─────────────────────────────────────────────────────────────────
    // 公开 API
    // ─────────────────────────────────────────────────────────────────

    /// <summary>
    /// 对输入 Texture2D 做关键点检测。
    /// 返回 9 个关键点在原图像素坐标系中的位置（左上角原点，Y 向下）。
    /// </summary>
    public Vector2[] Detect(Texture2D srcTexture)
    {
        if (_session == null)
        {
            Debug.LogError("[CatLandmark] 模型尚未加载！");
            return null;
        }

        _lastOrigW = srcTexture.width;
        _lastOrigH = srcTexture.height;

        float[] inputData = Preprocess(srcTexture);
        float[] rawOutput = RunInference(inputData);   // 长度 18
        _lastLandmarks = RestoreCoordinates(rawOutput, _lastOrigW, _lastOrigH);
        return _lastLandmarks;
    }

    // ─────────────────────────────────────────────────────────────────
    // 模型加载
    // ─────────────────────────────────────────────────────────────────
    private void LoadModel()
    {
        string modelPath = Path.Combine(Application.streamingAssetsPath, modelFileName);
        if (!File.Exists(modelPath))
            throw new FileNotFoundException($"[CatLandmark] 找不到模型: {modelPath}");

        var opts = new SessionOptions();
        opts.GraphOptimizationLevel = GraphOptimizationLevel.ORT_ENABLE_ALL; 
        opts.AppendExecutionProvider_CUDA(0); 
        _session = new InferenceSession(modelPath, opts);
        Debug.Log($"[CatLandmark] 模型加载成功: {modelPath}");
    }

    // ─────────────────────────────────────────────────────────────────
    // 预处理
    //   输入: 任意尺寸 Texture2D (RGB)
    //   输出: float[3 * 224 * 224]  NCHW，已做 ImageNet 归一化
    //
    //   与 predict.py transforms.Compose 完全一致:
    //     Resize(224) → ToTensor(÷255) → Normalize(mean, std)
    // ─────────────────────────────────────────────────────────────────
    private float[] Preprocess(Texture2D src)
    {
        // Step 1: GPU Blit resize 到 224×224
        RenderTexture rt = RenderTexture.GetTemporary(
            INPUT_SIZE, INPUT_SIZE, 0, RenderTextureFormat.ARGB32);
        Graphics.Blit(src, rt);
        RenderTexture.active = rt;

        Texture2D resized = new Texture2D(INPUT_SIZE, INPUT_SIZE, TextureFormat.RGB24, false);
        resized.ReadPixels(new Rect(0, 0, INPUT_SIZE, INPUT_SIZE), 0, 0);
        resized.Apply();

        RenderTexture.active = null;
        RenderTexture.ReleaseTemporary(rt);

        Color32[] pixels = resized.GetPixels32();
        Destroy(resized);

        // Step 2: NCHW float + ImageNet 归一化
        // Unity GetPixels32 (0,0) 在左下角，需垂直翻转
        float[] data = new float[3 * INPUT_SIZE * INPUT_SIZE];
        int area = INPUT_SIZE * INPUT_SIZE;

        for (int row = 0; row < INPUT_SIZE; row++)
        {
            int srcRow = INPUT_SIZE - 1 - row;  // 翻转 Y 轴
            for (int col = 0; col < INPUT_SIZE; col++)
            {
                Color32 c = pixels[srcRow * INPUT_SIZE + col];

                float r = (c.r / 255f - MEAN[0]) / STD[0];
                float g = (c.g / 255f - MEAN[1]) / STD[1];
                float b = (c.b / 255f - MEAN[2]) / STD[2];

                int idx = row * INPUT_SIZE + col;
                data[0 * area + idx] = r;   // channel R
                data[1 * area + idx] = g;   // channel G
                data[2 * area + idx] = b;   // channel B
            }
        }

        return data;
    }

    // ─────────────────────────────────────────────────────────────────
    // ONNX 推理
    //
    // 修复说明（对应三处编译错误）：
    //   错误1/2: DenseTensor<T>(float[], long[]) 构造函数不存在
    //     → 改用 DenseTensor<T>(Memory<T>, ReadOnlySpan<int>)
    //       其中 dims 用 int[] 而非 long[]
    //   错误3: Tensor<T>.Buffer 属性不存在
    //     → 改用 .ToArray()，所有 OnnxRuntime 版本均支持
    // ─────────────────────────────────────────────────────────────────
    private float[] RunInference(float[] inputData)
    {
        // ✅ 修复1/2：Memory<float> + int[] 维度，不再使用 long[]
        var memory = new System.Memory<float>(inputData);
        int[] dims = { 1, 3, INPUT_SIZE, INPUT_SIZE };
        var tensor = new DenseTensor<float>(memory, dims);

        var inputs = new List<NamedOnnxValue>
        {
            NamedOnnxValue.CreateFromTensor("input", tensor)
        };

        using var results = _session.Run(inputs);

        // ✅ 修复3：用 .ToArray() 替代 .Buffer.Span
        var outputTensor = results.First(r => r.Name == "landmarks").AsTensor<float>();
        return outputTensor.ToArray();   // float[18]
    }

    // ─────────────────────────────────────────────────────────────────
    // 坐标还原
    //   输出 18 个 [0,1] 归一化坐标 → 原图像素坐标
    //   对应 predict.py: pred.reshape(9,2) * IMG_SIZE → 再按原图缩放
    // ─────────────────────────────────────────────────────────────────
    private static Vector2[] RestoreCoordinates(float[] raw, int origW, int origH)
    {
        var pts = new Vector2[9];
        for (int i = 0; i < 9; i++)
        {
            pts[i] = new Vector2(
                raw[i * 2 + 0] * origW,
                raw[i * 2 + 1] * origH);
        }
        return pts;
    }

    // ─────────────────────────────────────────────────────────────────
    // 调试：在 Game 视图叠加显示关键点
    // ─────────────────────────────────────────────────────────────────
    private void OnGUI()
    {
        if (!showDebugGUI || _lastLandmarks == null) return;

        float scaleX = Screen.width / (float)_lastOrigW;
        float scaleY = Screen.height / (float)_lastOrigH;

        GUIStyle style = new GUIStyle(GUI.skin.label) { fontSize = 10 };

        for (int i = 0; i < _lastLandmarks.Length; i++)
        {
            float sx = _lastLandmarks[i].x * scaleX;
            float sy = _lastLandmarks[i].y * scaleY;

            GUI.color = GetLandmarkColor(i);
            GUI.DrawTexture(new Rect(sx - 4, sy - 4, 8, 8), Texture2D.whiteTexture);

            GUI.color = Color.white;
            GUI.Label(new Rect(sx + 5, sy - 6, 60, 16), LANDMARK_NAMES[i], style);
        }

        GUI.color = Color.white;
    }

    private static Color GetLandmarkColor(int idx) => idx switch
    {
        0 or 1 => Color.cyan,
        2 => Color.yellow,
        3 or 4 or 5 => Color.green,
        _ => Color.magenta
    };
}

// ─────────────────────────────────────────────────────────────────────────────
// 结果结构体（供外部调用者使用）
// ─────────────────────────────────────────────────────────────────────────────
[Serializable]
public struct CatLandmarkResult
{
    /// <summary>9 个关键点，原图像素坐标（左上角原点，Y 向下）</summary>
    public Vector2[] Points;

    public static readonly string[] Names =
    {
        "左眼", "右眼", "嘴",
        "左耳1", "左耳2", "左耳3",
        "右耳1", "右耳2", "右耳3"
    };

    public override string ToString()
    {
        if (Points == null) return "CatLandmarkResult(empty)";
        var sb = new System.Text.StringBuilder();
        for (int i = 0; i < Points.Length; i++)
            sb.AppendLine($"  [{i}] {Names[i]}: ({Points[i].x:F1}, {Points[i].y:F1})");
        return sb.ToString();
    }
}