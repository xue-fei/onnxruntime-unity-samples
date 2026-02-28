using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;
using System.IO;

/// <summary>
/// resnet18_image_encoder.onnx https://github.com/NVIDIA-AI-IOT/nanosam/issues/41
/// mobile_sam_mask_decoder.onnx https://huggingface.co/dragonSwing/nanosam/tree/main
/// </summary>
public class NanoSAMDemo : MonoBehaviour
{
    [Header("模型路径（相对StreamingAssets）")]
    public string decoderModelName = "models/NanoSAM/mobile_sam_mask_decoder.onnx";
    public string encoderModelName = "models/NanoSAM/resnet18_image_encoder.onnx"; // 可选

    [Header("UI引用")]
    public RawImage targetImage;      // 显示原始图像
    public RawImage maskOverlay;      // 显示分割Mask（叠加在targetImage上）
    public Text statusText;

    [Header("测试图像")]
    public Texture2D testTexture;

    // ─── 私有成员 ───
    private NanoSAM _segmenter;
    private Microsoft.ML.OnnxRuntime.InferenceSession _encoderSession;
    private float[] _cachedEmbedding;   // 缓存当前图像的embedding
    private Texture2D _currentMaskTex;

    private List<Vector2> _promptPoints = new List<Vector2>();
    private List<int> _promptLabels = new List<int>();

    // ─── 生命周期 ────────────────────────────────────────
    void Start()
    {
        if (testTexture != null)
        {
            targetImage.texture = testTexture;
            maskOverlay.texture = testTexture;
        }
        StartCoroutine(InitializeModels());
    }

    IEnumerator InitializeModels()
    {
        SetStatus("正在加载模型...");
        yield return null;

        string decoderPath = Path.Combine(Application.streamingAssetsPath, decoderModelName);

        if (!File.Exists(decoderPath))
        {
            SetStatus($"❌ 模型未找到: {decoderPath}");
            yield break;
        }

        _segmenter = new NanoSAM(decoderPath);

        // 尝试加载encoder（可选）
        string encoderPath = Path.Combine(Application.streamingAssetsPath, encoderModelName);
        if (File.Exists(encoderPath))
        {
            _encoderSession = new Microsoft.ML.OnnxRuntime.InferenceSession(encoderPath);
            Debug.Log("[Demo] Encoder loaded.");
        }

        SetStatus("✅ 模型加载完成。点击图像进行分割（左键=前景，右键=背景）");

        // 显示测试图像并预计算embedding
        if (testTexture != null)
        {
            targetImage.texture = testTexture;
            yield return StartCoroutine(ComputeEmbedding(testTexture));
        }
    }

    // ─── 计算图像Embedding ───────────────────────────────
    IEnumerator ComputeEmbedding(Texture2D tex)
    {
        SetStatus("正在计算图像Embedding...");
        yield return null;

        if (_encoderSession != null)
        {
            // 使用真实encoder
            float[] preprocessed = ImageEncoder.PreprocessImage(tex);
            _cachedEmbedding = ImageEncoder.RunEncoderOnnx(_encoderSession, preprocessed);
            Debug.Log($"[Demo] Embedding computed, size={_cachedEmbedding.Length}");
        }
        else
        {
            // 没有encoder时使用随机embedding（仅调试用）
            Debug.LogWarning("[Demo] 未找到encoder，使用随机embedding（结果无意义，仅测试流程）");
            _cachedEmbedding = new float[256 * 64 * 64];
            var rng = new System.Random(42);
            for (int i = 0; i < _cachedEmbedding.Length; i++)
                _cachedEmbedding[i] = (float)(rng.NextDouble() * 0.1);
        }

        SetStatus("✅ 准备好。左键点击=前景点，右键=背景点，Space=运行分割，R=重置");
    }

    // ─── 输入处理 ────────────────────────────────────────
    void Update()
    {
        if (_cachedEmbedding == null) return;

        // 鼠标点击收集提示点
        if (Input.GetMouseButtonDown(0) || Input.GetMouseButtonDown(1))
        {
            if (TryGetNormalizedClickPos(out Vector2 normPos))
            {
                int label = Input.GetMouseButtonDown(0) ? 1 : 0;
                _promptPoints.Add(normPos);
                _promptLabels.Add(label);
                Debug.Log($"[Demo] Point added: {normPos}, label={label}");
                DrawPointMarker(normPos, label == 1 ? Color.green : Color.red);
            }
        }

        // Space运行分割
        if (Input.GetKeyDown(KeyCode.Space) && _promptPoints.Count > 0)
        {
            StartCoroutine(RunSegmentation());
        }

        // R重置
        if (Input.GetKeyDown(KeyCode.R))
        {
            ResetPrompts();
        }
    }

    IEnumerator RunSegmentation()
    {
        SetStatus("分割中...");
        yield return null;

        try
        {
            int w = testTexture != null ? testTexture.width : 512;
            int h = testTexture != null ? testTexture.height : 512;

            var maskTex = _segmenter.RunDecoder(
                _cachedEmbedding,
                _promptPoints,
                _promptLabels,
                w, h);

            if (_currentMaskTex != null)
                Destroy(_currentMaskTex);
            _currentMaskTex = maskTex;

            maskOverlay.texture = maskTex;
            SetStatus($"✅ 分割完成 ({_promptPoints.Count}个提示点)");
        }
        catch (System.Exception e)
        {
            SetStatus($"❌ 分割失败: {e.Message}");
            Debug.LogException(e);
        }
    }

    // ─── 工具 ────────────────────────────────────────────
    bool TryGetNormalizedClickPos(out Vector2 normPos)
    {
        normPos = Vector2.zero;
        if (targetImage == null) return false;

        var rectT = targetImage.rectTransform;
        if (!RectTransformUtility.ScreenPointToLocalPointInRectangle(
                rectT, Input.mousePosition,
                null, out Vector2 localPos)) return false;

        Rect rect = rectT.rect;
        normPos = new Vector2(
            (localPos.x - rect.x) / rect.width,
            (localPos.y - rect.y) / rect.height);

        // 过滤越界点
        if (normPos.x < 0 || normPos.x > 1 || normPos.y < 0 || normPos.y > 1)
            return false;

        return true;
    }

    void DrawPointMarker(Vector2 normPos, Color color)
    {
        // 简单在maskOverlay上标记（可扩展为UI点标记）
        Debug.Log($"[Demo] Marker at {normPos} color={color}");
    }

    void ResetPrompts()
    {
        _promptPoints.Clear();
        _promptLabels.Clear();
        if (maskOverlay != null)
        {
            //maskOverlay.texture = null;
        }
        SetStatus("已重置。重新点击添加提示点");
    }

    void SetStatus(string msg)
    {
        Debug.Log("[Demo] " + msg);
        if (statusText != null) statusText.text = msg;
    }

    // ─── 清理 ────────────────────────────────────────────
    void OnDestroy()
    {
        _segmenter?.Dispose();
        _encoderSession?.Dispose();
        if (_currentMaskTex != null) Destroy(_currentMaskTex);
    }
}