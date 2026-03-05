using System;
using System.Collections;
using System.IO;
using UnityEngine;
using UnityEngine.UI;

/// <summary>
/// Unity MonoBehaviour 封装
/// 挂到场景任意 GameObject，通过 Inspector 配置路径和 UI 引用即可使用。
///
/// 使用流程：
///   1. 将 u2net_human_seg.onnx 放到 StreamingAssets/Models/
///   2. 将本脚本挂到 GameObject
///   3. 在 Inspector 配置 SourceImage（待分割原图）和显示用的 RawImage
///   4. 运行或调用 RunSegmentation()
/// </summary>
public class U2NetHumanDemo : MonoBehaviour
{
    // ── Inspector 配置 ───────────────────────────────────────────────
    [Header("模型路径（相对 StreamingAssets）")]
    [SerializeField] private string modelFileName = "models/U2Net/u2net_human_seg.onnx";

    [Header("输入图像")]
    [Tooltip("待分割的人体图像 Texture2D（运行时赋值或 Inspector 指定）")]
    [SerializeField] public Texture2D sourceTexture;

    [Header("UI 显示（可选）")]
    [SerializeField] private RawImage displayOriginal;   // 显示原图
    [SerializeField] private RawImage displayMask;       // 显示灰度 mask
    [SerializeField] private RawImage displayCutout;     // 显示抠图结果

    [Header("自动启动")]
    [SerializeField] private bool runOnStart = false;

    // ── 私有状态 ─────────────────────────────────────────────────────
    private U2NetHuman _segmentor;
    private bool _modelLoaded;
    private bool _isRunning;

    // ── 对外事件 ─────────────────────────────────────────────────────
    /// <summary>分割完成后触发，参数为 mask Texture2D（灰度）</summary>
    public event Action<Texture2D> OnSegmentationComplete;
    /// <summary>抠图完成后触发，参数为带透明通道的 Texture2D</summary>
    public event Action<Texture2D> OnCutoutComplete;

    // ── 生命周期 ─────────────────────────────────────────────────────
    private IEnumerator Start()
    {
        yield return LoadModelAsync();

        if (runOnStart && sourceTexture != null)
            RunSegmentation();
    }

    private void OnDestroy()
    {
        _segmentor?.Dispose();
    }

    // ── 模型加载 ─────────────────────────────────────────────────────

    /// <summary>
    /// 异步从 StreamingAssets 加载 ONNX 模型。
    /// Android / WebGL 需要用 UnityWebRequest，PC 直接读文件。
    /// </summary>
    private IEnumerator LoadModelAsync()
    {
        string modelPath = Path.Combine(Application.streamingAssetsPath, modelFileName);
        Debug.Log($"[U2Net] 加载模型: {modelPath}");

        // PC / Editor：直接读文件
        if (!File.Exists(modelPath))
        {
            Debug.LogError($"[U2Net] 找不到模型文件: {modelPath}\n"
                         + $"请将 u2net_human_seg.onnx 放到 StreamingAssets/{modelFileName}");
            Debug.LogError("https://github.com/danielgatis/rembg/releases/download/v0.0.0/u2net_human_seg.onnx");
            yield break;
        }
        // 文件 IO 放到子线程，避免主线程卡顿（模型约 173 MB）
        byte[] modelBytes = null;
        bool loadDone = false;
        System.Threading.ThreadPool.QueueUserWorkItem(_ =>
        {
            modelBytes = File.ReadAllBytes(modelPath);
            loadDone = true;
        });
        yield return new WaitUntil(() => loadDone);
        _segmentor = new U2NetHuman(modelBytes);

        _modelLoaded = true;
        Debug.Log("[U2Net] 模型加载完成，可以开始推理");
    }

    // ── 推理入口 ─────────────────────────────────────────────────────

    /// <summary>
    /// 对 sourceTexture 执行人体分割（在主线程同步推理，适合单张图）。
    /// 建议在 Editor 或 PC 低延迟场景使用；移动端请用 RunSegmentationAsync。
    /// </summary>
    public void RunSegmentation()
    {
        if (!_modelLoaded)
        {
            Debug.LogWarning("[U2Net] 模型尚未加载完成");
            return;
        }
        if (sourceTexture == null)
        {
            Debug.LogWarning("[U2Net] 未设置 sourceTexture");
            return;
        }
        if (_isRunning)
        {
            Debug.LogWarning("[U2Net] 推理进行中，请等待");
            return;
        }

        StartCoroutine(RunSegmentationCoroutine(sourceTexture));
    }

    /// <summary>
    /// 对外部传入的 Texture2D 执行分割
    /// </summary>
    public void RunSegmentation(Texture2D texture)
    {
        sourceTexture = texture;
        RunSegmentation();
    }

    // ── 推理协程（推理放子线程，Unity API 调用回主线程）───────────────

    private IEnumerator RunSegmentationCoroutine(Texture2D src)
    {
        _isRunning = true;
        Debug.Log($"[U2Net] 开始推理: {src.width}×{src.height}");

        // 显示原图
        if (displayOriginal != null) displayOriginal.texture = src;

        // ── 预处理（主线程，需要 Unity Texture API）──────────────────
        float[] rawPred = null;
        bool inferDone = false;
        int origW = src.width;
        int origH = src.height;

        // 在主线程做预处理（GetPixels32、RenderTexture 必须主线程）
        float[] inputTensor = null;
        try
        {
            inputTensor = PreprocessOnMainThread(src);
        }
        catch (Exception e)
        {
            Debug.LogError($"[U2Net] 预处理失败: {e}");
            _isRunning = false;
            yield break;
        }

        // ── 推理放子线程（ONNX Runtime 线程安全）────────────────────
        System.Threading.ThreadPool.QueueUserWorkItem(_ =>
        {
            try
            {
                rawPred = _segmentor.PredictRaw_FromTensor(inputTensor);
            }
            catch (Exception e)
            {
                Debug.LogError($"[U2Net] 推理失败: {e}");
            }
            finally
            {
                inferDone = true;
            }
        });

        yield return new WaitUntil(() => inferDone);

        if (rawPred == null)
        {
            _isRunning = false;
            yield break;
        }

        // ── 后处理（主线程，生成 Texture2D）─────────────────────────
        Texture2D maskTex = BuildMaskOnMainThread(rawPred, origW, origH);
        Texture2D cutoutTex = BuildCutoutOnMainThread(src, rawPred, origW, origH);

        // 更新 UI
        if (displayMask != null) displayMask.texture = maskTex;
        if (displayCutout != null) displayCutout.texture = cutoutTex;

        // 触发事件
        OnSegmentationComplete?.Invoke(maskTex);
        OnCutoutComplete?.Invoke(cutoutTex);

        Debug.Log("[U2Net] 推理完成");

        displayOriginal.SetNativeSize();
        displayMask.SetNativeSize();
        displayCutout.SetNativeSize();

        _isRunning = false;
    }

    // ── 预处理（主线程版，供协程调用）──────────────────────────────────

    private static readonly float[] MEAN = { 0.485f, 0.456f, 0.406f };
    private static readonly float[] STD = { 0.229f, 0.224f, 0.225f };
    private const int INPUT_SIZE = 320;

    private float[] PreprocessOnMainThread(Texture2D src)
    {
        // 双线性缩放到 320×320
        RenderTexture rt = RenderTexture.GetTemporary(
            INPUT_SIZE, INPUT_SIZE, 0,
            RenderTextureFormat.ARGB32, RenderTextureReadWrite.Linear);
        Graphics.Blit(src, rt);

        RenderTexture prev = RenderTexture.active;
        RenderTexture.active = rt;
        Texture2D scaled = new Texture2D(INPUT_SIZE, INPUT_SIZE, TextureFormat.RGB24, false);
        scaled.ReadPixels(new Rect(0, 0, INPUT_SIZE, INPUT_SIZE), 0, 0);
        scaled.Apply();
        RenderTexture.active = prev;
        RenderTexture.ReleaseTemporary(rt);

        Color32[] pixels = scaled.GetPixels32();
        Destroy(scaled);

        int hw = INPUT_SIZE * INPUT_SIZE;
        float[] tensor = new float[3 * hw];

        for (int row = 0; row < INPUT_SIZE; row++)
        {
            int srcRow = INPUT_SIZE - 1 - row;   // Unity Y 轴翻转
            for (int col = 0; col < INPUT_SIZE; col++)
            {
                Color32 c = pixels[srcRow * INPUT_SIZE + col];
                float r = (c.r / 255f - MEAN[0]) / STD[0];
                float g = (c.g / 255f - MEAN[1]) / STD[1];
                float b = (c.b / 255f - MEAN[2]) / STD[2];
                int idx = row * INPUT_SIZE + col;
                tensor[0 * hw + idx] = r;
                tensor[1 * hw + idx] = g;
                tensor[2 * hw + idx] = b;
            }
        }
        return tensor;
    }

    // ── 后处理辅助 ──────────────────────────────────────────────────

    /// <summary>
    /// 生成灰度 mask Texture2D（RGB 均为 mask 值，A=255）
    /// </summary>
    private Texture2D BuildMaskOnMainThread(float[] pred, int origW, int origH)
    {
        Texture2D tex = new Texture2D(origW, origH, TextureFormat.RGBA32, false);
        Color32[] px = new Color32[origW * origH];

        for (int y = 0; y < origH; y++)
        {
            float srcYf = (origH - 1 - y) / (float)(origH - 1) * (INPUT_SIZE - 1);
            int y0 = Mathf.FloorToInt(srcYf);
            int y1 = Mathf.Min(y0 + 1, INPUT_SIZE - 1);
            float fy = srcYf - y0;

            for (int x = 0; x < origW; x++)
            {
                float srcXf = x / (float)(origW - 1) * (INPUT_SIZE - 1);
                int x0 = Mathf.FloorToInt(srcXf);
                int x1 = Mathf.Min(x0 + 1, INPUT_SIZE - 1);
                float fx = srcXf - x0;

                float v = pred[y0 * INPUT_SIZE + x0] * (1 - fx) * (1 - fy)
                        + pred[y0 * INPUT_SIZE + x1] * fx * (1 - fy)
                        + pred[y1 * INPUT_SIZE + x0] * (1 - fx) * fy
                        + pred[y1 * INPUT_SIZE + x1] * fx * fy;

                byte val = (byte)(Mathf.Clamp01(v) * 255f);
                px[y * origW + x] = new Color32(val, val, val, 255);
            }
        }

        tex.SetPixels32(px);
        tex.Apply();
        return tex;
    }

    /// <summary>
    /// 生成抠图 Texture2D：原图 RGB + mask 作为 Alpha 通道
    /// 背景透明（Alpha=0），人体保留（Alpha=mask值）
    /// </summary>
    private Texture2D BuildCutoutOnMainThread(Texture2D src, float[] pred,
                                               int origW, int origH)
    {
        Texture2D tex = new Texture2D(origW, origH, TextureFormat.RGBA32, false);
        Color32[] srcPx = src.GetPixels32();
        Color32[] outPx = new Color32[origW * origH];

        for (int y = 0; y < origH; y++)
        {
            float srcYf = (origH - 1 - y) / (float)(origH - 1) * (INPUT_SIZE - 1);
            int y0 = Mathf.FloorToInt(srcYf);
            int y1 = Mathf.Min(y0 + 1, INPUT_SIZE - 1);
            float fy = srcYf - y0;

            for (int x = 0; x < origW; x++)
            {
                float srcXf = x / (float)(origW - 1) * (INPUT_SIZE - 1);
                int x0 = Mathf.FloorToInt(srcXf);
                int x1 = Mathf.Min(x0 + 1, INPUT_SIZE - 1);
                float fx = srcXf - x0;

                float v = pred[y0 * INPUT_SIZE + x0] * (1 - fx) * (1 - fy)
                        + pred[y0 * INPUT_SIZE + x1] * fx * (1 - fy)
                        + pred[y1 * INPUT_SIZE + x0] * (1 - fx) * fy
                        + pred[y1 * INPUT_SIZE + x1] * fx * fy;

                byte alpha = (byte)(Mathf.Clamp01(v) * 255f);

                // Unity GetPixels32 Y 轴：pixel[0] 在左下角
                // src 和 out 都是 Unity 坐标系，直接对应
                Color32 srcColor = srcPx[y * origW + x];
                outPx[y * origW + x] = new Color32(srcColor.r, srcColor.g,
                                                    srcColor.b, alpha);
            }
        }

        tex.SetPixels32(outPx);
        tex.Apply();
        return tex;
    }
}