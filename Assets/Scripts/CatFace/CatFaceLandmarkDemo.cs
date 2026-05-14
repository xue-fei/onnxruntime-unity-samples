using System.Collections;
using UnityEngine;
using UnityEngine.UI;

[RequireComponent(typeof(CatFaceLandmarkDetector))]
public class CatFaceLandmarkDemo : MonoBehaviour
{
    // ── Inspector 配置 ────────────────────────────────────────────────
    [Header("输入图片")]
    [Tooltip("Texture2D")]
    public Texture2D _currentTexture;

    [Header("UI 显示")]
    [Tooltip("场景中用于显示猫脸图片的 RawImage（Canvas 下）")]
    public RawImage displayImage;

    [Tooltip("关键点标记 Prefab，建议用一个小圆 UI Image（Pivot 设为 0.5,0.5）")]
    public GameObject landmarkDotPrefab;

    [Tooltip("关键点标签字体大小")]
    public int labelFontSize = 12;

    // ── 运行时状态 ────────────────────────────────────────────────────
    private CatFaceLandmarkDetector _detector;
    private GameObject[] _dotObjects;   // 9 个关键点点对象
    private Text[] _labelTexts;   // 9 个标签文字

    // 关键点颜色
    private static readonly Color[] DOT_COLORS =
    {
        Color.cyan,    // 0 左眼
        Color.cyan,    // 1 右眼
        Color.yellow,  // 2 嘴
        Color.green,   // 3 左耳1
        Color.green,   // 4 左耳2
        Color.green,   // 5 左耳3
        Color.magenta, // 6 右耳1
        Color.magenta, // 7 右耳2
        Color.magenta, // 8 右耳3
    };

    private static readonly string[] LANDMARK_NAMES =
    {
        "左眼", "右眼", "嘴",
        "左耳1", "左耳2", "左耳3",
        "右耳1", "右耳2", "右耳3"
    };

    // ─────────────────────────────────────────────────────────────────
    // 生命周期
    // ─────────────────────────────────────────────────────────────────
    private void Awake()
    {
        displayImage.texture = _currentTexture;
        displayImage.SetNativeSize();
        _detector = GetComponent<CatFaceLandmarkDetector>();
        // 关闭 Detector 自带的 OnGUI 调试显示，由本脚本统一管理 UI
        _detector.showDebugGUI = false;
    }

    private void Start()
    {
        StartCoroutine(LoadAndInfer());
    }

    private void Update()
    {
        // 按空格重新推理
        if (Input.GetKeyDown(KeyCode.Space))
            StartCoroutine(LoadAndInfer());
    }

    // ─────────────────────────────────────────────────────────────────
    // 主流程：加载图片 → 推理 → 显示结果
    // ─────────────────────────────────────────────────────────────────
    private IEnumerator LoadAndInfer()
    {
        // ── 3. 显示原图到 UI RawImage ─────────────────────────────────
        if (displayImage != null)
        {
            displayImage.texture = _currentTexture;
            // 保持图片原始宽高比
            FitRawImageToTexture(displayImage, _currentTexture);
        }

        // 等一帧，确保 UI Layout 更新完毕后再计算坐标映射
        yield return null;

        // ── 4. 推理 ───────────────────────────────────────────────────
        Debug.Log($"[Demo] 开始推理，图片尺寸: {_currentTexture.width}×{_currentTexture.height}");
        Vector2[] landmarks = _detector.Detect(_currentTexture);

        if (landmarks == null || landmarks.Length != 9)
        {
            Debug.LogError("[Demo] 推理失败或输出异常");
            yield break;
        }

        // ── 5. 打印结果到 Console ─────────────────────────────────────
        Debug.Log("[Demo] 推理完成！关键点坐标（原图像素）：");
        for (int i = 0; i < 9; i++)
            Debug.Log($"  [{i}] {LANDMARK_NAMES[i]}: ({landmarks[i].x:F1}, {landmarks[i].y:F1})");

        // ── 6. 在 UI 上绘制关键点 ─────────────────────────────────────
        if (displayImage != null && landmarkDotPrefab != null)
            DrawLandmarksOnUI(landmarks, _currentTexture.width, _currentTexture.height);
    }

    // ─────────────────────────────────────────────────────────────────
    // 在 UI RawImage 上绘制 9 个关键点
    //
    //   坐标映射:
    //     原图坐标 (px, py) 范围 [0, origW] × [0, origH]
    //     → RawImage 的 AnchoredPosition
    //
    //   注意:
    //     RawImage 的坐标原点在中心（UI 默认），Y 轴向上
    //     原图坐标原点在左上角，Y 轴向下
    //     需要做两步转换：
    //       uiX =  (px / origW - 0.5) * rectW
    //       uiY = -(py / origH - 0.5) * rectH
    // ─────────────────────────────────────────────────────────────────
    private void DrawLandmarksOnUI(Vector2[] landmarks, int origW, int origH)
    {
        // 清除旧的点
        ClearDots();

        _dotObjects = new GameObject[9];
        _labelTexts = new Text[9];

        Rect imgRect = displayImage.rectTransform.rect;
        float rectW = imgRect.width;
        float rectH = imgRect.height;

        for (int i = 0; i < 9; i++)
        {
            // ── 坐标转换 ──────────────────────────────────────────────
            float normX = landmarks[i].x / origW;       // [0, 1]
            float normY = landmarks[i].y / origH;       // [0, 1]
            float uiX = (normX - 0.5f) * rectW;      // UI X（中心为0）
            float uiY = -(normY - 0.5f) * rectH;      // UI Y（中心为0，翻转Y）

            // ── 实例化关键点 Prefab ───────────────────────────────────
            GameObject dot = Instantiate(landmarkDotPrefab, displayImage.transform);
            RectTransform dotRT = dot.GetComponent<RectTransform>();
            if (dotRT != null)
            {
                dotRT.anchorMin = new Vector2(0.5f, 0.5f);
                dotRT.anchorMax = new Vector2(0.5f, 0.5f);
                dotRT.pivot = new Vector2(0.5f, 0.5f);
                dotRT.anchoredPosition = new Vector2(uiX, uiY);
                dotRT.sizeDelta = new Vector2(10f, 10f);
            }

            // 设置颜色
            Image dotImg = dot.GetComponent<Image>();
            if (dotImg != null)
                dotImg.color = DOT_COLORS[i];

            // ── 添加标签文字 ──────────────────────────────────────────
            GameObject labelObj = new GameObject($"Label_{LANDMARK_NAMES[i]}");
            labelObj.transform.SetParent(displayImage.transform, false);
            RectTransform labelRT = labelObj.AddComponent<RectTransform>();
            labelRT.anchorMin = new Vector2(0.5f, 0.5f);
            labelRT.anchorMax = new Vector2(0.5f, 0.5f);
            labelRT.pivot = new Vector2(0f, 0.5f);
            labelRT.anchoredPosition = new Vector2(uiX + 8f, uiY);
            labelRT.sizeDelta = new Vector2(50f, 24f);

            Text label = labelObj.AddComponent<Text>();
            label.text = LANDMARK_NAMES[i];
            label.fontSize = labelFontSize;
            label.color = DOT_COLORS[i];
            label.font = Resources.GetBuiltinResource<Font>("LegacyRuntime.ttf");
            label.alignment = TextAnchor.MiddleLeft;

            _dotObjects[i] = dot;
            _labelTexts[i] = label;
        }
    }

    // ─────────────────────────────────────────────────────────────────
    // 清除上一次的关键点 UI 对象
    // ─────────────────────────────────────────────────────────────────
    private void ClearDots()
    {
        if (_dotObjects != null)
        {
            foreach (var d in _dotObjects)
                if (d != null) Destroy(d);
            _dotObjects = null;
        }
        if (_labelTexts != null)
        {
            foreach (var t in _labelTexts)
                if (t != null) Destroy(t.gameObject);
            _labelTexts = null;
        }
    }

    // ─────────────────────────────────────────────────────────────────
    // 让 RawImage 保持图片宽高比填充
    // ─────────────────────────────────────────────────────────────────
    private void FitRawImageToTexture(RawImage img, Texture2D tex)
    {
        float texAspect = (float)tex.width / tex.height;
        RectTransform rt = img.rectTransform;
        float containerW = rt.rect.width > 0 ? rt.rect.width : Screen.width;
        float containerH = rt.rect.height > 0 ? rt.rect.height : Screen.height;
        float containerAsp = containerW / containerH;

        if (texAspect > containerAsp)
        {
            // 宽度撑满，高度等比收缩
            img.uvRect = new Rect(0, (1f - containerAsp / texAspect) / 2f,
                                  1f, containerAsp / texAspect);
        }
        else
        {
            // 高度撑满，宽度等比收缩
            img.uvRect = new Rect((1f - texAspect / containerAsp) / 2f, 0,
                                  texAspect / containerAsp, 1f);
        }
    }

    private void OnDestroy()
    {
        ClearDots();
    }
}