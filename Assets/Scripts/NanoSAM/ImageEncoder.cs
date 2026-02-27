using UnityEngine;

/// <summary>
/// 将Unity Texture2D编码为SAM所需的image embedding格式
/// 真实项目建议用image_encoder.onnx替代本脚本
/// </summary>
public static class ImageEncoder
{
    private const int SAM_SIZE = 1024;

    // ImageNet均值/标准差
    private static readonly float[] MEAN = { 0.485f, 0.456f, 0.406f };
    private static readonly float[] STD = { 0.229f, 0.224f, 0.225f };

    /// <summary>
    /// 将Texture2D预处理为SAM输入格式 [1,3,1024,1024] CHW，RGB
    /// </summary>
    public static float[] PreprocessImage(Texture2D tex)
    {
        // 缩放至1024x1024
        var rt = RenderTexture.GetTemporary(SAM_SIZE, SAM_SIZE, 0,
                     RenderTextureFormat.ARGB32);
        Graphics.Blit(tex, rt);

        var resized = new Texture2D(SAM_SIZE, SAM_SIZE, TextureFormat.RGB24, false);
        var prev = RenderTexture.active;
        RenderTexture.active = rt;
        resized.ReadPixels(new Rect(0, 0, SAM_SIZE, SAM_SIZE), 0, 0);
        resized.Apply();
        RenderTexture.active = prev;
        RenderTexture.ReleaseTemporary(rt);

        Color[] pixels = resized.GetPixels();
        float[] data = new float[3 * SAM_SIZE * SAM_SIZE];

        for (int y = 0; y < SAM_SIZE; y++)
        {
            for (int x = 0; x < SAM_SIZE; x++)
            {
                // Unity纹理Y轴翻转
                Color c = pixels[(SAM_SIZE - 1 - y) * SAM_SIZE + x];

                int idx = y * SAM_SIZE + x;
                data[0 * SAM_SIZE * SAM_SIZE + idx] = (c.r - MEAN[0]) / STD[0];
                data[1 * SAM_SIZE * SAM_SIZE + idx] = (c.g - MEAN[1]) / STD[1];
                data[2 * SAM_SIZE * SAM_SIZE + idx] = (c.b - MEAN[2]) / STD[2];
            }
        }

        UnityEngine.Object.Destroy(resized);
        return data;
    }

    /// <summary>
    /// 如果你有image_encoder.onnx，用此方法运行编码器得到embedding
    /// </summary>
    public static float[] RunEncoderOnnx(
        Microsoft.ML.OnnxRuntime.InferenceSession encoderSession,
        float[] preprocessedImage)
    {
        var tensor = new Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<float>(
            preprocessedImage, new[] { 1, 3, SAM_SIZE, SAM_SIZE });

        var inputs = new[]
        {
            Microsoft.ML.OnnxRuntime.NamedOnnxValue
                .CreateFromTensor("image", tensor)
        };

        using var results = encoderSession.Run(inputs);
        foreach (var r in results)
        {
            if (r.Name.Contains("embed") || r.Name.Contains("feature"))
            {
                return (r.Value as
                    Microsoft.ML.OnnxRuntime.Tensors.DenseTensor<float>)
                    .Buffer.ToArray();
            }
        }
        throw new System.Exception("未找到encoder输出");
    }
}