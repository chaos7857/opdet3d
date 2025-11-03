import cv2
from PIL import Image
import os

def video_to_gif(
    input_video_path: str,
    output_gif_path: str,
    start_second: int = 0,    # 起始时间（秒），默认从开头开始
    end_second: int = None,   # 结束时间（秒），默认到视频结尾
    target_fps: int = 10,     # GIF 帧率（建议 5-15，太高会增大体积）
    target_resolution: tuple = None  # 目标分辨率 (宽, 高)，如 (640, 360)，默认保持原视频比例
):
    # 1. 检查输入视频是否存在
    if not os.path.exists(input_video_path):
        raise FileNotFoundError(f"输入视频不存在：{input_video_path}")
    
    # 2. 打开视频文件
    cap = cv2.VideoCapture(input_video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件：{input_video_path}")
    
    # 3. 获取视频基础信息
    original_fps = cap.get(cv2.CAP_PROP_FPS)  # 原视频帧率
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 总帧数
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 原宽度
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # 原高度
    total_seconds = total_frames / original_fps  # 视频总时长（秒）
    
    # 4. 处理起始/结束时间（防止超出视频时长）
    if start_second < 0:
        start_second = 0
    if end_second is None or end_second > total_seconds:
        end_second = total_seconds
    if start_second >= end_second:
        raise ValueError("起始时间必须小于结束时间")
    
    # 5. 计算需要提取的帧范围（按原视频帧率换算）
    start_frame = int(start_second * original_fps)
    end_frame = int(end_second * original_fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # 跳转到起始帧
    
    # 6. 处理目标分辨率（若未指定，保持原比例；若指定，强制缩放）
    if target_resolution is None:
        # 默认按 GIF 常见比例缩小（如原 1080p→640x360）
        scale_ratio = 0.5  # 缩放比例，可调整
        target_width = int(original_width * scale_ratio)
        target_height = int(original_height * scale_ratio)
    else:
        target_width, target_height = target_resolution
    
    # 7. 逐帧提取并处理（按目标帧率采样，避免帧过多）
    frame_interval = int(original_fps / target_fps)  # 每 N 帧取 1 帧（匹配目标帧率）
    frames = []  # 存储待合成 GIF 的帧
    current_frame = start_frame
    
    while current_frame <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break  # 读取失败或到末尾，退出
        
        # ① 调整帧分辨率
        resized_frame = cv2.resize(frame, (target_width, target_height))
        # ② 转换颜色格式：OpenCV (BGR) → PIL (RGB)
        rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
        # ③ 转换为 PIL Image 对象并添加到列表
        pil_frame = Image.fromarray(rgb_frame)
        frames.append(pil_frame)
        
        # 按间隔跳帧，避免重复采样
        current_frame += frame_interval
    
    # 8. 生成 GIF（若有有效帧）
    if frames:
        # save_all：保存所有帧；duration：每帧停留时间（毫秒）= 1000/帧率
        # loop=0：无限循环；loop=1：循环 1 次（即播放 2 遍），以此类推
        frames[0].save(
            output_gif_path,
            save_all=True,
            append_images=frames[1:],
            duration=int(1000 / target_fps),
            loop=0,
            disposal=2  # 帧切换时清除上一帧（避免残影）
        )
        print(f"GIF 生成成功！路径：{output_gif_path}")
        print(f"GIF 信息：分辨率 {target_width}x{target_height} | 帧率 {target_fps} | 总帧数 {len(frames)}")
    else:
        raise ValueError("未提取到有效帧，无法生成 GIF（可能视频损坏或时间范围错误）")
    
    # 9. 释放资源
    cap.release()


# ------------------- 使用示例 -------------------
if __name__ == "__main__":
    # 输入视频路径（替换为你的视频路径，支持 mp4、avi 等常见格式）
    INPUT_VIDEO = "25455306.mp4"
    # 输出 GIF 路径
    OUTPUT_GIF = "output.gif"
    
    # 调用函数（根据需求调整参数）
    video_to_gif(
        input_video_path=INPUT_VIDEO,
        output_gif_path=OUTPUT_GIF,
        start_second=1,    # 从第 1 秒开始
        end_second=13,      # 到第 5 秒结束（仅转换 4 秒内容）
        target_fps=8,      # GIF 帧率 8（流畅且体积适中）
        target_resolution=(640, 360)  # 目标分辨率 640x360
    )