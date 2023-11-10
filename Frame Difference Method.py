import cv2
import numpy as np

def detect_drones(frame1, frame2):
    # 转换为灰度图像
    gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    # 计算光流
    flow = cv2.calcOpticalFlowFarneback(gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    # 计算光流的大小和角度
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # 将角度转换为颜色编码
    hue = angle * 180 / np.pi / 2

    # 创建HSV图像并设置饱和度和值为最大值
    hsv = np.zeros_like(frame1)
    hsv[..., 1] = 255
    hsv[..., 2] = 255

    # 将颜色编码转换为RGB图像
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    # 二值化处理，根据光流大小阈值进行筛选
    mask = cv2.threshold(magnitude, 30, 255, cv2.THRESH_BINARY)[1]

    # 对筛选后的光流进行形态学操作，去除噪点
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 在原始帧上绘制无人机区域
    frame1[mask != 0] = [0, 0, 255]

    return frame1

# 打开视频文件
cap = cv2.VideoCapture('output.mp4')

# 读取第一帧
_, frame1 = cap.read()

while True:
    # 读取第二帧
    _, frame2 = cap.read()

    # 如果无法读取帧，则退出循环
    if not _:
        break

    # 对无人机进行检测
    frame_with_drones = detect_drones(frame1, frame2)

    # 显示结果
    cv2.imshow("Drone Detection", frame_with_drones)

    # 更新帧
    frame1 = frame2

    # 按下 q 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放资源
cap.release()
cv2.destroyAllWindows()
