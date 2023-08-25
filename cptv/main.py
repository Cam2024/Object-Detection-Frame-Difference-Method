# import cv2
#
# def detect_motion(frame1, frame2, frame3):
#     # 转换为灰度图像
#     gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
#     gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
#     gray3 = cv2.cvtColor(frame3, cv2.COLOR_BGR2GRAY)
#
#     # 计算三帧差分图像
#     diff1 = cv2.absdiff(gray1, gray2)
#     diff2 = cv2.absdiff(gray2, gray3)
#
#     # 使用阈值进行图像二值化
#     _, thresh1 = cv2.threshold(diff1, 30, 255, cv2.THRESH_BINARY)
#     _, thresh2 = cv2.threshold(diff2, 30, 255, cv2.THRESH_BINARY)
#
#     # 使用逻辑与运算得到最终的差分图像
#     motion_diff = cv2.bitwise_and(thresh1, thresh2)
#
#     # 进行形态学操作，去除噪点
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
#     motion_diff = cv2.morphologyEx(motion_diff, cv2.MORPH_OPEN, kernel)
#
#     # 寻找轮廓
#     contours, _ = cv2.findContours(motion_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     # 绘制边界框
#     for contour in contours:
#         if cv2.contourArea(contour) > 500:
#             (x, y, w, h) = cv2.boundingRect(contour)
#             cv2.rectangle(frame2, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
#     # 返回带有边界框的图像
#     return frame2
#
# # 打开视频文件
# cap = cv2.VideoCapture('output.mp4')
#
# # 读取第一帧
# _, frame1 = cap.read()
#
# # 读取第二帧
# _, frame2 = cap.read()
#
# while True:
#     # 读取第三帧
#     _, frame3 = cap.read()
#
#     # 如果无法读取帧，则退出循环
#     if not _:
#         break
#
#     # 运动目标检测
#     motion_detected_frame = detect_motion(frame1, frame2, frame3)
#
#     # 显示结果
#     cv2.imshow("Motion Detection", motion_detected_frame)
#
#     # 更新帧
#     frame1 = frame2
#     frame2 = frame3
#
#     # 按下 q 键退出循环
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# # 释放资源
# cap.release()
# cv2.destroyAllWindows()





#
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








# #相邻帧间差分法
# # 导入必要的软件包
# import cv2
#
# # 视频文件输入初始化
# camera = cv2.VideoCapture('a.mp4')
#
# # 视频文件输出参数设置
# out_fps = 12.0  # 输出文件的帧率
# fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')
# out1 = cv2.VideoWriter('v1.avi', fourcc, out_fps, (500, 400))
# out2 = cv2.VideoWriter('v2.avi', fourcc, out_fps, (500, 400))
#
# # 初始化当前帧的前帧
# lastFrame = None
#
# # 遍历视频的每一帧
# while camera.isOpened():
#
#     # 读取下一帧
#     (ret, frame) = camera.read()
#
#     # 如果不能抓取到一帧，说明我们到了视频的结尾
#     if not ret:
#         break
#
#         # 调整该帧的大小
#     frame = cv2.resize(frame, (500, 400), interpolation=cv2.INTER_CUBIC)
#
#     # 如果第一帧是None，对其进行初始化
#     if lastFrame is None:
#         lastFrame = frame
#         continue
#
#         # 计算当前帧和前帧的不同
#     frameDelta = cv2.absdiff(lastFrame, frame)
#
#     # 当前帧设置为下一帧的前帧
#     lastFrame = frame.copy()
#
#     # 结果转为灰度图
#     thresh = cv2.cvtColor(frameDelta, cv2.COLOR_BGR2GRAY)
#
#     # 图像二值化
#     thresh = cv2.threshold(thresh, 25, 255, cv2.THRESH_BINARY)[1]
#
#     '''
#     #去除图像噪声,先腐蚀再膨胀(形态学开运算)
#     thresh=cv2.erode(thresh,None,iterations=1)
#     thresh = cv2.dilate(thresh, None, iterations=2)
#     '''
#
#     # 阀值图像上的轮廓位置
#     cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     # 遍历轮廓
#     for c in cnts:
#         # 忽略小轮廓，排除误差
#         if cv2.contourArea(c) < 300:
#             continue
#
#             # 计算轮廓的边界框，在当前帧中画出该框
#         (x, y, w, h) = cv2.boundingRect(c)
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
#         # 显示当前帧
#     cv2.imshow("frame", frame)
#     cv2.imshow("frameDelta", frameDelta)
#     cv2.imshow("thresh", thresh)
#
#     # 保存视频
#     out1.write(frame)
#     out2.write(frameDelta)
#
#     # 如果q键被按下，跳出循环
#     if cv2.waitKey(200) & 0xFF == ord('q'):
#         break
#
#     # 清理资源并关闭打开的窗口
# out1.release()
# out2.release()
# camera.release()
# cv2.destroyAllWindows()







#
# #三帧差分法
# # 导入必要的软件包
# import cv2
#
# # 视频文件输入初始化
# camera = cv2.VideoCapture('output.mp4')
#
# # 视频文件输出参数设置
# out_fps = 12.0  # 输出文件的帧率
# fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', '2')
# out1 = cv2.VideoWriter('v3.avi', fourcc, out_fps, (500, 400))
# out2 = cv2.VideoWriter('v4.avi', fourcc, out_fps, (500, 400))
#
# # 初始化当前帧的前两帧
# lastFrame1 = None
# lastFrame2 = None
#
# # 遍历视频的每一帧
# while camera.isOpened():
#
#     # 读取下一帧
#     (ret, frame) = camera.read()
#
#     # 如果不能抓取到一帧，说明我们到了视频的结尾
#     if not ret:
#         break
#
#     # 调整该帧的大小
#     frame = cv2.resize(frame, (500, 400), interpolation=cv2.INTER_CUBIC)
#
#     # 如果第一二帧是None，对其进行初始化,计算第一二帧的不同
#     if lastFrame2 is None:
#         if lastFrame1 is None:
#             lastFrame1 = frame
#         else:
#             lastFrame2 = frame
#             global frameDelta1  # 全局变量
#             frameDelta1 = cv2.absdiff(lastFrame1, lastFrame2)  # 帧差一
#         continue
#
#     # 计算当前帧和前帧的不同,计算三帧差分
#     frameDelta2 = cv2.absdiff(lastFrame2, frame)  # 帧差二
#     thresh = cv2.bitwise_and(frameDelta1, frameDelta2)  # 图像与运算
#     thresh2 = thresh.copy()
#
#     # 当前帧设为下一帧的前帧,前帧设为下一帧的前前帧,帧差二设为帧差一
#     lastFrame1 = lastFrame2
#     lastFrame2 = frame.copy()
#     frameDelta1 = frameDelta2
#
#     # 结果转为灰度图
#     thresh = cv2.cvtColor(thresh, cv2.COLOR_BGR2GRAY)
#
#     # 图像二值化
#     thresh = cv2.threshold(thresh, 25, 255, cv2.THRESH_BINARY)[1]
#
#     # 去除图像噪声,先腐蚀再膨胀(形态学开运算)
#     thresh = cv2.dilate(thresh, None, iterations=3)
#     thresh = cv2.erode(thresh, None, iterations=1)
#
#     # 阀值图像上的轮廓位置
#     cnts, hierarchy = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#
#     # 遍历轮廓
#     for c in cnts:
#         # 忽略小轮廓，排除误差
#         if cv2.contourArea(c) < 300:
#             continue
#
#         # 计算轮廓的边界框，在当前帧中画出该框
#         (x, y, w, h) = cv2.boundingRect(c)
#         cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
#
#     # 显示当前帧
#     cv2.imshow("frame", frame)
#     cv2.imshow("thresh", thresh)
#     cv2.imshow("threst2", thresh2)
#
#     # 保存视频
#     out1.write(frame)
#     out2.write(thresh2)
#
#     # 如果q键被按下，跳出循环
#     if cv2.waitKey(200) & 0xFF == ord('q'):
#         break
#
# # 清理资源并关闭打开的窗口
# out1.release()
# out2.release()
# camera.release()
# cv2.destroyAllWindows()









# #帧差法精准版
# import cv2
# import numpy as np
#
# camera = cv2.VideoCapture('output.mp4')  # 参数0表示第一个摄像头
# # 判断视频是否打开
# if (camera.isOpened()):
#     print('摄像头成功打开')
# else:
#     print('摄像头未打开')
#
# # 测试用,查看视频size
# size = (int(camera.get(cv2.CAP_PROP_FRAME_WIDTH)),
#         int(camera.get(cv2.CAP_PROP_FRAME_HEIGHT)))
# print('size:' + repr(size))
#
# es = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 4))
# background = None
#
# while True:
#     # 读取视频流
#     grabbed, frame_lwpCV = camera.read()
#
#     # 对帧进行预处理，先转灰度图，再进行高斯滤波。
#     # 用高斯滤波进行模糊处理，进行处理的原因：每个输入的视频都会因自然震动、光照变化或者摄像头本身等原因而产生噪声。对噪声进行平滑是为了避免在运动和跟踪时将其检测出来。
#     gray_lwpCV = cv2.cvtColor(frame_lwpCV, cv2.COLOR_BGR2GRAY)
#     gray_lwpCV = cv2.GaussianBlur(gray_lwpCV, (21, 21), 0)
#
#     # 将第一帧设置为整个输入的背景
#     if background is None:
#         background = gray_lwpCV
#         continue
#     # 对于每个从背景之后读取的帧都会计算其与背景之间的差异，并得到一个差分图（different map）。
#     # 还需要应用阈值来得到一幅黑白图像，并通过下面代码来膨胀（dilate）图像，从而对孔（hole）和缺陷（imperfection）进行归一化处理
#     diff = cv2.absdiff(background, gray_lwpCV)
#     diff = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)[1]  # 二值化阈值处理
#     diff = cv2.dilate(diff, es, iterations=2)  # 形态学膨胀
#
#     # 显示矩形框
#     contours, hierarchy = cv2.findContours(diff.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 该函数计算一幅图像中目标的轮廓
#     for c in contours:
#         if cv2.contourArea(c) < 1500:  # 对于矩形区域，只显示大于给定阈值的轮廓，所以一些微小的变化不会显示。对于光照不变和噪声低的摄像头可不设定轮廓最小尺寸的阈值
#             continue
#         (x, y, w, h) = cv2.boundingRect(c)  # 该函数计算矩形的边界框
#         cv2.rectangle(frame_lwpCV, (x, y), (x + w, y + h), (0, 255, 0), 2)
#     frame_lwpCV = cv2.resize(frame_lwpCV, (500, 400), interpolation=cv2.INTER_CUBIC)
#     diff = cv2.resize(diff, (500, 400), interpolation=cv2.INTER_CUBIC)
#     cv2.imshow('contours', frame_lwpCV)
#     cv2.imshow('dis', diff)
#
#     key = cv2.waitKey(1) & 0xFF
#     # 按'q'健退出循环
#     if key == ord('q'):
#         break
# # When everything done, release the capture
# camera.release()
# cv2.destroyAllWindows()










#
# import cv2
# import numpy as np
#
# # 创建视频捕捉对象
# cap = cv2.VideoCapture('output.mp4')
#
# # 读取第一帧
# ret, old_frame = cap.read()
# old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
#
# # 设置角点检测参数
# feature_params = dict(maxCorners=100, qualityLevel=0.3, minDistance=7, blockSize=7)
#
# # 设置Lucas-Kanade光流参数
# lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
#
# # 选择追踪的角点
# p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
#
# # 创建一个掩膜用于绘制轨迹
# mask = np.zeros_like(old_frame)
#
# while True:
#     # 读取当前帧
#     ret, frame = cap.read()
#     # frame = cv2.resize(frame, (500, 400), interpolation=cv2.INTER_CUBIC)
#     if not ret:
#         break
#     frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     # 计算光流
#     p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
#
#     # 选择好的追踪点
#     good_new = p1[st == 1]
#     good_old = p0[st == 1]
#
#     # 绘制轨迹线
#     for i, (new, old) in enumerate(zip(good_new, good_old)):
#         a, b = new.ravel()
#         c, d = old.ravel()
#         mask = cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
#         frame = cv2.circle(frame, (int(a), int(b)), 5, (0, 0, 255), -1)
#     img = cv2.add(frame, mask)
#
#     # 显示结果
#     cv2.imshow('Optical Flow', img)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
#     # 更新追踪点和当前帧
#     old_gray = frame_gray.copy()
#     p0 = good_new.reshape(-1, 1, 2)
#
# # 释放资源
# cap.release()
# cv2.destroyAllWindows()
# #
