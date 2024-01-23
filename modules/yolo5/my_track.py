import os
import torch
import torch.backends.cudnn as cudnn
import cv2
import matplotlib.pyplot as plt
from modules.yolo5.models.experimental import attempt_load
from modules.yolo5.utils.torch_utils import select_device, load_classifier, time_sync
from modules.yolo5.utils.datasets import LoadStreams, LoadImages
from modules.yolo5.utils.general import check_img_size, check_requirements, check_imshow, colorstr, is_ascii, non_max_suppression, \
    apply_classifier, scale_coords, strip_optimizer, set_logging, increment_path, save_one_box
import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


def load_video(video_path):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 检查是否成功打开视频文件
    if not cap.isOpened():
        print("Error: Couldn't open the video file.")
        return None

        # 获取视频的帧率、宽度和高度
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 初始化帧计数器和帧列表
    frame_count = 0
    frames = []

    # 逐帧读取视频并保存为图片
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:  # 如果无法获取帧，跳出循环
            break
        frame_count += 1
        img_name = f"frame_{frame_count}.jpg"  # 保存图片的文件名格式
        cv2.imwrite("F:\\3D-ZeF_Data\\testImages\\"+img_name, frame)  # 保存当前帧为图片
        frames.append(img_name)  # 将图片文件名添加到帧列表中

    # 释放视频文件句柄
    cap.release()
    return frames, fps, width, height

def create_dataloader(folder_path, batch_size, shuffle=False):
    # 数据预处理：将图片转换为PyTorch的张量，并归一化到[-1,1]范围（通常是用于神经网络的输入）
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((640, 640))])
    # 使用ImageFolder加载数据，它会自动根据文件夹名称排序并创建数据集对象
    dataset = datasets.ImageFolder(folder_path, transform=transform)
    # 创建DataLoader对象，指定批次大小和是否混洗数据
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader


if __name__ == '__main__':
    weights = "F:/3D-ZeF_Data/data/best.pt"
    device = '0'
    device = select_device(device)

    video_path = "F:\\3D-ZeF_Data\\mFishvideo.mp4"
    # 加载模型
    model = attempt_load(weights, map_location=device)
    print(load_video(video_path))
    dataset = create_dataloader("F:\\3D-ZeF_Data\\imgs", 1)

    for img in dataset:
        img = img[0].to(device)
        img = img / 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim
        print(img.shape)
        pred = model(img, augment=False, visualize=False)[0]
        # NMS
        pred = non_max_suppression(pred, conf_thres=0.6, iou_thres=0.5, classes=None, agnostic=True, max_det=2)
        # 遍历每个预测结果
        for det in pred:
            if det is not None and len(det):
                # 获取边界框坐标
                det = det.cpu().numpy()
                for box in det:
                    # 提取边界框坐标和类别
                    x1, y1, x2, y2, conf, cls = box
                    # 画矩形框
                    cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    # 显示类别和置信度
                    label = f"{classes[int(cls)]}: {conf:.2f}"
                    cv2.putText(frame, label, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # 保存带有预测结果的图片
        # cv2.imwrite("path/to/save/result_image.jpg", frame)

