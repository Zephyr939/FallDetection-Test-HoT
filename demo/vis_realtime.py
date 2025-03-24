import os 
import sys
import cv2
import glob
import copy
import json
import torch
import argparse
import numpy as np
import time
#import datetime
import queue
import threading
from collections import deque
#from tqdm import tqdm
#from lib.preprocess import h36m_coco_format, revise_kpts
#from lib.hrnet.gen_kpts import gen_video_kpts as hrnet_pose
import warnings
import matplotlib
import matplotlib.pyplot as plt 
#from mpl_toolkits.mplot3d import Axes3D
#import matplotlib.gridspec as gridspec

plt.switch_backend('agg')
warnings.filterwarnings('ignore')
matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
if project_root not in sys.path:
    sys.path.append(project_root)

sys.path.append(os.getcwd())
from common.utils import *
from common.camera import *
from model.mixste.hot_mixste import Model

# 导入跌倒检测模型
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(__file__)), 'train'))
from train_lstm import FallDetectionLSTM

class FallDetector:
    """跌倒检测器类"""
    def __init__(self, model_dir, device, seq_length=30):
        self.seq_length = seq_length
        self.device = device
        
        # 加载模型配置
        summary_file = os.path.join(model_dir, 'training_summary.json')
        with open(summary_file, 'r', encoding='utf-8') as f:
            config = json.load(f)
        
        # 提取模型参数
        self.model_params = config['parameters']['model_params']
        self.norm_params = config['parameters']['normalization_params']
        self.mean = np.array(self.norm_params['mean'])
        self.std = np.array(self.norm_params['std'])
        
        # 创建模型实例
        self.model = FallDetectionLSTM(
            input_dim=51,  # 17个关键点 * 3个坐标
            hidden_dim=self.model_params['hidden_dim'],
            num_layers=self.model_params['num_layers'],
            dropout=self.model_params['dropout']
        ).to(device)
        
        # 加载模型权重
        model_path = os.path.join(model_dir, 'best_model.pth')
        checkpoint = torch.load(model_path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # 初始化序列缓冲区
        self.pose_buffer = deque(maxlen=seq_length)
        self.last_prediction_time = None
        self.last_prediction = None
        
    def add_pose(self, pose_3d):
        """添加新的3D姿态到缓冲区"""
        if pose_3d is not None:
            self.pose_buffer.append(pose_3d)
    
    def get_prediction(self):
        """获取当前序列的预测结果"""
        if len(self.pose_buffer) < self.seq_length:
            return None, None
            
        # 准备序列数据
        sequence = np.array(list(self.pose_buffer))
        flattened_sequence = sequence.reshape(self.seq_length, -1)
        
        # 标准化数据
        normalized_sequence = (flattened_sequence - self.mean) / self.std
        
        # 预测
        with torch.no_grad():
            sequence_tensor = torch.FloatTensor(normalized_sequence).unsqueeze(0).to(self.device)
            prediction = self.model(sequence_tensor)
            prediction = prediction.cpu().numpy()[0][0]
            
        current_time = time.time()
        self.last_prediction_time = current_time
        self.last_prediction = prediction
        
        # 返回预测结果和延迟时间
        delay = (current_time - self.last_prediction_time) if self.last_prediction_time else 0
        return prediction, delay

class Buffer:
    """缓冲区类，用于在不同处理阶段之间传递数据"""
    def __init__(self, maxsize=300):
        self.queue = queue.Queue(maxsize)
        self.lock = threading.Lock()
    
    def put(self, item, block=True, timeout=None):
        with self.lock:
            return self.queue.put(item, block=block, timeout=timeout)
    
    def get(self, block=True, timeout=None):
        with self.lock:
            return self.queue.get(block=block, timeout=timeout)
    
    def empty(self):
        with self.lock:
            return self.queue.empty()
    
    def full(self):
        with self.lock:
            return self.queue.full()
    
    def qsize(self):
        with self.lock:
            return self.queue.qsize()

human_tracking = {
    "current_id": None,      # 当前跟踪的人体ID
    "frames_without_human": 0,  # 连续没有检测到人体的帧数
    "reset_required": False,  # 是否需要重置数据流
    "status_message": "System initializing"  # 状态信息
}

def update_status(message):
    """
    更新状态消息，确保使用ASCII字符
    """
    human_tracking["status_message"] = message

def load_models(args):
    """
    加载所有需要的模型
    """
    print("Loading models...")
    
    # 加载人体检测器
    if args.detector == 'yolo11':
        from lib.yolo11.human_detector import load_model as yolo_model
        from lib.yolo11.human_detector import reset_target
        from lib.yolo11.human_detector import get_default_args as get_yolo_args
        
        # 使用默认参数加载YOLO模型
        yolo_args = get_yolo_args()
        reset_target()
        detector = yolo_model(yolo_args, inp_dim=832)
        print("Loaded YOLO model")
    else:  # 默认使用YOLOv3
        from lib.yolov3.human_detector import load_model as yolo_model
        detector = yolo_model(inp_dim=832)  # 使用与原代码相同的参数
        print("Loaded YOLOv3 model")
    
    # 加载姿态估计模型 (HRNet)
    from lib.hrnet.lib.config import cfg, update_config
    from lib.hrnet.lib.models.pose_hrnet import PoseHighResolutionNet
    from lib.hrnet.gen_kpts import parse_args
    
    # 使用与原代码相同的配置
    hrnet_cfg = 'demo/lib/hrnet/experiments/w48_512x384_adam_lr1e-3.yaml'
    
    hrnet_args = parse_args()
    hrnet_args.cfg = hrnet_cfg
    update_config(cfg, hrnet_args)
    
    pose_model = PoseHighResolutionNet(cfg)
    pose_model.load_state_dict(torch.load(cfg.OUTPUT_DIR))
    pose_model = pose_model.cuda()
    pose_model.eval()
    print("Loaded HRNet model")
    
    # 加载3D姿态估计模型 (MixSTE)
    mixste_args = copy.deepcopy(args)
    mixste_args.layers, mixste_args.channel, mixste_args.d_hid, mixste_args.frames = 8, 512, 1024, 243
    mixste_args.token_num, mixste_args.layer_index = 81, 3
    mixste_args.pad = (mixste_args.frames - 1) // 2
    mixste_args.previous_dir = 'checkpoint/pretrained/hot_mixste'
    mixste_args.n_joints, mixste_args.out_joints = 17, 17
    
    mixste_model = Model(mixste_args).cuda()
    model_dict = mixste_model.state_dict()
    model_path = sorted(glob.glob(os.path.join(mixste_args.previous_dir, '*.pth')))[0]
    pre_dict = torch.load(model_path)
    state_dict = {k: v for k, v in pre_dict.items() if k in model_dict.keys()}
    model_dict.update(state_dict)
    mixste_model.load_state_dict(model_dict)
    mixste_model.eval()
    print("Loaded MixSTE model")
    
    print("Loaded all models")
    return detector, pose_model, mixste_model

def detect_human(frame, detector, args):
    """
    使用YOLO检测人体
    """
    try:
        if args.detector == 'yolo11':
            from lib.yolo11.human_detector import yolo_human_det as yolo_det
            bboxes, status = yolo_det(frame, detector, args.detector)
            if status is not None:
                # 更新状态消息
                update_status(status)
            return bboxes
        else:
            from lib.yolov3.human_detector import yolo_human_det as yolo_det
            bboxes = yolo_det(frame, detector)
            return bboxes
    except Exception as e:
        update_status("Detection error")
        return None

def extract_pose2D(frame, bboxes, pose_model):
    """
    使用HRNet提取2D姿态
    """
    from lib.hrnet.lib.utils.inference import get_max_preds
    from lib.hrnet.lib.config import cfg
    
    # 初始化返回值
    keypoints = np.zeros((1, 17, 2))
    scores = np.zeros((1, 17))
    
    # 检查边界框是否为None或空列表
    if bboxes is None or len(bboxes) == 0:
        return None, None
    
    try:
        # 预处理
        with torch.no_grad():
            inputs = torch.zeros(1, 3, 384, 288).cuda()
            valid_bbox_found = False
            
            for i, bbox in enumerate(bboxes):
                if i >= 1:  # 只处理第一个检测到的人
                    break
                    
                # 确保边界框坐标是有效的
                if bbox is None or len(bbox) < 4:
                    continue
                    
                # 确保边界框坐标是整数并且在图像范围内
                x1 = max(0, int(bbox[0]))
                y1 = max(0, int(bbox[1]))
                x2 = min(frame.shape[1], int(bbox[2]))
                y2 = min(frame.shape[0], int(bbox[3]))
                
                # 确保边界框有足够的大小
                if x2 <= x1 or y2 <= y1 or (x2 - x1) < 10 or (y2 - y1) < 10:
                    continue
                    
                # 裁剪并调整大小
                person_img = frame[y1:y2, x1:x2]
                if person_img.size == 0:
                    continue
                
                valid_bbox_found = True
                    
                person_img = cv2.resize(person_img, (288, 384))
                person_img = person_img[:, :, ::-1]  # BGR to RGB
                
                person_img = person_img / 255.
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                person_img = (person_img - mean) / std
                person_img = person_img.transpose((2, 0, 1))
                
                inputs[0] = torch.from_numpy(person_img).float().cuda()
                
                # 前向传播
                output = pose_model(inputs)
                
                # 获取关键点
                preds, _ = get_max_preds(output.detach().cpu().numpy())
                
                # 缩放回原始大小
                preds[0, :, 0] = preds[0, :, 0] * (bbox[2] - bbox[0]) / 288 + bbox[0]
                preds[0, :, 1] = preds[0, :, 1] * (bbox[3] - bbox[1]) / 384 + bbox[1]
                
                # 计算置信度
                heatmaps = output.detach().cpu().numpy()
                scores = np.zeros((1, preds.shape[1]))
                for n in range(preds.shape[0]):
                    for j in range(preds.shape[1]):
                        hm = heatmaps[n][j]
                        scores[n, j] = np.max(hm)
                
                return preds[0], scores[0]
            
            # 如果没有找到有效的边界框
            if not valid_bbox_found:
                return None, None
    
    except Exception as e:
        return None, None
    
    return None, None

def predict_pose3D(keypoints_2d_buffer, mixste_model, args, img_size):
    """
    使用MixSTE预测3D姿态
    """
    # 检查输入数据是否有效
    if keypoints_2d_buffer is None or len(keypoints_2d_buffer) == 0:
        return None
    
    # 确保输入数据是numpy数组
    if not isinstance(keypoints_2d_buffer, np.ndarray):
        keypoints_2d_buffer = np.array(keypoints_2d_buffer)
        
    # 检查关键点数据的形状
    if not all(kp.shape == (17, 2) for kp in keypoints_2d_buffer):
        return None
    
    # 转换为输入格式
    with torch.no_grad():
        frames = len(keypoints_2d_buffer)
        input_2D_no = keypoints_2d_buffer
        
        # 规范化坐标
        input_2D = normalize_screen_coordinates(input_2D_no, w=img_size[1], h=img_size[0])  
        
        # 数据增强（左右翻转）
        joints_left =  [4, 5, 6, 11, 12, 13]
        joints_right = [1, 2, 3, 14, 15, 16]
        
        try:
            input_2D_aug = copy.deepcopy(input_2D)
            input_2D_aug[:, :, 0] *= -1
            input_2D_aug[:, joints_left + joints_right] = input_2D_aug[:, joints_right + joints_left]
            input_2D = np.concatenate((np.expand_dims(input_2D, axis=0), np.expand_dims(input_2D_aug, axis=0)), 0)
            
            input_2D = input_2D[np.newaxis, :, :, :, :]
            input_2D = torch.from_numpy(input_2D.astype('float32')).cuda()
            
            # 前向传播
            predicted_3d_pos = mixste_model(input_2D)
            
            # 取出结果
            predicted_3d_pos = predicted_3d_pos[0, 0].cpu().numpy()
            
            # 返回3D姿态
            return predicted_3d_pos
        except Exception as e:
            return None

def process_args():
    """
    处理命令行参数
    """
    parser = argparse.ArgumentParser(description='实时3D姿态估计和跌倒检测')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
    parser.add_argument('--camera', type=int, default=0, help='摄像头ID')
    parser.add_argument('--detector', type=str, default='yolo11', choices=['yolov3', 'yolo11'], help='选择人体检测器')
    parser.add_argument('--output_file', type=str, default='realtime_3d_pose.npy', help='3D姿态输出文件名')
    parser.add_argument('--model_dir', type=str, default='checkpoint/fall_detection_lstm/2025-03-22-2328-b', help='跌倒检测模型目录,包含best_model.pth和training_summary.json')
    parser.add_argument('--seq_length', type=int, default=30, help='用于跌倒检测的序列长度')
    
    # 解析已知参数,忽略未知参数(这些参数可能是HRNet的)
    args, unknown = parser.parse_known_args()
    return args

def main():
    args = process_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    detector, pose_model, mixste_model = load_models(args)
    
    # 初始化跌倒检测器
    fall_detector = FallDetector(args.model_dir, device, args.seq_length)
    
    # 创建缓冲区
    yolo_buffer = Buffer(maxsize=15)  # 约0.5s @ 30fps
    hrnet_buffer = Buffer(maxsize=30)  # 约1s @ 30fps
    mixste_buffer = deque(maxlen=243)  # MixSTE的输入
    
    # 存储3D关键点和跌倒检测结果
    poses_3d = []
    fall_status = {
        "is_falling": False,
        "detection_delay": 0,
        "last_update_time": None,
        "has_human": False,
        "status_text": "Waiting for human"
    }
    
    # 打开摄像头
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        print("❌ 无法打开摄像头")
        return
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    img_size = (height, width, 3)
    
    # 帧率统计
    fps_time = time.time()
    fps_count = 0
    fps_display = 0
    
    # 姿态提取线程
    def pose_extraction_thread():
        while True:
            if yolo_buffer.empty():
                time.sleep(0.01)
                continue
                
            frame = yolo_buffer.get()
            bboxes = detect_human(frame, detector, args)
            
            # 确保bboxes不为None
            if bboxes is None:
                bboxes = []
            
            # 过滤掉无效的边界框
            valid_bboxes = []
            for bbox in bboxes:
                if bbox is not None and len(bbox) == 4:
                    # 确保边界框坐标有效
                    if bbox[2] > bbox[0] and bbox[3] > bbox[1]:
                        valid_bboxes.append(bbox)
            
            # 处理人体跟踪状态
            if len(valid_bboxes) > 0:
                # 找到最大的边界框（假设是主要目标）
                max_area = 0
                max_bbox = None
                for bbox in valid_bboxes:
                    area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
                    if area > max_area:
                        max_area = area
                        max_bbox = bbox
                
                # 计算边界框中心点作为简单的ID标识
                if max_bbox is not None:
                    center_x = (max_bbox[0] + max_bbox[2]) / 2
                    center_y = (max_bbox[1] + max_bbox[3]) / 2
                    current_id = f"{int(center_x)}_{int(center_y)}"
                    
                    # 检查ID是否变化
                    if human_tracking["current_id"] is None:
                        # 首次检测到人体
                        human_tracking["current_id"] = current_id
                        human_tracking["frames_without_human"] = 0
                        human_tracking["reset_required"] = True
                        update_status("检测到新的人体目标")
                    elif human_tracking["current_id"] != current_id:
                        # ID变化，需要重置
                        human_tracking["current_id"] = current_id
                        human_tracking["frames_without_human"] = 0
                        human_tracking["reset_required"] = True
                        update_status("人体ID变化")
                    else:
                        # 同一个人，继续跟踪
                        human_tracking["frames_without_human"] = 0
                        update_status("跟踪目标中")
                
                # 提取姿态
                keypoints, scores = extract_pose2D(frame, [max_bbox], pose_model)
                if keypoints is not None:
                    hrnet_buffer.put((frame, keypoints, scores, human_tracking["reset_required"]))
                    # 重置标志已处理
                    human_tracking["reset_required"] = False
            else:
                # 没有检测到人
                human_tracking["frames_without_human"] += 1
                if human_tracking["frames_without_human"] > 30:  # 1秒没有检测到人
                    if human_tracking["current_id"] is not None:
                        human_tracking["current_id"] = None
                        human_tracking["reset_required"] = True
                        update_status("未检测到人体目标")
                        # 重置跌倒检测状态
                        fall_status["has_human"] = False
                        fall_status["is_falling"] = False
                        fall_status["last_update_time"] = None
                        fall_status["status_text"] = "No human detected"
                
                # 放回一个空结果
                hrnet_buffer.put((frame, None, None, human_tracking["reset_required"]))
    
    # 3D姿态预测线程
    def pose3d_prediction_thread():
        nonlocal fps_count, fps_display
        
        while True:
            if hrnet_buffer.empty():
                time.sleep(0.01)
                continue
                
            frame, keypoints, scores, reset_required = hrnet_buffer.get()
            
            # 如果需要重置，清空MixSTE缓冲区
            if reset_required and len(mixste_buffer) > 0:
                update_status("重置姿态数据流")
                mixste_buffer.clear()
            
            if keypoints is not None:
                # 添加到MixSTE缓冲区
                mixste_buffer.append(keypoints)
                
                # 当收集到至少30帧时，进行一次3D预测
                if len(mixste_buffer) >= 30:
                    # 准备输入数据
                    if len(mixste_buffer) < 243:
                        # 不足243帧，复制最后一帧填充
                        buffer_list = list(mixste_buffer)
                        while len(buffer_list) < 243:
                            buffer_list.append(buffer_list[-1])
                        input_keypoints = buffer_list
                    else:
                        # 已有243帧，直接使用
                        input_keypoints = list(mixste_buffer)
                    
                    # 预测3D姿态
                    pose3d = predict_pose3D(input_keypoints, mixste_model, args, img_size)
                    
                    # 保存结果并进行跌倒检测
                    if pose3d is not None:
                        poses_3d.append(pose3d)
                        
                        # 添加到跌倒检测器并获取预测结果
                        fall_detector.add_pose(pose3d)
                        fall_prob, delay = fall_detector.get_prediction()
                        
                        # 更新跌倒状态
                        if fall_prob is not None:
                            fall_status["is_falling"] = fall_prob > 0.5
                            fall_status["detection_delay"] = delay
                            fall_status["last_update_time"] = time.time()
                            fall_status["has_human"] = True
                            
                            # 更新状态消息
                            if fall_status["is_falling"]:
                                fall_status["status_text"] = "Fall detected!"
                            else:
                                fall_status["status_text"] = "Normal activity"
                        
                        fps_count += 1
            
            # 计算FPS
            if time.time() - fps_time > 1.0:  # 每秒更新一次
                fps_display = fps_count
                fps_count = 0
    
    # 启动线程
    pose_thread = threading.Thread(target=pose_extraction_thread)
    pose_thread.daemon = True
    pose_thread.start()
    
    pose3d_thread = threading.Thread(target=pose3d_prediction_thread)
    pose3d_thread.daemon = True
    pose3d_thread.start()
    
    print("✓ 系统已启动，按 'q' 退出")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ 无法获取摄像头帧")
                break
            
            # 将帧放入YOLO缓冲区
            if not yolo_buffer.full():
                yolo_buffer.put(frame)
            
            # 显示当前画面
            current_time = time.time()
            if current_time - fps_time > 1.0:
                fps_time = current_time
            
            # 显示系统状态信息
            cv2.putText(frame, f"FPS: {fps_display}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Poses: {len(poses_3d)}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # 显示跌倒检测结果
            if fall_status["has_human"]:
                # 有人体时显示跌倒状态和延迟
                status_color = (0, 0, 255) if fall_status["is_falling"] else (0, 255, 0)
                cv2.putText(frame, f"Fall: {fall_status['is_falling']}", (10, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)
                if fall_status["last_update_time"] is not None:
                    delay = time.time() - fall_status["last_update_time"]
                    cv2.putText(frame, f"Delay: {delay:.1f}s", (10, 150),
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            else:
                # 无人体时显示等待状态
                cv2.putText(frame, "Waiting for human...", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
            # 显示系统状态
            cv2.putText(frame, f"Status: {fall_status['status_text']}", (10, 190),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow('Realtime 3D Pose Estimation', frame)
            
            # 检查退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    
    except KeyboardInterrupt:
        print("用户中断运行")
    
    finally:
        # 保存3D姿态数据
        if poses_3d:
            poses_3d_array = np.array(poses_3d)
            output_file = args.output_file
            np.save(output_file, poses_3d_array)
            print(f"✓ 已保存3D姿态数据到 {output_file}")
            print(f"总共生成 {len(poses_3d)} 帧3D姿态数据")
            print(f"平均处理速度: {fps_display} FPS")
        
        # 释放资源
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()