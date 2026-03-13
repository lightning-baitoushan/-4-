import cv2
import mediapipe as mp
import time
import numpy as np
import threading
import queue
import re
import os
import json
import socket
from vosk import Model, KaldiRecognizer
from PIL import ImageFont, ImageDraw, Image

class KalmanFilter2D:
    """二维卡尔曼滤波器，用于平滑手势坐标"""

    def __init__(self, process_noise=1e-5, measurement_noise=1e-4):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.processNoiseCov = process_noise * np.eye(4, dtype=np.float32)
        self.kalman.measurementNoiseCov = measurement_noise * np.eye(2, dtype=np.float32)
        self.kalman.errorCovPost = 1.0 * np.eye(4, dtype=np.float32)
        self.initialized = False
        self.last_x = 0
        self.last_y = 0
        self.last_time = time.time()

    def predict_and_correct(self, x, y, adaptive=True):
        measurement = np.array([[x], [y]], np.float32)

        if not self.initialized:
            self.kalman.statePost = np.array([[x], [y], [0], [0]], np.float32)
            self.initialized = True
            self.last_x = x
            self.last_y = y
            self.last_time = time.time()
            return x, y

        if adaptive:
            current_time = time.time()
            dt = current_time - self.last_time
            if dt > 0:
                velocity = np.sqrt((x - self.last_x) ** 2 + (y - self.last_y) ** 2) / dt
                base_process_noise = 1e-5
                base_measurement_noise = 1e-4

                if velocity > 0.5:
                    process_noise = base_process_noise * 10
                    measurement_noise = base_measurement_noise * 0.5
                elif velocity < 0.1:
                    process_noise = base_process_noise * 0.1
                    measurement_noise = base_measurement_noise * 2
                else:
                    process_noise = base_process_noise
                    measurement_noise = base_measurement_noise

                self.kalman.processNoiseCov = process_noise * np.eye(4, dtype=np.float32)
                self.kalman.measurementNoiseCov = measurement_noise * np.eye(2, dtype=np.float32)

            self.last_x = x
            self.last_y = y
            self.last_time = current_time

        prediction = self.kalman.predict()
        corrected = self.kalman.correct(measurement)

        return float(corrected[0][0]), float(corrected[1][0])

class ESP8266ArmController:
    def __init__(self, esp_ip='10.40.122.71', esp_port=8080):
        # Mediapipe 手势识别初始化
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.85,
            min_tracking_confidence=0.75
        )
        self.mp_draw = mp.solutions.drawing_utils

        # ESP8266网络通信初始化
        self.esp_ip = esp_ip
        self.esp_port = esp_port
        self.socket_client = None
        self.connection_established = False
        self.connection_thread = None

        # 舵机参数
        self.servo_angles = [90, 90, 90, 90]  # 底座, 前臂, 后臂, 钳子
        self.servo_directions = [0, 0, 0, 0]  # -1:逆时针, 0:停止, 1:顺时针
        self.last_sent_angles = [90, 90, 90, 90]

        # 舵机范围 (根据实际机械臂调整)
        self.servo_ranges = [
            (0, 180),  # 底座
            (45, 135),  # 前臂
            (30, 150),  # 后臂
            (40, 120)  # 钳子
        ]

        self.servo_names = ['底座', '前臂', '后臂', '钳子']
        self.servo_ranges_dict = {
            '底座': (0, 180),
            '前臂': (45, 135),
            '后臂': (30, 150),
            '钳子': (40, 120)
        }

        # 卡尔曼滤波器初始化（每只手21个关节点）
        self.left_kalman_filters = [KalmanFilter2D() for _ in range(21)]
        self.right_kalman_filters = [KalmanFilter2D() for _ in range(21)]

        # 图像处理参数
        self.enable_histogram_equalization = True
        self.enable_gaussian_blur = True
        self.enable_kalman_filter = True
        self.enable_adaptive_kalman = True

        # 语音系统
        self.voice_enabled = True  # 默认开启语音
        self.voice_paused = False
        self.voice_command_queue = queue.Queue()
        self.voice_thread = None
        self.voice_recognizer = None
        self.audio_stream = None
        self.pyaudio_instance = None

        # 初始化语音识别系统（默认使用Vosk）
        self.init_voice_recognition()

        # 初始化网络连接
        self.init_network_connection()

        # 打印系统状态
        self.print_system_status()

        # 测试数字提取功能
        self.test_number_extraction()

        # 启动语音识别线程
        self.start_voice_recognition()

    def print_system_status(self):
        """打印系统初始化状态"""
        print("\n" + "=" * 50)
        print("ESP8266机械臂控制系统")
        print("=" * 50)
        print("✅ Mediapipe手势识别已初始化")
        print(f"✅ 网络连接: {'已连接' if self.connection_established else '未连接'}")
        print(f"✅ ESP8266地址: {self.esp_ip}:{self.esp_port}")
        print(f"✅ 语音识别 (Vosk): {'已初始化' if self.voice_recognizer else '未初始化'}")
        print(f"✅ 卡尔曼滤波: {'启用' if self.enable_kalman_filter else '禁用'}")
        print(f"✅ 直方图均衡化: {'启用' if self.enable_histogram_equalization else '禁用'}")
        print("=" * 50)

    def init_network_connection(self):
        """初始化网络连接"""
        def connect_to_esp():
            while True:
                try:
                    self.socket_client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    self.socket_client.settimeout(5)
                    self.socket_client.connect((self.esp_ip, self.esp_port))
                    self.connection_established = True
                    print(f"✅ 成功连接到ESP8266: {self.esp_ip}:{self.esp_port}")
                    break
                except Exception as e:
                    print(f"⚠️  连接ESP8266失败: {e}")
                    print("正在重试...")
                    time.sleep(3)

        self.connection_thread = threading.Thread(target=connect_to_esp, daemon=True)
        self.connection_thread.start()

    def init_voice_recognition(self):
        """初始化语音识别系统"""
        # 初始化Vosk语音识别
        model_path = "vosk-model-small-cn-0.22"
        if not os.path.exists(model_path):
            print(f"⚠️  警告: Vosk模型未找到，语音识别不可用")
            print(f"请下载模型: https://alphacephei.com/vosk/models")
            print(f"解压到当前目录: {model_path}")
            return

        try:
            model = Model(model_path)
            # 设置关键词列表，只关注命令相关词汇，提高识别速度
            keywords = [
                "底座", "前臂", "后臂", "钳子", "底", "前", "后", "钳",
                "顺时针", "逆时针", "左转", "右转", "向上", "向下", "张开", "闭合",
                "度", "角度",
                "零", "一", "二", "三", "四", "五", "六", "七", "八", "九", "十", "百",
                "归位", "复位", "暂停", "继续", "开启", "关闭"
            ]
            # 使用关键词列表创建识别器，提高识别速度和准确性
            self.voice_recognizer = KaldiRecognizer(model, 16000, json.dumps(keywords, ensure_ascii=False))
            print("✅ 语音识别系统初始化成功")

        except Exception as e:
            print(f"❌ 语音识别系统初始化失败: {e}")
            self.voice_recognizer = None

    def extract_angle_from_text(self, t):
        """从文本中提取角度数字"""
        if not t:
            return None

        # 方法1：直接匹配连续数字（如25、150等）
        matches = re.findall(r'\d+', t)
        if matches:
            # 取最长匹配的数字
            max_len_match = max(matches, key=len)
            angle = int(max_len_match)
            return angle

        # 方法2：匹配复杂中文数字组合（如二十五、一百二十等）
        complex_chinese_nums = {
            '二十五': 25, '二十六': 26, '二十七': 27, '二十八': 28, '二十九': 29,
            '三十五': 35, '三十六': 36, '三十七': 37, '三十八': 38, '三十九': 39,
            '四十五': 45, '四十六': 46, '四十七': 47, '四十八': 48, '四十九': 49,
            '五十五': 55, '五十六': 56, '五十七': 57, '五十八': 58, '五十九': 59,
            '六十五': 65, '六十六': 66, '六十七': 67, '六十八': 68, '六十九': 69,
            '七十五': 75, '七十六': 76, '七十七': 77, '七十八': 78, '七十九': 79,
            '八十五': 85, '八十六': 86, '八十七': 87, '八十八': 88, '八十九': 89,
            '九十五': 95, '九十六': 96, '九十七': 97, '九十八': 98, '九十九': 99,
            '一百': 100, '一百零五': 105, '一百一十': 110, '一百二十': 120,
            '一百五十': 150, '一百八十': 180,
            '二十': 20, '三十': 30, '四十': 40, '五十': 50,
            '六十': 60, '七十': 70, '八十': 80, '九十': 90,
            '十一': 11, '十二': 12, '十三': 13, '十四': 14, '十五': 15,
            '十六': 16, '十七': 17, '十八': 18, '十九': 19
        }

        for chinese_num, value in complex_chinese_nums.items():
            if chinese_num in t:
                return value

        # 方法3：匹配单个中文数字
        simple_chinese_nums = {
            '一': 1, '二': 2, '两': 2, '三': 3, '四': 4, '五': 5,
            '六': 6, '七': 7, '八': 8, '九': 9, '十': 10,
            '零': 0, '洞': 0, '幺': 1
        }

        # 查找文本中的中文数字
        found_nums = []
        for char in t:
            if char in simple_chinese_nums:
                found_nums.append(simple_chinese_nums[char])

        if found_nums:
            # 如果找到多个数字，组合它们
            if len(found_nums) >= 2:
                # 处理各种数字组合情况
                # 1. 处理包含"十"的组合，如"二十四" -> [2, 10, 4] -> 24
                if 10 in found_nums:
                    ten_index = found_nums.index(10)
                    # 检查"十"前面的数字
                    if ten_index > 0:
                        # 前面有数字，如"二十四" -> 2*10 + 4
                        if ten_index + 1 < len(found_nums):
                            combined = found_nums[ten_index - 1] * 10 + found_nums[ten_index + 1]
                            return combined
                        else:
                            # 前面有数字，后面没有，如"二十" -> 2*10
                            combined = found_nums[ten_index - 1] * 10
                            return combined
                    else:
                        # "十"在开头，如"十四" -> 10 + 4
                        if ten_index + 1 < len(found_nums):
                            combined = 10 + found_nums[ten_index + 1]
                            return combined
                
                # 2. 处理连续数字组合，如"二五" -> 25, "一二三" -> 123
                # 过滤掉10（十），因为它需要特殊处理
                filtered_nums = [num for num in found_nums if num != 10]
                if filtered_nums:
                    # 组合所有数字
                    combined = 0
                    for num in filtered_nums:
                        combined = combined * 10 + num
                    return combined
            else:
                # 单个数字，检查是否是"十"
                if found_nums[0] == 10:
                    # 只有"十"，返回10
                    return 10
                else:
                    return found_nums[0]

        return None

    def test_number_extraction(self):
        """测试数字提取功能"""
        print("\n" + "=" * 50)
        print("数字提取测试")
        print("=" * 50)

        test_cases = [
            "底座左转25度",
            "前臂向上15度",
            "钳子张开30度",
            "后臂向下45度",
            "底座到90度",
            "前臂到一百二十度",
            "左转三十五度",
            "右转五十度",
            "张开二十度",
            "闭合十度"
        ]

        for test in test_cases:
            angle = self.extract_angle_from_text(test)
            if angle is not None:
                print(f"'{test}' -> {angle}度")
            else:
                print(f"'{test}' -> 未提取到数字")

        print("=" * 50)

    def start_voice_recognition(self):
        """启动语音识别线程"""
        if not self.voice_recognizer:
            print("语音识别器不可用，跳过语音识别")
            return

        def voice_recognition_worker():
            try:
                # 使用Vosk离线识别
                self.start_vosk_recognition()

            except ImportError as e:
                print(f"❌ 依赖项缺失: {e}")
                print("请安装: pip install pyaudio")
                print("或使用Windows命令: pip install pipwin && pipwin install pyaudio")
            except Exception as e:
                print(f"❌ 语音识别线程启动失败: {e}")

        # 启动语音识别线程
        self.voice_thread = threading.Thread(target=voice_recognition_worker, daemon=True)
        self.voice_thread.start()
        print("语音识别线程已启动")

    def start_vosk_recognition(self):
        """启动Vosk离线语音识别"""
        if not self.voice_recognizer:
            print("Vosk模型不可用，无法启动离线识别")
            return

        try:
            import pyaudio

            # 初始化音频
            self.pyaudio_instance = pyaudio.PyAudio()

            # 获取音频设备
            print("\n搜索音频输入设备...")
            input_devices = []
            for i in range(self.pyaudio_instance.get_device_count()):
                dev_info = self.pyaudio_instance.get_device_info_by_index(i)
                if dev_info['maxInputChannels'] > 0:
                    input_devices.append((i, dev_info['name']))
                    print(f"  设备 {i}: {dev_info['name']}")

            # 选择设备
            device_index = None
            for idx, name in input_devices:
                if 'microphone' in name.lower() or 'mic' in name.lower() or '麦克风' in name:
                    device_index = idx
                    print(f"选择设备: {name}")
                    break

            if device_index is None and input_devices:
                device_index = input_devices[0][0]
                print(f"使用默认设备: {input_devices[0][1]}")

            # 打开音频流
            self.audio_stream = self.pyaudio_instance.open(
                format=pyaudio.paInt16,
                channels=1,
                rate=16000,
                input=True,
                frames_per_buffer=8000,
                input_device_index=device_index,
                stream_callback=None
            )
            self.audio_stream.start_stream()

            print("✅ Vosk语音识别已启动，请说中文指令...")

            # 语音识别循环
            while True:
                try:
                    data = self.audio_stream.read(8000, exception_on_overflow=False)

                    if self.voice_recognizer.AcceptWaveform(data):
                        result = json.loads(self.voice_recognizer.Result())
                        if 'text' in result and result['text'].strip():
                            command = result['text'].strip()
                            print(f"\n🎤 离线识别到指令: '{command}'")
                            self.parse_voice_command(command)
                            # 处理完命令后重置识别器
                            self.voice_recognizer.Reset()

                except OSError as e:
                    if "Input overflowed" in str(e):
                        continue
                    else:
                        print(f"音频流错误: {e}")
                        time.sleep(0.1)
                except Exception as e:
                    print(f"语音处理异常: {e}")
                    time.sleep(0.1)

        except ImportError:
            print("❌ 未安装pyaudio，离线语音识别不可用")
        except Exception as e:
            print(f"❌ Vosk离线识别启动失败: {e}")

    def parse_voice_command(self, text):
        """解析语音指令"""
        original_text = text
        text = text.lower().strip()

        # 如果语音模式关闭，不处理语音指令
        if not self.voice_enabled:
            print("语音模式已关闭，忽略指令")
            return

        # 语音模式控制
        if '开启语音' in text or '打开语音' in text:
            self.voice_enabled = True
            print("语音模式已开启")
            return

        if '关闭语音' in text or '关闭语音模式' in text:
            self.voice_enabled = False
            print("语音模式已关闭")
            return

        if '暂停' in text or '停止运动' in text:
            self.voice_paused = True
            print("机械臂运动已暂停")
            return

        if '继续' in text or '恢复运动' in text:
            self.voice_paused = False
            print("机械臂运动已恢复")
            return

        if '归位' in text or '回到中间' in text or '复位' in text:
            target_angles = [90, 90, 90, 90]
            self.voice_command_queue.put(('set_all', target_angles))
            print("所有舵机已归位")
            return

        if '急停' in text or '紧急停止' in text:
            self.servo_directions = [0, 0, 0, 0]
            print("紧急停止已执行")
            return

        # 快速匹配：识别单个关键词对应到相应舵机
        quick_match = {
            '底': ('底座', 0),
            '前': ('前臂', 1),
            '后': ('后臂', 2),
            '钳': ('钳子', 3)
        }
        
        for key, (name, index) in quick_match.items():
            if key in text:
                print(f"✅ 快速匹配到舵机: {name}")
                
                # 提取角度
                angle = self.extract_angle_from_text(text)

                if angle is not None:
                    print(f"✅ 提取到角度: {angle}度")

                    # 处理 "到XX度" 命令
                    if '到' in text or '转到' in text or '设置为' in text or '位置' in text:
                        min_angle, max_angle = self.servo_ranges_dict[name]
                        if min_angle <= angle <= max_angle:
                            self.voice_command_queue.put(('set', index, angle))
                            print(f"{name}已转到{angle}度")
                            return
                        else:
                            print(f"{name}角度超出范围")
                            return

                    # 处理方向命令
                    current_angle = self.servo_angles[index]

                    # 优先处理顺逆时针命令
                    if '逆时针' in text:
                        new_angle = max(self.servo_ranges[index][0], current_angle - angle)
                        self.voice_command_queue.put(('set', index, new_angle))
                        print(f"{name}逆时针转动{angle}度")
                        return
                    elif '顺时针' in text:
                        new_angle = min(self.servo_ranges[index][1], current_angle + angle)
                        self.voice_command_queue.put(('set', index, new_angle))
                        print(f"{name}顺时针转动{angle}度")
                        return
                    elif '左' in text or '往左' in text:
                        new_angle = max(self.servo_ranges[index][0], current_angle - angle)
                        self.voice_command_queue.put(('set', index, new_angle))
                        print(f"{name}左转{angle}度")
                        return
                    elif '右' in text or '往右' in text:
                        new_angle = min(self.servo_ranges[index][1], current_angle + angle)
                        self.voice_command_queue.put(('set', index, new_angle))
                        print(f"{name}右转{angle}度")
                        return

        # 常规匹配：完整舵机名称
        for i, name in enumerate(self.servo_names):
            if name in text:
                print(f"✅ 找到舵机: {name}")

                # 提取角度
                angle = self.extract_angle_from_text(text)

                if angle is not None:
                    print(f"✅ 提取到角度: {angle}度")

                    # 处理 "到XX度" 命令
                    if '到' in text or '转到' in text or '设置为' in text or '位置' in text:
                        min_angle, max_angle = self.servo_ranges_dict[name]
                        if min_angle <= angle <= max_angle:
                            self.voice_command_queue.put(('set', i, angle))
                            print(f"{name}已转到{angle}度")
                            return
                        else:
                            print(f"{name}角度超出范围")
                            return

                    # 处理方向命令
                    current_angle = self.servo_angles[i]

                    # 优先处理顺逆时针命令
                    if '逆时针' in text:
                        new_angle = max(self.servo_ranges[i][0], current_angle - angle)
                        self.voice_command_queue.put(('set', i, new_angle))
                        print(f"{name}逆时针转动{angle}度")
                        return
                    elif '顺时针' in text:
                        new_angle = min(self.servo_ranges[i][1], current_angle + angle)
                        self.voice_command_queue.put(('set', i, new_angle))
                        print(f"{name}顺时针转动{angle}度")
                        return
                    elif '左' in text or '往左' in text:
                        new_angle = max(self.servo_ranges[i][0], current_angle - angle)
                        self.voice_command_queue.put(('set', i, new_angle))
                        print(f"{name}左转{angle}度")
                        return
                    elif '右' in text or '往右' in text:
                        new_angle = min(self.servo_ranges[i][1], current_angle + angle)
                        self.voice_command_queue.put(('set', i, new_angle))
                        print(f"{name}右转{angle}度")
                        return

        print(f"❌ 未识别指令: '{original_text}'")

    def process_voice_commands(self):
        """处理语音命令队列"""
        try:
            while not self.voice_command_queue.empty():
                command = self.voice_command_queue.get_nowait()
                if command[0] == 'set':
                    _, servo_idx, angle = command
                    self.servo_angles[servo_idx] = angle
                    print(f"设置舵机 {servo_idx} 到 {angle}度")
                    self.send_to_esp8266()
                elif command[0] == 'set_all':
                    _, angles = command
                    for i in range(4):
                        self.servo_angles[i] = angles[i]
                    print("设置所有舵机到90度")
                    self.send_to_esp8266()
        except queue.Empty:
            pass

    def preprocess_frame(self, frame):
        """图像预处理"""
        processed = frame.copy()

        # 直方图均衡化
        if self.enable_histogram_equalization:
            yuv = cv2.cvtColor(processed, cv2.COLOR_BGR2YUV)
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            processed = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)

        # 高斯模糊
        if self.enable_gaussian_blur:
            processed = cv2.GaussianBlur(processed, (3, 3), 0)

        return processed

    def apply_kalman_filter(self, hand_landmarks, kalman_filters):
        """应用卡尔曼滤波器平滑手势数据"""
        for i in range(21):
            x = hand_landmarks.landmark[i].x
            y = hand_landmarks.landmark[i].y
            filtered_x, filtered_y = kalman_filters[i].predict_and_correct(x, y, self.enable_adaptive_kalman)
            hand_landmarks.landmark[i].x = filtered_x
            hand_landmarks.landmark[i].y = filtered_y
        return hand_landmarks

    def is_finger_extended(self, landmarks, finger_idx, is_left_hand=False):
        """判断手指是否伸直"""
        # 手指关节点索引
        finger_tips = [4, 8, 12, 16, 20]  # 拇指，食指，中指，无名指，小指
        finger_pips = [2, 6, 10, 14, 18]  # 近端指间关节

        if finger_idx == 0:  # 拇指
            if is_left_hand:
                # 左手拇指：镜像翻转后，x坐标比较需要反转
                return landmarks[finger_tips[0]].x > landmarks[finger_pips[0]].x
            else:
                # 右手拇指：正常判断
                return landmarks[finger_tips[0]].x < landmarks[finger_pips[0]].x
        else:  # 其他手指
            return landmarks[finger_tips[finger_idx]].y < landmarks[finger_pips[finger_idx]].y

    def get_hand_info(self, hand_landmarks, is_left_hand):
        """获取手势信息"""
        landmarks = hand_landmarks.landmark

        # 检测每根手指状态
        fingers_extended = []
        for i in range(5):
            fingers_extended.append(self.is_finger_extended(landmarks, i, is_left_hand))

        # 根据左右手调整控制逻辑
        if is_left_hand:
            # 左手：食指(1)控制底座，拇指(0)控制钳子
            control_finger_1 = 1  # 食指
            control_finger_2 = 0  # 拇指
        else:
            # 右手：拇指(0)控制后臂，食指(1)控制前臂
            control_finger_1 = 0  # 拇指
            control_finger_2 = 1  # 食指

        # 控制手指状态
        finger1_extended = fingers_extended[control_finger_1]
        finger2_extended = fingers_extended[control_finger_2]

        # 中指、无名指、小指状态（用于确定转动方向）
        middle_finger = fingers_extended[2]  # 中指
        ring_finger = fingers_extended[3]  # 无名指
        pinky_finger = fingers_extended[4]  # 小指

        # 三个手指同时弯曲 = 顺时针，同时伸直 = 逆时针
        if not middle_finger and not ring_finger and not pinky_finger:
            rotation_direction = 1  # 顺时针
        elif middle_finger and ring_finger and pinky_finger:
            rotation_direction = -1  # 逆时针
        else:
            rotation_direction = 0  # 停止

        return {
            'finger1_extended': finger1_extended,
            'finger2_extended': finger2_extended,
            'rotation_direction': rotation_direction,
            'all_fingers': fingers_extended
        }

    def update_servo_directions(self, left_hand_info, right_hand_info):
        """更新舵机转动方向"""
        # 初始化所有舵机为停止状态
        self.servo_directions = [0, 0, 0, 0]

        # 左手控制：食指(底座)和拇指(钳子)
        if left_hand_info:
            if not left_hand_info['finger1_extended']:  # 食指弯曲
                self.servo_directions[0] = left_hand_info['rotation_direction']  # 底座
            if not left_hand_info['finger2_extended']:  # 拇指弯曲
                self.servo_directions[3] = left_hand_info['rotation_direction']  # 钳子

        # 右手控制：拇指(后臂)和食指(前臂)
        if right_hand_info:
            if not right_hand_info['finger1_extended']:  # 拇指弯曲
                self.servo_directions[2] = right_hand_info['rotation_direction']  # 后臂
            if not right_hand_info['finger2_extended']:  # 食指弯曲
                self.servo_directions[1] = right_hand_info['rotation_direction']  # 前臂

    def update_servo_angles(self):
        """根据方向更新舵机角度"""
        step = 1  # 每次调整的步长

        for i in range(4):
            min_angle, max_angle = self.servo_ranges[i]

            if self.servo_directions[i] == 1:  # 顺时针
                self.servo_angles[i] = min(max_angle, self.servo_angles[i] + step)
            elif self.servo_directions[i] == -1:  # 逆时针
                self.servo_angles[i] = max(min_angle, self.servo_angles[i] - step)
            # 为0时不改变角度

    def send_to_esp8266(self):
        """发送舵机角度到ESP8266"""
        if not self.connection_established or not self.socket_client:
            return

        # 检查角度是否有变化
        angles_changed = False
        for i in range(4):
            if abs(self.servo_angles[i] - self.last_sent_angles[i]) > 0:
                angles_changed = True
                break

        if not angles_changed:
            return

        # 构建数据包：角度1,角度2,角度3,角度4
        data = f"{self.servo_angles[0]},{self.servo_angles[1]},{self.servo_angles[2]},{self.servo_angles[3]}\n"

        try:
            self.socket_client.send(data.encode())
            self.last_sent_angles = self.servo_angles.copy()
            print(f"📤 发送到ESP8266: {data.strip()}")
        except Exception as e:
            print(f"❌ 发送数据失败: {e}")
            # 尝试重新连接
            self.connection_established = False
            self.init_network_connection()

    def draw_control_info(self, frame, left_hand_info, right_hand_info):
        """在画面上绘制控制信息"""
        h, w = frame.shape[:2]

        # 将OpenCV图像转换为PIL图像以正确显示中文
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(frame_pil)
        
        # 加载中文字体
        try:
            # 尝试使用Windows系统字体
            font = ImageFont.truetype("simhei.ttf", 14)
        except:
            try:
                # 尝试使用其他系统字体
                font = ImageFont.truetype("Arial.ttf", 14)
            except:
                # 如果都失败，使用默认字体
                font = ImageFont.load_default()

        # 绘制标题
        draw.text((10, 15), "ESP8266机械臂控制系统", font=font, fill=(0, 255, 255))

        # 绘制舵机角度信息
        servo_text = f"舵机角度: 底座={self.servo_angles[0]} 前臂={self.servo_angles[1]} 后臂={self.servo_angles[2]} 钳子={self.servo_angles[3]}"
        draw.text((10, 55), servo_text, font=font, fill=(255, 255, 0))

        # 绘制转动方向
        direction_text = f"转动方向: {self.servo_directions}"
        draw.text((10, 85), direction_text, font=font, fill=(255, 255, 0))

        # 绘制网络状态
        network_status = f"网络连接: {'✅ 已连接' if self.connection_established else '❌ 未连接'}"
        draw.text((10, 115), network_status, font=font, fill=(0, 255, 0) if self.connection_established else (0, 0, 255))

        # 绘制ESP8266地址
        esp_address = f"ESP8266地址: {self.esp_ip}:{self.esp_port}"
        draw.text((10, 145), esp_address, font=font, fill=(200, 200, 200))

        # 绘制手势控制说明
        instructions_y = 175
        instructions = [
            "左手控制:",
            "  食指弯曲 -> 底座转动",
            "  拇指弯曲 -> 钳子转动",
            "右手控制:",
            "  拇指弯曲 -> 后臂转动",
            "  食指弯曲 -> 前臂转动",
            "方向控制:",
            "  中+无+小指弯曲 -> 顺时针",
            "  中+无+小指伸直 -> 逆时针"
        ]

        for instruction in instructions:
            draw.text((10, instructions_y), instruction, font=font, fill=(200, 200, 200))
            instructions_y += 25

        # 绘制语音状态
        voice_mode = '开启' if self.voice_enabled else '关闭'
        motion_status = '暂停' if self.voice_paused else '正常'
        
        voice_status = f"语音模式: {voice_mode} | 运动状态: {motion_status}"
        draw.text((w - 400, 15), voice_status, font=font, fill=(0, 255, 0))

        # 绘制优化状态
        optimizations = [
            f"卡尔曼滤波: {'启用' if self.enable_kalman_filter else '禁用'}",
            f"直方均衡化: {'启用' if self.enable_histogram_equalization else '禁用'}"
        ]

        for i, opt in enumerate(optimizations):
            draw.text((w - 200, 55 + i * 25), opt, font=font, fill=(100, 255, 100))

        # 绘制手势状态
        if left_hand_info:
            left_status = f"左手: 食指{'弯曲' if not left_hand_info['finger1_extended'] else '伸直'} 拇指{'弯曲' if not left_hand_info['finger2_extended'] else '伸直'}"
            draw.text((10, h - 60), left_status, font=font, fill=(255, 100, 100))

        if right_hand_info:
            right_status = f"右手: 拇指{'弯曲' if not right_hand_info['finger1_extended'] else '伸直'} 食指{'弯曲' if not right_hand_info['finger2_extended'] else '伸直'}"
            draw.text((10, h - 30), right_status, font=font, fill=(100, 100, 255))

        # 将PIL图像转换回OpenCV图像
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        return frame

    def run(self):
        """主运行循环"""
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("❌ 无法打开摄像头")
            return

        print("\n" + "=" * 50)
        print("系统已启动！")
        print("控制说明:")
        print("  1. 手势控制:")
        print("     - 左手食指控制底座，左手拇指控制钳子")
        print("     - 右手拇指控制后臂，右手食指控制前臂")
        print("     - 中+无+小指弯曲:顺时针，伸直:逆时针")
        print("  2. 语音控制:")
        print("     - 说'开启语音'/'关闭语音' 控制语音模式")
        print("     - 说'底座左转30度'/'钳子张开20度'等")
        print("     - 说'归位'让所有舵机回到中间位置")
        print("     - 说'暂停'/'继续' 控制运动")
        print("  3. 键盘快捷键:")
        print("     - Q: 退出")
        print("     - R: 复位舵机")
        print("     - V: 切换语音模式")
        print("     - K: 切换卡尔曼滤波")
        print("     - H: 切换直方均衡化")
        print("=" * 50)

        try:
            while True:
                # 读取摄像头帧
                ret, frame = cap.read()
                if not ret:
                    print("❌ 无法读取摄像头帧")
                    break

                # 镜像翻转（更符合直觉）
                frame = cv2.flip(frame, 1)

                # 图像预处理
                processed_frame = self.preprocess_frame(frame)

                # 转换为RGB格式
                rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)

                # 手势识别
                results = self.hands.process(rgb_frame)

                # 处理语音命令
                self.process_voice_commands()

                # 初始化手部信息
                left_hand_info = None
                right_hand_info = None

                # 处理识别到的手势
                if results.multi_hand_landmarks and results.multi_handedness:
                    for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                        hand_label = handedness.classification[0].label
                        is_left_hand = (hand_label == 'Left')

                        # 应用卡尔曼滤波
                        if self.enable_kalman_filter:
                            if is_left_hand:
                                hand_landmarks = self.apply_kalman_filter(hand_landmarks, self.left_kalman_filters)
                            else:
                                hand_landmarks = self.apply_kalman_filter(hand_landmarks, self.right_kalman_filters)

                        # 绘制手部关键点
                        self.mp_draw.draw_landmarks(frame, hand_landmarks, self.mp_hands.HAND_CONNECTIONS)

                        # 获取手部信息
                        hand_info = self.get_hand_info(hand_landmarks, is_left_hand)

                        # 更新手部信息
                        if is_left_hand:
                            left_hand_info = hand_info
                        else:
                            right_hand_info = hand_info

                # 只有在语音未暂停时才更新手势控制
                if not self.voice_paused:
                    # 更新舵机转动方向
                    self.update_servo_directions(left_hand_info, right_hand_info)

                    # 更新舵机角度
                    self.update_servo_angles()

                    # 发送到ESP8266
                    self.send_to_esp8266()

                # 绘制控制信息
                frame = self.draw_control_info(frame, left_hand_info, right_hand_info)

                # 显示画面
                cv2.imshow('ESP8266机械臂控制', frame)

                # 键盘控制
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):  # 退出
                    break
                elif key == ord('r'):  # 复位
                    self.servo_angles = [90, 90, 90, 90]
                    self.send_to_esp8266()
                    print("舵机已复位")
                elif key == ord('v'):  # 切换语音模式
                    self.voice_enabled = not self.voice_enabled
                    status = "开启" if self.voice_enabled else "关闭"
                    print(f"语音模式已{status}")
                elif key == ord('k'):  # 切换卡尔曼滤波
                    self.enable_kalman_filter = not self.enable_kalman_filter
                    status = "开启" if self.enable_kalman_filter else "关闭"
                    print(f"卡尔曼滤波已{status}")
                elif key == ord('h'):  # 切换直方均衡化
                    self.enable_histogram_equalization = not self.enable_histogram_equalization
                    status = "开启" if self.enable_histogram_equalization else "关闭"
                    print(f"直方均衡化已{status}")

        except KeyboardInterrupt:
            print("\n用户中断程序")
        finally:
            # 释放资源
            cap.release()
            cv2.destroyAllWindows()
            if self.socket_client:
                try:
                    self.socket_client.close()
                except:
                    pass
            print("程序已退出")

if __name__ == "__main__":
    # 默认ESP8266 IP地址和端口
    # 请根据实际网络环境修改
    controller = ESP8266ArmController(esp_ip='10.40.122.71', esp_port=8080)
    controller.run()