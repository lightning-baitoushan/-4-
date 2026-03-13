import cv2
import mediapipe as mp
import serial
import serial.tools.list_ports
import time
import numpy as np
import threading
import queue
import re
import os
import json
from vosk import Model, KaldiRecognizer
import sys
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


class MultiModalArmController:
    def __init__(self, serial_port='COM3', baud_rate=9600):
        # Mediapipe 手势识别初始化
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.85,
            min_tracking_confidence=0.75
        )
        self.mp_draw = mp.solutions.drawing_utils

        # 串口通信初始化
        try:
            self.serial_port_name = self.find_arduino_port()
            if not self.serial_port_name:
                self.serial_port_name = serial_port
                print(f"使用指定串口: {self.serial_port_name}")

            self.arduino = serial.Serial(self.serial_port_name, baud_rate, timeout=1)
            time.sleep(2)  # 等待Arduino初始化
            print(f"✅ 串口连接成功: {self.serial_port_name}")
        except Exception as e:
            print(f"❌ 串口连接失败: {e}")
            self.arduino = None

        # 舵机参数
        self.servo_angles = [90, 90, 90, 90]  # 底座, 小臂, 大臂, 钳子
        self.servo_directions = [0, 0, 0, 0]  # -1:逆时针, 0:停止, 1:顺时针
        self.last_sent_angles = [90, 90, 90, 90]

        # 舵机范围 (根据实际机械臂调整)
        self.servo_ranges = [
            (0, 180),  # 底座
            (45, 135),  # 小臂
            (30, 150),  # 大臂
            (40, 120)  # 钳子
        ]

        self.servo_names = ['底座', '小臂', '大臂', '钳子']
        self.servo_ranges_dict = {
            '底座': (0, 180),
            '小臂': (45, 135),
            '大臂': (30, 150),
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

        # 打印系统状态
        self.print_system_status()

        # 测试数字提取功能
        self.test_number_extraction()

        # 启动语音识别线程
        self.start_voice_recognition()

    def print_system_status(self):
        """打印系统初始化状态"""
        print("\n" + "=" * 50)
        print("多模态机械臂控制系统")
        print("=" * 50)
        print("✅ Mediapipe手势识别已初始化")
        print(f"✅ 串口连接: {'已连接' if self.arduino else '未连接'}")
        print(f"✅ 语音识别 (Vosk): {'已初始化' if self.voice_recognizer else '未初始化'}")
        print(f"✅ 卡尔曼滤波: {'启用' if self.enable_kalman_filter else '禁用'}")
        print(f"✅ 直方图均衡化: {'启用' if self.enable_histogram_equalization else '禁用'}")
        print("=" * 50)

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
                "底座", "小臂", "大臂", "钳子", "底", "小", "大", "钳",
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
        """从文本中提取角度数字 - 改进版"""
        if not t:
            return None

        print(f"  提取角度从文本: '{t}'")

        # 方法1：直接匹配连续数字（如25、150等）
        matches = re.findall(r'\d+', t)
        if matches:
            # 取最长匹配的数字
            max_len_match = max(matches, key=len)
            angle = int(max_len_match)
            print(f"  匹配到数字: {matches}, 使用: {angle}")
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
                print(f"  匹配到复杂中文数字: {chinese_num} -> {value}")
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
            print(f"  找到单个中文数字: {found_nums}")
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
                            print(f"  组合数字 (含十): {found_nums} -> {combined}")
                            return combined
                        else:
                            # 前面有数字，后面没有，如"二十" -> 2*10
                            combined = found_nums[ten_index - 1] * 10
                            print(f"  组合数字 (含十): {found_nums} -> {combined}")
                            return combined
                    else:
                        # "十"在开头，如"十四" -> 10 + 4
                        if ten_index + 1 < len(found_nums):
                            combined = 10 + found_nums[ten_index + 1]
                            print(f"  组合数字 (含十): {found_nums} -> {combined}")
                            return combined
                
                # 2. 处理连续数字组合，如"二五" -> 25, "一二三" -> 123
                # 过滤掉10（十），因为它需要特殊处理
                filtered_nums = [num for num in found_nums if num != 10]
                if filtered_nums:
                    # 组合所有数字
                    combined = 0
                    for num in filtered_nums:
                        combined = combined * 10 + num
                    print(f"  组合数字 (连续): {filtered_nums} -> {combined}")
                    return combined
            else:
                # 单个数字，检查是否是"十"
                if found_nums[0] == 10:
                    # 只有"十"，返回10
                    print(f"  单个数字: {found_nums} -> 10")
                    return 10
                else:
                    return found_nums[0]

        # 方法4：匹配"X十Y"的模式
        pattern_ten = re.search(r'(\S)十(\S)', t)
        if pattern_ten:
            tens = pattern_ten.group(1)
            ones = pattern_ten.group(2)
            tens_dict = {'二': 2, '两': 2, '三': 3, '四': 4, '五': 5,
                         '六': 6, '七': 7, '八': 8, '九': 9}
            ones_dict = {'一': 1, '二': 2, '三': 3, '四': 4, '五': 5,
                         '六': 6, '七': 7, '八': 8, '九': 9, '零': 0}

            if tens in tens_dict and ones in ones_dict:
                angle = tens_dict[tens] * 10 + ones_dict[ones]
                print(f"  匹配到'X十Y'模式: {tens}十{ones} -> {angle}")
                return angle

        # 方法5：匹配"X十"的模式（如"二十"、"三十"）
        pattern_tens = re.search(r'(\S)十', t)
        if pattern_tens:
            tens = pattern_tens.group(1)
            tens_dict = {'二': 2, '两': 2, '三': 3, '四': 4, '五': 5,
                         '六': 6, '七': 7, '八': 8, '九': 9}
            if tens in tens_dict:
                angle = tens_dict[tens] * 10
                print(f"  匹配到'X十'模式: {tens}十 -> {angle}")
                return angle

        # 方法6：匹配带"点"的数字
        if '点' in t:
            parts = t.split('点')
            if parts and parts[0] in simple_chinese_nums:
                angle = simple_chinese_nums[parts[0]]
                print(f"  匹配到带'点'的数字: {t} -> {angle}")
                return angle

        print(f"  未找到数字")
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
            "闭合十度",
            "底座左转二五度",  # 语音可能识别为"二五"
            "前臂向上二十五度",
            "钳子张开一百度",
            "归位",
            "开启语音",
            "左转10度",
            "右转5度",
            "上移20度",
            "下移15度",
            "左转一百八十度",
            "右转一百五十度"
        ]

        for test in test_cases:
            print(f"\n测试: '{test}'")
            angle = self.extract_angle_from_text(test)
            if angle is not None:
                print(f"  结果: {angle}度")
            else:
                print(f"  结果: 未提取到数字")

        print("=" * 50)

    def log(self, text):
        """日志输出"""
        print(f"�  {text}")

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
                import traceback
                traceback.print_exc()

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
            self.log("语音系统已就绪，请说中文指令")

            # 语音识别循环
            while True:
                try:
                    data = self.audio_stream.read(8000, exception_on_overflow=False)

                    if self.voice_recognizer.AcceptWaveform(data):
                        result = json.loads(self.voice_recognizer.Result())
                        if 'text' in result and result['text'].strip():
                            command = result['text'].strip()
                            print(f"\n🎤 离线识别到指令: '{command}'")
                            success = self.parse_voice_command(command)
                            if success:
                                # 处理完命令后重置识别器
                                self.voice_recognizer.Reset()
                    else:
                        # 处理部分结果，提高响应速度
                        partial = json.loads(self.voice_recognizer.PartialResult())
                        if 'partial' in partial and partial['partial']:
                            partial_text = partial['partial'].strip()
                            # 检查部分结果是否可以处理
                            if self._can_process_partial_result(partial_text):
                                print(f"🔄 处理部分结果: '{partial_text}'")
                                success = self.parse_voice_command(partial_text)
                                if success:
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



    def process_voice_command(self, text):
        """处理语音命令（统一接口）"""
        # 快速处理：转换为小写并去除空格
        text = text.lower().replace(' ', '')
        print(f"\n🔍 处理语音命令: '{text}'")
        # 调用现有的命令解析方法
        self.parse_voice_command(text)

    def _can_process_partial_result(self, text):
        """检查部分结果是否可以处理"""
        # 严格检查：是否包含舵机名称、方向和度
        text = text.lower()
        # 包含完整舵机名称和单个关键词
        has_servo = any(key in text for key in ['底座', '小臂', '大臂', '钳子', '底', '小', '大', '钳'])
        has_direction = any(key in text for key in ['左', '右', '顺时针', '逆时针', '向上', '向下', '张开', '闭合'])
        # 检查是否包含'度'字，确保命令完整性
        has_degree = any(key in text for key in ['度', '角度'])
        # 检查是否包含数字或中文数字
        has_number = any(char.isdigit() for char in text) or any(key in text for key in ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九', '十', '百'])
        
        # 优化判断逻辑：
        # 只有在同时检测到舵机名称、方向和度时才处理命令
        # 确保命令的完整性，避免误触发
        if has_servo and has_direction and has_degree:
            # 进一步检查：如果包含'度'字，尝试提取角度
            if has_number:
                return True
            else:
                # 检查'度'字前是否有中文数字
                degree_pos = text.find('度')
                if degree_pos > 0:
                    # 检查'度'字前的文本是否包含中文数字
                    pre_degree_text = text[:degree_pos]
                    has_chinese_number = any(key in pre_degree_text for key in ['零', '一', '二', '三', '四', '五', '六', '七', '八', '九', '十', '百'])
                    if has_chinese_number:
                        return True
            return False
        return False

    def parse_voice_command(self, text):
        """解析语音指令"""
        original_text = text
        text = text.lower().strip()

        print(f"\n🔍 原始指令: '{original_text}'")
        print(f"🔍 处理后: '{text}'")

        # 如果语音模式关闭，不处理语音指令
        if not self.voice_enabled:
            print("语音模式已关闭，忽略指令")
            return False

        # 语音模式控制
        if '开启语音' in text or '打开语音' in text:
            self.voice_enabled = True
            self.log("语音模式已开启")
            return True

        if '关闭语音' in text or '关闭语音模式' in text:
            self.voice_enabled = False
            self.log("语音模式已关闭")
            return True

        if '暂停' in text or '停止运动' in text:
            self.voice_paused = True
            self.log("机械臂运动已暂停")
            return True

        if '继续' in text or '恢复运动' in text:
            self.voice_paused = False
            self.log("机械臂运动已恢复")
            return True

        if '归位' in text or '回到中间' in text or '复位' in text:
            target_angles = [90, 90, 90, 90]
            self.voice_command_queue.put(('set_all', target_angles))
            self.log("所有舵机已归位")
            return True

        if '急停' in text or '紧急停止' in text:
            self.servo_directions = [0, 0, 0, 0]
            self.log("紧急停止已执行")
            return True

        # 快速匹配：识别单个关键词对应到相应舵机
        # 底 -> 底座, 小 -> 小臂, 大 -> 大臂
        quick_match = {
            '底': ('底座', 0),
            '小': ('小臂', 1),
            '大': ('大臂', 2)
        }
        
        for key, (name, index) in quick_match.items():
            if key in text:
                print(f"✅ 快速匹配到舵机: {name}")
                
                # 提取角度 - 使用类方法
                angle = self.extract_angle_from_text(text)

                if angle is not None:
                    print(f"✅ 提取到角度: {angle}度")

                    # 处理 "到XX度" 命令
                    if '到' in text or '转到' in text or '设置为' in text or '位置' in text:
                        min_angle, max_angle = self.servo_ranges_dict[name]
                        if min_angle <= angle <= max_angle:
                            self.voice_command_queue.put(('set', index, angle))
                            self.log(f"{name}已转到{angle}度")
                            return True
                        else:
                            self.log(f"{name}角度超出范围")
                            return True

                    # 处理方向命令
                    current_angle = self.servo_angles[index]

                    # 优先处理顺逆时针命令，忽略左右命令
                    if '逆时针' in text:
                        if index == 0:  # 底座：逆时针命令执行顺时针逻辑
                            new_angle = min(self.servo_ranges[index][1], current_angle + angle)
                            self.log(f"{name}逆时针转动{angle}度")
                        else:
                            new_angle = max(self.servo_ranges[index][0], current_angle - angle)
                            self.log(f"{name}逆时针转动{angle}度")
                        self.voice_command_queue.put(('set', index, new_angle))
                        return True
                    elif '顺时针' in text:
                        if index == 0:  # 底座：顺时针命令执行逆时针逻辑
                            new_angle = max(self.servo_ranges[index][0], current_angle - angle)
                            self.log(f"{name}顺时针转动{angle}度")
                        else:
                            new_angle = min(self.servo_ranges[index][1], current_angle + angle)
                            self.log(f"{name}顺时针转动{angle}度")
                        self.voice_command_queue.put(('set', index, new_angle))
                        return True
                    elif '左' in text or '往左' in text:
                        if index == 0:  # 底座：左转命令执行右转逻辑
                            new_angle = min(self.servo_ranges[index][1], current_angle + angle)
                            self.log(f"{name}左转{angle}度")
                        else:
                            new_angle = max(self.servo_ranges[index][0], current_angle - angle)
                            self.log(f"{name}左转{angle}度")
                        self.voice_command_queue.put(('set', index, new_angle))
                        return True
                    elif '右' in text or '往右' in text:
                        if index == 0:  # 底座：右转命令执行左转逻辑
                            new_angle = max(self.servo_ranges[index][0], current_angle - angle)
                            self.log(f"{name}右转{angle}度")
                        else:
                            new_angle = min(self.servo_ranges[index][1], current_angle + angle)
                            self.log(f"{name}右转{angle}度")
                        self.voice_command_queue.put(('set', index, new_angle))
                        return True

                    elif '上' in text or '抬高' in text or '升起' in text:
                        new_angle = min(self.servo_ranges[index][1], current_angle + angle)
                        self.voice_command_queue.put(('set', index, new_angle))
                        self.log(f"{name}向上{angle}度")
                        return True

                    elif '下' in text or '降低' in text or '下降' in text:
                        new_angle = max(self.servo_ranges[index][0], current_angle - angle)
                        self.voice_command_queue.put(('set', index, new_angle))
                        self.log(f"{name}向下{angle}度")
                        return True

                    elif '张开' in text or '打开' in text or '松开' in text:
                        if index == 3:  # 钳子：张开命令执行闭合逻辑
                            new_angle = max(self.servo_ranges[index][0], current_angle - angle)
                            self.log(f"{name}张开{angle}度")
                        else:
                            new_angle = min(self.servo_ranges[index][1], current_angle + angle)
                            self.log(f"{name}张开{angle}度")
                        self.voice_command_queue.put(('set', index, new_angle))
                        return True

                    elif '闭合' in text or '关闭' in text or '夹紧' in text or '抓紧' in text:
                        if index == 3:  # 钳子：闭合命令执行张开逻辑
                            new_angle = min(self.servo_ranges[index][1], current_angle + angle)
                            self.log(f"{name}闭合{angle}度")
                        else:
                            new_angle = max(self.servo_ranges[index][0], current_angle - angle)
                            self.log(f"{name}闭合{angle}度")
                        self.voice_command_queue.put(('set', index, new_angle))
                        return True

                    else:
                        # 如果没有明确的方向，但指定了角度，默认增加角度
                        new_angle = min(self.servo_ranges[index][1], current_angle + angle)
                        self.voice_command_queue.put(('set', index, new_angle))
                        self.log(f"{name}增加{angle}度")
                        return True
                else:
                    # 没有指定角度，检查是否有模糊表述
                    print(f"⚠️  未检测到明确数字")

                    # 检查是否有模糊的角度指示
                    if '一点' in text or '一些' in text or '稍微' in text:
                        angle = 5
                        print(f"  使用默认小角度: {angle}度")
                    elif '很多' in text or '大幅' in text or '大幅度' in text:
                        angle = 30
                        print(f"  使用默认大角度: {angle}度")
                    elif '半圈' in text or '一半' in text:
                        angle = 90
                        print(f"  使用半圈角度: {angle}度")
                    else:
                        angle = 10  # 默认转动10度
                        print(f"  使用默认角度: {angle}度")

                    current_angle = self.servo_angles[index]

                    # 优先处理顺逆时针命令，忽略左右命令
                    if '逆时针' in text:
                        new_angle = max(self.servo_ranges[index][0], current_angle - angle)
                        self.voice_command_queue.put(('set', index, new_angle))
                        self.log(f"{name}逆时针转动{angle}度")
                        return True
                    elif '顺时针' in text:
                        new_angle = min(self.servo_ranges[index][1], current_angle + angle)
                        self.voice_command_queue.put(('set', index, new_angle))
                        self.log(f"{name}顺时针转动{angle}度")
                        return True
                    elif '左' in text or '往左' in text:
                        new_angle = max(self.servo_ranges[index][0], current_angle - angle)
                        self.voice_command_queue.put(('set', index, new_angle))
                        self.log(f"{name}左转{angle}度")
                        return True
                    elif '右' in text or '往右' in text:
                        new_angle = min(self.servo_ranges[index][1], current_angle + angle)
                        self.voice_command_queue.put(('set', index, new_angle))
                        self.log(f"{name}右转{angle}度")
                        return True

                    elif '上' in text or '抬高' in text or '升起' in text:
                        new_angle = min(self.servo_ranges[index][1], current_angle + angle)
                        self.voice_command_queue.put(('set', index, new_angle))
                        self.log(f"{name}向上{angle}度")
                        return True

                    elif '下' in text or '降低' in text or '下降' in text:
                        new_angle = max(self.servo_ranges[index][0], current_angle - angle)
                        self.voice_command_queue.put(('set', index, new_angle))
                        self.log(f"{name}向下{angle}度")
                        return True

                    elif '张开' in text or '打开' in text or '松开' in text:
                        new_angle = min(self.servo_ranges[index][1], current_angle + angle)
                        self.voice_command_queue.put(('set', index, new_angle))
                        self.log(f"{name}张开{angle}度")
                        return True

                    elif '闭合' in text or '关闭' in text or '夹紧' in text or '抓紧' in text:
                        new_angle = max(self.servo_ranges[index][0], current_angle - angle)
                        self.voice_command_queue.put(('set', index, new_angle))
                        self.log(f"{name}闭合{angle}度")
                        return True

                    else:
                        # 如果只有舵机名没有方向，默认右转10度
                        new_angle = min(self.servo_ranges[index][1], current_angle + angle)
                        self.voice_command_queue.put(('set', index, new_angle))
                        self.log(f"{name}转动{angle}度")
                        return True

        # 常规匹配：完整舵机名称
        for i, name in enumerate(self.servo_names):
            if name in text:
                print(f"✅ 找到舵机: {name}")

                # 提取角度 - 使用类方法
                angle = self.extract_angle_from_text(text)

                if angle is not None:
                    print(f"✅ 提取到角度: {angle}度")

                    # 处理 "到XX度" 命令
                    if '到' in text or '转到' in text or '设置为' in text or '位置' in text:
                        min_angle, max_angle = self.servo_ranges_dict[name]
                        if min_angle <= angle <= max_angle:
                            self.voice_command_queue.put(('set', i, angle))
                            self.log(f"{name}已转到{angle}度")
                            return True
                        else:
                            self.log(f"{name}角度超出范围")
                            return True

                    # 处理方向命令
                    current_angle = self.servo_angles[i]

                    # 优先处理顺逆时针命令，忽略左右命令
                    if '逆时针' in text:
                        if i == 0:  # 底座：逆时针命令执行顺时针逻辑
                            new_angle = min(self.servo_ranges[i][1], current_angle + angle)
                            self.log(f"{name}逆时针转动{angle}度")
                        else:
                            new_angle = max(self.servo_ranges[i][0], current_angle - angle)
                            self.log(f"{name}逆时针转动{angle}度")
                        self.voice_command_queue.put(('set', i, new_angle))
                        return True
                    elif '顺时针' in text:
                        if i == 0:  # 底座：顺时针命令执行逆时针逻辑
                            new_angle = max(self.servo_ranges[i][0], current_angle - angle)
                            self.log(f"{name}顺时针转动{angle}度")
                        else:
                            new_angle = min(self.servo_ranges[i][1], current_angle + angle)
                            self.log(f"{name}顺时针转动{angle}度")
                        self.voice_command_queue.put(('set', i, new_angle))
                        return True
                    elif '左' in text or '往左' in text:
                        if i == 0:  # 底座：左转命令执行右转逻辑
                            new_angle = min(self.servo_ranges[i][1], current_angle + angle)
                            self.log(f"{name}左转{angle}度")
                        else:
                            new_angle = max(self.servo_ranges[i][0], current_angle - angle)
                            self.log(f"{name}左转{angle}度")
                        self.voice_command_queue.put(('set', i, new_angle))
                        return True
                    elif '右' in text or '往右' in text:
                        if i == 0:  # 底座：右转命令执行左转逻辑
                            new_angle = max(self.servo_ranges[i][0], current_angle - angle)
                            self.log(f"{name}右转{angle}度")
                        else:
                            new_angle = min(self.servo_ranges[i][1], current_angle + angle)
                            self.log(f"{name}右转{angle}度")
                        self.voice_command_queue.put(('set', i, new_angle))
                        return True

                    elif '上' in text or '抬高' in text or '升起' in text:
                        new_angle = min(self.servo_ranges[i][1], current_angle + angle)
                        self.voice_command_queue.put(('set', i, new_angle))
                        self.log(f"{name}向上{angle}度")
                        return True

                    elif '下' in text or '降低' in text or '下降' in text:
                        new_angle = max(self.servo_ranges[i][0], current_angle - angle)
                        self.voice_command_queue.put(('set', i, new_angle))
                        self.log(f"{name}向下{angle}度")
                        return True

                    elif '张开' in text or '打开' in text or '松开' in text:
                        if i == 3:  # 钳子：张开命令执行闭合逻辑
                            new_angle = max(self.servo_ranges[i][0], current_angle - angle)
                            self.log(f"{name}张开{angle}度")
                        else:
                            new_angle = min(self.servo_ranges[i][1], current_angle + angle)
                            self.log(f"{name}张开{angle}度")
                        self.voice_command_queue.put(('set', i, new_angle))
                        return True

                    elif '闭合' in text or '关闭' in text or '夹紧' in text or '抓紧' in text:
                        if i == 3:  # 钳子：闭合命令执行张开逻辑
                            new_angle = min(self.servo_ranges[i][1], current_angle + angle)
                            self.log(f"{name}闭合{angle}度")
                        else:
                            new_angle = max(self.servo_ranges[i][0], current_angle - angle)
                            self.log(f"{name}闭合{angle}度")
                        self.voice_command_queue.put(('set', i, new_angle))
                        return True

                    else:
                        # 如果没有明确的方向，但指定了角度，默认增加角度
                        new_angle = min(self.servo_ranges[i][1], current_angle + angle)
                        self.voice_command_queue.put(('set', i, new_angle))
                        self.log(f"{name}增加{angle}度")
                        return True
                else:
                    # 没有指定角度，检查是否有模糊表述
                    print(f"⚠️  未检测到明确数字")

                    # 检查是否有模糊的角度指示
                    if '一点' in text or '一些' in text or '稍微' in text:
                        angle = 5
                        print(f"  使用默认小角度: {angle}度")
                    elif '很多' in text or '大幅' in text or '大幅度' in text:
                        angle = 30
                        print(f"  使用默认大角度: {angle}度")
                    elif '半圈' in text or '一半' in text:
                        angle = 90
                        print(f"  使用半圈角度: {angle}度")
                    else:
                        angle = 10  # 默认转动10度
                        print(f"  使用默认角度: {angle}度")

                    current_angle = self.servo_angles[i]

                    # 优先处理顺逆时针命令，忽略左右命令
                    if '逆时针' in text:
                        if i == 0:  # 底座：逆时针命令执行顺时针逻辑
                            new_angle = min(self.servo_ranges[i][1], current_angle + angle)
                            self.log(f"{name}逆时针转动{angle}度")
                        else:
                            new_angle = max(self.servo_ranges[i][0], current_angle - angle)
                            self.log(f"{name}逆时针转动{angle}度")
                        self.voice_command_queue.put(('set', i, new_angle))
                        return True
                    elif '顺时针' in text:
                        if i == 0:  # 底座：顺时针命令执行逆时针逻辑
                            new_angle = max(self.servo_ranges[i][0], current_angle - angle)
                            self.log(f"{name}顺时针转动{angle}度")
                        else:
                            new_angle = min(self.servo_ranges[i][1], current_angle + angle)
                            self.log(f"{name}顺时针转动{angle}度")
                        self.voice_command_queue.put(('set', i, new_angle))
                        return True
                    elif '左' in text or '往左' in text:
                        if i == 0:  # 底座：左转命令执行右转逻辑
                            new_angle = min(self.servo_ranges[i][1], current_angle + angle)
                            self.log(f"{name}左转{angle}度")
                        else:
                            new_angle = max(self.servo_ranges[i][0], current_angle - angle)
                            self.log(f"{name}左转{angle}度")
                        self.voice_command_queue.put(('set', i, new_angle))
                        return True
                    elif '右' in text or '往右' in text:
                        if i == 0:  # 底座：右转命令执行左转逻辑
                            new_angle = max(self.servo_ranges[i][0], current_angle - angle)
                            self.log(f"{name}右转{angle}度")
                        else:
                            new_angle = min(self.servo_ranges[i][1], current_angle + angle)
                            self.log(f"{name}右转{angle}度")
                        self.voice_command_queue.put(('set', i, new_angle))
                        return True

                    elif '上' in text or '抬高' in text or '升起' in text:
                        new_angle = min(self.servo_ranges[i][1], current_angle + angle)
                        self.voice_command_queue.put(('set', i, new_angle))
                        self.log(f"{name}向上{angle}度")
                        return True

                    elif '下' in text or '降低' in text or '下降' in text:
                        new_angle = max(self.servo_ranges[i][0], current_angle - angle)
                        self.voice_command_queue.put(('set', i, new_angle))
                        self.log(f"{name}向下{angle}度")
                        return True

                    elif '张开' in text or '打开' in text or '松开' in text:
                        if i == 3:  # 钳子：张开命令执行闭合逻辑
                            new_angle = max(self.servo_ranges[i][0], current_angle - angle)
                            self.log(f"{name}张开{angle}度")
                        else:
                            new_angle = min(self.servo_ranges[i][1], current_angle + angle)
                            self.log(f"{name}张开{angle}度")
                        self.voice_command_queue.put(('set', i, new_angle))
                        return True

                    elif '闭合' in text or '关闭' in text or '夹紧' in text or '抓紧' in text:
                        if i == 3:  # 钳子：闭合命令执行张开逻辑
                            new_angle = min(self.servo_ranges[i][1], current_angle + angle)
                            self.log(f"{name}闭合{angle}度")
                        else:
                            new_angle = max(self.servo_ranges[i][0], current_angle - angle)
                            self.log(f"{name}闭合{angle}度")
                        self.voice_command_queue.put(('set', i, new_angle))
                        return True

                    else:
                        # 如果只有舵机名没有方向，默认右转10度
                        new_angle = min(self.servo_ranges[i][1], current_angle + angle)
                        self.voice_command_queue.put(('set', i, new_angle))
                        self.log(f"{name}转动{angle}度")
                        return True

        print(f"❌ 未识别指令: '{original_text}'")
        self.log("指令未识别，请重试")
        return False

    def process_voice_commands(self):
        """处理语音命令队列"""
        try:
            while not self.voice_command_queue.empty():
                command = self.voice_command_queue.get_nowait()
                if command[0] == 'set':
                    _, servo_idx, angle = command
                    self.servo_angles[servo_idx] = angle
                    print(f"设置舵机 {servo_idx} 到 {angle}度")
                    self.send_to_arduino()
                elif command[0] == 'set_all':
                    _, angles = command
                    for i in range(4):
                        self.servo_angles[i] = angles[i]
                    print("设置所有舵机到90度")
                    self.send_to_arduino()
        except queue.Empty:
            pass

    def find_arduino_port(self):
        """自动查找Arduino端口"""
        ports = serial.tools.list_ports.comports()
        arduino_patterns = ['arduino', 'ch340', 'ch341', 'usb serial', 'usb-serial']

        for port in ports:
            port_info = port.description.lower()
            for pattern in arduino_patterns:
                if pattern in port_info:
                    print(f"找到Arduino设备: {port.device}")
                    return port.device

        print("⚠️  未自动找到Arduino，请手动指定端口")
        return None



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
            # 左手：食指(1)控制底座，拇指(0)控制小臂
            control_finger_1 = 1  # 食指
            control_finger_2 = 0  # 拇指
        else:
            # 右手：拇指(0)控制大臂，食指(1)控制钳子
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

        # 左手控制：食指(底座)和拇指(小臂)
        if left_hand_info:
            if not left_hand_info['finger1_extended']:  # 食指弯曲
                self.servo_directions[0] = left_hand_info['rotation_direction']  # 底座
            if not left_hand_info['finger2_extended']:  # 拇指弯曲
                self.servo_directions[1] = left_hand_info['rotation_direction']  # 小臂

        # 右手控制：拇指(大臂)和食指(钳子)
        if right_hand_info:
            if not right_hand_info['finger1_extended']:  # 拇指弯曲
                self.servo_directions[2] = right_hand_info['rotation_direction']  # 大臂
            if not right_hand_info['finger2_extended']:  # 食指弯曲
                self.servo_directions[3] = right_hand_info['rotation_direction']  # 钳子

    def update_servo_angles(self):
        """根据方向更新舵机角度"""
        step = 2  # 每次调整的步长

        for i in range(4):
            min_angle, max_angle = self.servo_ranges[i]

            if self.servo_directions[i] == 1:  # 顺时针
                self.servo_angles[i] = min(max_angle, self.servo_angles[i] + step)
            elif self.servo_directions[i] == -1:  # 逆时针
                self.servo_angles[i] = max(min_angle, self.servo_angles[i] - step)
            # 为0时不改变角度

    def send_to_arduino(self):
        """发送舵机角度到Arduino"""
        if not self.arduino:
            return

        # 检查角度是否有变化
        angles_changed = False
        for i in range(4):
            if abs(self.servo_angles[i] - self.last_sent_angles[i]) > 1:
                angles_changed = True
                break

        if not angles_changed:
            return

        # 构建数据包：角度1,角度2,角度3,角度4
        data = f"{self.servo_angles[0]},{self.servo_angles[1]},{self.servo_angles[2]},{self.servo_angles[3]}\n"

        try:
            self.arduino.write(data.encode())
            self.last_sent_angles = self.servo_angles.copy()
            print(f"📤 发送到Arduino: {data.strip()}")
        except Exception as e:
            print(f"❌ 发送数据失败: {e}")

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
        draw.text((10, 15), "多模态机械臂控制系统", font=font, fill=(0, 255, 255))

        # 绘制舵机角度信息
        servo_text = f"舵机角度: 底座={self.servo_angles[0]} 小臂={self.servo_angles[1]} 大臂={self.servo_angles[2]} 钳子={self.servo_angles[3]}"
        draw.text((10, 55), servo_text, font=font, fill=(255, 255, 0))

        # 绘制转动方向
        direction_text = f"转动方向: {self.servo_directions}"
        draw.text((10, 85), direction_text, font=font, fill=(255, 255, 0))

        # 绘制手势控制说明
        instructions_y = 115
        instructions = [
            "左手控制:",
            "  食指弯曲 -> 底座转动",
            "  拇指弯曲 -> 小臂转动",
            "右手控制:",
            "  拇指弯曲 -> 大臂转动",
            "  食指弯曲 -> 钳子转动",
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
        recognition_mode = '离线识别 (Vosk)'
        
        voice_status = f"语音模式: {voice_mode} | 运动状态: {motion_status}"
        recognition_mode_text = f"识别模式: {recognition_mode}"
        
        draw.text((w - 400, 15), voice_status, font=font, fill=(0, 255, 0))
        draw.text((w - 400, 45), recognition_mode_text, font=font, fill=(0, 255, 255))

        # 绘制优化状态
        optimizations = [
            f"卡尔曼滤波: {'启用' if self.enable_kalman_filter else '禁用'}",
            f"直方均衡化: {'启用' if self.enable_histogram_equalization else '禁用'}"
        ]

        for i, opt in enumerate(optimizations):
            draw.text((w - 200, 85 + i * 25), opt, font=font, fill=(100, 255, 100))

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
        print("     - 左手食指控制底座，左手拇指控制小臂")
        print("     - 右手拇指控制大臂，右手食指控制钳子")
        print("     - 中+无+小指弯曲:顺时针，伸直:逆时针")
        print("  2. 语音控制:")
        print("     - 说'开启语音'/'关闭语音' 控制语音模式")
        print("     - 说'底座左转30度'/'钳子张开20度'等")
        print("     - 说'归位'让所有舵机回到中间位置")
        print("     - 说'暂停'/'继续' 控制运动")
        print("  3. 键盘快捷键:")
        print("     - Q: 退出")
        print("     - R: 复位舵机")
        print("     - T: 测试语音")
        print("     - V: 切换语音模式")
        print("     - K: 切换卡尔曼滤波")
        print("     - H: 切换直方均衡化")
        print("     - S: 模拟语音指令")
        print("=" * 50)

        # 欢迎语音
        self.log("多模态机械臂控制系统已启动")

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

                    # 发送到Arduino
                    self.send_to_arduino()

                # 绘制控制信息
                frame = self.draw_control_info(frame, left_hand_info, right_hand_info)

                # 显示画面
                cv2.imshow('多模态机械臂控制', frame)

                # 键盘控制
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):  # 退出
                    break
                elif key == ord('r'):  # 复位
                    self.servo_angles = [90, 90, 90, 90]
                    self.send_to_arduino()
                    print("舵机已复位")
                    self.log("舵机已复位")
                elif key == ord('t'):  # 测试语音
                    self.log("语音测试正常")
                elif key == ord('v'):  # 切换语音模式
                    self.voice_enabled = not self.voice_enabled
                    status = "开启" if self.voice_enabled else "关闭"
                    print(f"语音模式已{status}")
                    self.log(f"语音模式已{status}")
                elif key == ord('k'):  # 切换卡尔曼滤波
                    self.enable_kalman_filter = not self.enable_kalman_filter
                    status = "开启" if self.enable_kalman_filter else "关闭"
                    print(f"卡尔曼滤波已{status}")
                elif key == ord('h'):  # 切换直方均衡化
                    self.enable_histogram_equalization = not self.enable_histogram_equalization
                    status = "开启" if self.enable_histogram_equalization else "关闭"
                    print(f"直方均衡化已{status}")
                elif key == ord('s'):  # 模拟语音指令
                    # 模拟语音指令用于测试
                    test_commands = [
                        "开启语音",
                        "底座左转30度",
                        "前臂向上20度",
                        "钳子张开15度",
                        "归位"
                    ]
                    import random
                    test_cmd = random.choice(test_commands)
                    print(f"模拟语音指令: {test_cmd}")
                    self.parse_voice_command(test_cmd)

        finally:
            # 清理资源
            cap.release()
            cv2.destroyAllWindows()
            if self.arduino:
                self.arduino.close()
                print("串口已关闭")
            if self.audio_stream:
                self.audio_stream.stop_stream()
                self.audio_stream.close()
            if self.pyaudio_instance:
                self.pyaudio_instance.terminate()

            self.log("系统已关闭")


if __name__ == "__main__":
    try:
        # 创建控制器实例
        controller = MultiModalArmController(serial_port='COM5', baud_rate=9600)

        # 运行主程序
        controller.run()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"程序运行错误: {e}")
        import traceback

        traceback.print_exc()