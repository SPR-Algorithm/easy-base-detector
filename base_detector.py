#coding=utf-8
import cv2
import numpy as np
import time
import serial
import yaml
import os

try:
    import mvsdk
except ImportError:
    mvsdk = None
    print("Warning: mvsdk not found. Only opencv mode is available.")

class App(object):
    def __init__(self, config):
        super(App, self).__init__()
        self.config = config
        self.pFrameBuffer = 0
        self.quit = False
        
        # --- 读取配置 ---
        self.debug = config.get("debug", True)
        self.mode = config.get("mode", "chassis")
        self.camera_type = config.get("camera", {}).get("type", "mvsdk")
        self.camera_id = config.get("camera", {}).get("id", 0)

        self.serial_enabled = config["serial"]["enabled"]
        self.serial_port = None
        if self.serial_enabled:
            try:
                self.serial_port = serial.Serial(config["serial"]["port"], config["serial"]["baudrate"])
                self.serial_port.flushInput()
                self.serial_port.flushOutput()
            except Exception as e:
                print(f"Serial Error: {e}")
                self.serial_enabled = False

        self.threshold = config["binary"]["default_threshold"]
        self.min_area = config["filter"]["min_area"]
        self.max_area = config["filter"]["max_area"]
        self.offset_threshold = config["chassis_control"]["threshold"]
        self.circularity_threshold = config["circularity"]["default_threshold"]
        
        # 读取偏移量参数
        self.center_offset = config["offset"]["center"]
        self.dart_offset_x = config["offset"].get("dart_x", 0) # 默认为0
        self.dart_offset_y = config["offset"].get("dart_y", 0) # 默认为0

    def setup_ui(self):
        """ 初始化分离的窗口和滑动条 """
        cv2.namedWindow("Monitor", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Monitor", 1280, 480)

        cv2.namedWindow("Settings", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Settings", 400, 400) # 稍微增加高度以容纳更多滑动条

        # 基础参数
        cv2.createTrackbar("Threshold", "Settings", self.threshold, 255, self.setThreshold)
        cv2.createTrackbar("Circularity", "Settings", int(self.circularity_threshold*100), 100, self.setsCircularityThreshold)
        cv2.createTrackbar("Min Area", "Settings", self.min_area, 10000, self.setsMinAreaThreshold)
        cv2.createTrackbar("Max Area", "Settings", self.max_area, 30000, self.setsMaxAreaThreshold)
        
        # 模式特定的参数
        # Chassis Offset: 范围 -100 到 +100 (默认值+100作为中点)
        cv2.createTrackbar("Chassis Offset", "Settings", self.center_offset + 100, 200, self.setsCenterThreshold)
        
        # Dart Offset X: 范围 -100 到 +100
        cv2.createTrackbar("Dart Off X", "Settings", self.dart_offset_x + 100, 200, self.setDartOffsetX)
        # Dart Offset Y: 范围 -100 到 +100
        cv2.createTrackbar("Dart Off Y", "Settings", self.dart_offset_y + 100, 200, self.setDartOffsetY)
        
        print("UI Initialized")

    def check_ui_alive(self):
        try:
            if cv2.getWindowProperty("Settings", cv2.WND_PROP_VISIBLE) < 1:
                self.setup_ui()
        except:
            self.setup_ui()

    def process_frame(self, frame):
        frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
        
        if len(frame.shape) == 3:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray_frame = frame
            if self.debug: 
                frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        _, binary = cv2.threshold(gray_frame, self.threshold, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        filtered_contours = [cnt for cnt in contours if self.min_area <= cv2.contourArea(cnt) <= self.max_area]

        target_found = False
        target_center = (0, 0)
        ellipse = None

        if filtered_contours:
            largest_contour = max(filtered_contours, key=cv2.contourArea)
            area = cv2.contourArea(largest_contour)
            perimeter = cv2.arcLength(largest_contour, True)

            if perimeter > 0:
                circularity = 4 * np.pi * (area / (perimeter ** 2))
                
                if circularity > self.circularity_threshold and len(largest_contour) >= 5:
                    target_found = True
                    ellipse = cv2.fitEllipse(largest_contour)
                    target_center = (int(ellipse[0][0]), int(ellipse[0][1]))

        # 执行业务逻辑
        if target_found:
            self.handle_logic(target_center, 640)

        # 可视化处理 (Debug)
        if self.debug:
            self.check_ui_alive()
            
            if target_found:
                # 画出原始检测到的圆
                cv2.ellipse(frame, ellipse, (255, 0, 0), 2)
                cv2.circle(frame, target_center, 5, (0, 0, 255), -1)
                
                # 如果是 Dart 模式，额外画出经过 offset 修正后的目标点（绿色点），方便调试
                if self.mode == "dart":
                    corrected_x = target_center[0] + self.dart_offset_x
                    corrected_y = target_center[1] + self.dart_offset_y
                    cv2.circle(frame, (corrected_x, corrected_y), 5, (0, 255, 0), -1) # 绿色为修正点
                    cv2.putText(frame, f"Raw:{target_center}", (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
                    cv2.putText(frame, f"Fix:{corrected_x},{corrected_y}", (10, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                else:
                    cv2.putText(frame, f"Pos: {target_center}", (target_center[0] + 10, target_center[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            cv2.drawContours(frame, contours, -1, (0, 255, 0), 1)

            binary_bgr = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)
            cv2.putText(binary_bgr, "Binary View", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Mode: {self.mode.upper()}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            combined_img = np.hstack([frame, binary_bgr])
            cv2.line(combined_img, (640, 0), (640, 480), (255, 255, 0), 2)

            cv2.imshow("Monitor", combined_img)
            
            settings_bg = np.zeros((100, 400, 3), dtype=np.uint8)
            cv2.putText(settings_bg, "Adjust Trackbars", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (200, 200, 200), 2)
            cv2.imshow("Settings", settings_bg)
            
            if (cv2.waitKey(1) & 0xFF) == ord('q'):
                self.quit = True
            
            if cv2.getWindowProperty("Monitor", cv2.WND_PROP_VISIBLE) < 1:
                self.quit = True

    def handle_logic(self, center, img_width):
        x, y = center
        
        # --- Chassis Mode (保持原样) ---
        if self.mode == "chassis":
            image_center_x = img_width // 2 + self.center_offset
            offset_x = x - image_center_x
            
            if self.serial_enabled and self.serial_port and self.serial_port.is_open:
                if abs(offset_x) <= self.offset_threshold:
                    self.serial_port.write(b'2')
                elif offset_x < 0:
                    self.serial_port.write(b'1')
                else:
                    self.serial_port.write(b'3')
            else:
                if self.debug:
                    if abs(offset_x) <= self.offset_threshold:
                        print(f"[Chassis] Center (Offset: {offset_x})")
                    elif offset_x < 0:
                        print(f"[Chassis] Left (Offset: {offset_x})")
                    else:
                        print(f"[Chassis] Right (Offset: {offset_x})")

        # --- Dart Mode (修改后：应用 offset) ---
        elif self.mode == "dart":
            # 计算修正后的坐标
            final_x = x + self.dart_offset_x
            final_y = y + self.dart_offset_y
            
            if self.serial_enabled and self.serial_port and self.serial_port.is_open:
                # 发送修正后的坐标
                data_str = f"{final_x},{final_y}\n"
                self.serial_port.write(data_str.encode('utf-8'))
            else:
                if self.debug:
                    # 打印调试信息：原始 -> 修正
                    print(f"[Dart] Raw:({x},{y}) | Offset:({self.dart_offset_x},{self.dart_offset_y}) -> Final:({final_x},{final_y})")

    def run(self):
        if self.debug:
            self.setup_ui()
        else:
            print("Running in HEADLESS mode (No GUI).")

        try:
            if self.camera_type == "opencv":
                self.run_opencv()
            else:
                self.run_mvsdk()
        except KeyboardInterrupt:
            self.quit = True

    def run_opencv(self):
        print(f"Starting OpenCV Camera (ID: {self.camera_id})...")
        cap = cv2.VideoCapture(self.camera_id)
        if not cap.isOpened(): return
        while not self.quit:
            ret, frame = cap.read()
            if not ret: break
            self.process_frame(frame)
        cap.release()

    def run_mvsdk(self):
        if mvsdk is None: return
        DevList = mvsdk.CameraEnumerateDevice()
        if len(DevList) < 1: return
        DevInfo = DevList[0] 
        
        hCamera = 0
        try:
            hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
        except mvsdk.CameraException: return

        cap = mvsdk.CameraGetCapability(hCamera)
        monoCamera = (cap.sIspCapacity.bMonoSensor != 0)
        
        if monoCamera:
            mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
        else:
            mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)
            
        mvsdk.CameraSetTriggerMode(hCamera, 0)
        mvsdk.CameraSetAeState(hCamera, 0)
        mvsdk.CameraSetExposureTime(hCamera, 30 * 1000)
        mvsdk.CameraPlay(hCamera)
        
        FrameBufferSize = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (1 if monoCamera else 3)
        self.pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)
        
        mvsdk.CameraSetCallbackFunction(hCamera, self.GrabCallback, 0)
        while not self.quit: time.sleep(0.1)
        mvsdk.CameraUnInit(hCamera)
        mvsdk.CameraAlignFree(self.pFrameBuffer)

    @mvsdk.method(mvsdk.CAMERA_SNAP_PROC) if mvsdk else lambda x: x
    def GrabCallback(self, hCamera, pRawData, pFrameHead, pContext):
        FrameHead = pFrameHead[0]
        pFrameBuffer = self.pFrameBuffer
        mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer, FrameHead)
        mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)
        
        frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(pFrameBuffer)
        frame = np.frombuffer(frame_data, dtype=np.uint8)
        frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth, 1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3) )
        self.process_frame(frame)

    # --- 滑动条回调函数 ---
    def setThreshold(self, x):
        self.threshold = x
    def setsCircularityThreshold(self, x):
        self.circularity_threshold = x / 100.0
    def setsMinAreaThreshold(self, x):
        self.min_area = x
    def setsMaxAreaThreshold(self, x):
        self.max_area = x
    
    # Chassis模式的 Offset 回调 (-100 ~ 100)
    def setsCenterThreshold(self, x):
        self.center_offset = x - 100

    # Dart模式的 X Offset 回调 (-100 ~ 100)
    def setDartOffsetX(self, x):
        self.dart_offset_x = x - 100

    # Dart模式的 Y Offset 回调 (-100 ~ 100)
    def setDartOffsetY(self, x):
        self.dart_offset_y = x - 100

def main():
    if not os.path.exists("config.yaml"): return
    with open("config.yaml", "r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    app = None
    try:
        app = App(config)
        app.run()
    except Exception:
        import traceback
        traceback.print_exc()
    finally:
        cv2.destroyAllWindows()
        if app and app.serial_enabled and app.serial_port and app.serial_port.is_open:
            app.serial_port.close()

if __name__ == "__main__":
    main()