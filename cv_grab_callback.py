#coding=utf-8
import cv2
import numpy as np
import mvsdk
import time
import serial

class App(object):
	def __init__(self):
		super(App, self).__init__()
		self.pFrameBuffer = 0
		self.quit = False
		self.serial_port = serial.Serial('/dev/ttyCH341USB0', 115200)  # 初始化串口通信
		self.serial_port.flushInput()  # 清空输入缓冲区
		self.serial_port.flushOutput()  # 清空输出缓冲区

	def main(self):
		# 枚举相机
		DevList = mvsdk.CameraEnumerateDevice()
		nDev = len(DevList)
		if nDev < 1:
			print("No camera was found!")
			return

		for i, DevInfo in enumerate(DevList):
			print("{}: {} {}".format(i, DevInfo.GetFriendlyName(), DevInfo.GetPortType()))
		i = 0 if nDev == 1 else int(input("Select camera: "))
		DevInfo = DevList[i]
		print(DevInfo)

		# 打开相机
		hCamera = 0
		try:
			hCamera = mvsdk.CameraInit(DevInfo, -1, -1)
		except mvsdk.CameraException as e:
			print("CameraInit Failed({}): {}".format(e.error_code, e.message) )
			return

		# 获取相机特性描述
		cap = mvsdk.CameraGetCapability(hCamera)

		# 判断是黑白相机还是彩色相机
		monoCamera = (cap.sIspCapacity.bMonoSensor != 0)

		# 黑白相机让ISP直接输出MONO数据，而不是扩展成R=G=B的24位灰度
		if monoCamera:
			mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_MONO8)
		else:
			mvsdk.CameraSetIspOutFormat(hCamera, mvsdk.CAMERA_MEDIA_TYPE_BGR8)

		# 相机模式切换成连续采集
		mvsdk.CameraSetTriggerMode(hCamera, 0)

		# 手动曝光，曝光时间30ms
		mvsdk.CameraSetAeState(hCamera, 0)
		mvsdk.CameraSetExposureTime(hCamera, 30 * 1000)

		# 让SDK内部取图线程开始工作
		mvsdk.CameraPlay(hCamera)

		# 计算RGB buffer所需的大小，这里直接按照相机的最大分辨率来分配
		FrameBufferSize = cap.sResolutionRange.iWidthMax * cap.sResolutionRange.iHeightMax * (1 if monoCamera else 3)

		# 分配RGB buffer，用来存放ISP输出的图像
		# 备注：从相机传输到PC端的是RAW数据，在PC端通过软件ISP转为RGB数据（如果是黑白相机就不需要转换格式，但是ISP还有其它处理，所以也需要分配这个buffer）
		self.pFrameBuffer = mvsdk.CameraAlignMalloc(FrameBufferSize, 16)

		# 设置采集回调函数
		self.quit = False
		mvsdk.CameraSetCallbackFunction(hCamera, self.GrabCallback, 0)

		# 等待退出
		while not self.quit:
			time.sleep(0.1)

		# 关闭相机
		mvsdk.CameraUnInit(hCamera)

		# 释放帧缓存
		mvsdk.CameraAlignFree(self.pFrameBuffer)

	@mvsdk.method(mvsdk.CAMERA_SNAP_PROC)
	def GrabCallback(self, hCamera, pRawData, pFrameHead, pContext):
		FrameHead = pFrameHead[0]
		pFrameBuffer = self.pFrameBuffer

		mvsdk.CameraImageProcess(hCamera, pRawData, pFrameBuffer, FrameHead)
		mvsdk.CameraReleaseImageBuffer(hCamera, pRawData)
		
		# 此时图片已经存储在pFrameBuffer中，对于彩色相机pFrameBuffer=RGB数据，黑白相机pFrameBuffer=8位灰度数据
		# 把pFrameBuffer转换成opencv的图像格式以进行后续算法处理
		frame_data = (mvsdk.c_ubyte * FrameHead.uBytes).from_address(pFrameBuffer)
		frame = np.frombuffer(frame_data, dtype=np.uint8)
		frame = frame.reshape((FrameHead.iHeight, FrameHead.iWidth, 1 if FrameHead.uiMediaType == mvsdk.CAMERA_MEDIA_TYPE_MONO8 else 3) )

		frame = cv2.resize(frame, (640,480), interpolation = cv2.INTER_LINEAR)
		# cv2.imshow("Press q to end", frame)
		# if (cv2.waitKey(1) & 0xFF) == ord('q'):
		# 	self.quit = True

		# 转换为灰度图像（如果是彩色相机）
		gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.shape[2] == 3 else frame

        # 以253为参数进行二值化
		_, binary = cv2.threshold(gray_frame, 253, 255, cv2.THRESH_BINARY)

		# 检测轮廓
		contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

		# 找到面积最大的轮廓
		if contours:
			largest_contour = max(contours, key=cv2.contourArea)

			if len(largest_contour) >= 5:
				ellipse = cv2.fitEllipse(largest_contour)
				center = (int(ellipse[0][0]), int(ellipse[0][1]))

				cv2.ellipse(frame, ellipse, (255, 0, 0), 2)
				cv2.circle(frame, center, 5, (0, 0, 255), -1)
				cv2.putText(frame, f"Center: {center}", (center[0] + 10, center[1] - 10),
							cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

				# 计算圆心与图像中心的x坐标偏移量
				image_center_x = frame.shape[1] // 2
				offset_x = center[0] - image_center_x
				print(f"Offset X: {offset_x}")

				# 根据偏移量向串口发送数据
				if abs(offset_x) <= 5:
					self.serial_port.write(b'2')  # 偏移量在区间内，发送0
				elif offset_x < 0:
					self.serial_port.write(b'1')  # 圆心在左边，发送-1
				else:
					self.serial_port.write(b'3')  # 圆心在右边，发送1

		cv2.drawContours(frame, contours, -1, (0, 255, 0), 2)

		frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
		cv2.imshow("Press q to end", binary)
		if (cv2.waitKey(1) & 0xFF) == ord('q'):
			self.quit = True

def main():
	try:
		app = App()
		app.main()
	finally:
		cv2.destroyAllWindows()
		if app.serial_port.is_open:
			app.serial_port.close()

main()
