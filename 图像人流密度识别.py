import numpy as np
import argparse
import cv2
import os

input_doc = "data"
output_doc = "getdata"
pictures = os.listdir(input_doc)
density = []
file = "Density.txt"
for i in range(len(pictures)):
	# 设置路径
	outputdoc = os.path.join(output_doc+'/'+pictures[i])
	inputdoc = os.path.join(input_doc+'/'+pictures[i])
	ap = argparse.ArgumentParser()
	ap.add_argument("-i", "--image", required=False, default=inputdoc,
		help="path to input image")
	ap.add_argument("-c", "--confidence", type=float, default=0.2,
		help="minimum probability to filter weak detections")
	args = vars(ap.parse_args())
	# 初始化经过培训的MobileNet SSD可以检测的类别标签列表，然后为每个类别生成一组边界框颜色
	CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
		"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
		"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
		"sofa", "train", "tvmonitor"]
	COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))
	# 加载模型
	net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")
	# 加载输入图像并通过将图像尺寸调整为固定的300x300像素然后对其进行归一化来构造图像的输入Blob
	image_path = args["image"]
	image = cv2.imread(image_path)
	(h, w) = image.shape[:2]
	blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843, (300, 300), 127.5)
	# 通过网络传递blob并获得检测和预测
	net.setInput(blob)
	# 不关掉会出现unkonwn error
	cv2.ocl.setUseOpenCL(False)
	detections = net.forward()
	numperson = 0
	# 循环检测
	for i in np.arange(0, detections.shape[2]):
		# 提取与预测相关的置信度（即概率）
		confidence = detections[0, 0, i, 2]
		# 通过确保“置信度”大于最小置信度来滤除弱检测
		if confidence > args["confidence"]:
			# 从“检测”中提取类标签的索引，然后计算对象边界框的（x，y）坐标
			idx = int(detections[0, 0, i, 1])
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
			(startX, startY, endX, endY) = box.astype("int")
			# 显示预测
			label = "{}".format(CLASSES[idx])
			# 找出是人的框
			if idx == 15:
				numperson = numperson + 1
				cv2.rectangle(image, (startX, startY), (endX, endY),
					COLORS[idx], 3)
				y = startY - 15 if startY - 15 > 15 else startY + 15
				cv2.putText(image, label, (startX, y),
					cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS[idx], 3)
	showtext = "Traffic Density:"+str(numperson)
	cv2.putText(image, showtext, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
	cv2.imwrite(outputdoc, image)
	density.append(numperson)
	print(inputdoc, "识别完成！")
	getdoc = inputdoc+"人群密度："+str(numperson)
	# 将人群密度写入txt文件
	with open(file, "a") as f:
		f.seek(0, 2)
		f.write(getdoc)
		f.write('\n')
print('人群密度为:', density)
print('人群密度已经记录在Density文件中，请查看。')
