import cv2 as cv

# 模型路径
model_bin = "MobileNetSSD_deploy.caffemodel"
config_text = "MobileNetSSD_deploy.prototxt.txt"
# 类别信息
objName = ["background",
"aeroplane", "bicycle", "bird", "boat",
"bottle", "bus", "car", "cat", "chair",
"cow", "diningtable", "dog", "horse",
"motorbike", "person", "pottedplant",
"sheep", "sofa", "train", "tvmonitor"]

# 加载模型
net = cv.dnn.readNetFromCaffe(config_text, model_bin)

# 获得所有层名称与索引
layerNames = net.getLayerNames()
lastLayerId = net.getLayerId(layerNames[-1])
lastLayer = net.getLayer(lastLayerId)
print(lastLayer.type)


# 打开摄像头
cap = cv.VideoCapture('v3.mp4')
while True:
    ret, frame = cap.read()
    if ret is False:
        print("读取结束！")
        break
    h, w = frame.shape[:2]
    blobImage = cv.dnn.blobFromImage(cv.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blobImage)
    cvOut = net.forward()
    numperson = 0
    for detection in cvOut[0, 0, :, :]:
        score = float(detection[2])
        objIndex = int(detection[1])
        if score > 0.5:
            left = detection[3]*w
            top = detection[4]*h
            right = detection[5]*w
            bottom = detection[6]*h
            # 绘制
            if objIndex == 15:
                numperson = numperson + 1
                cv.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), thickness=2)
                cv.putText(frame, "person", (int(left), int(top)),
                            cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    showtext = "Traffic Density:" + str(numperson)
    cv.putText(frame, showtext, (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    # 显示
    cv.imshow('video', frame)
    cv.waitKey(20)
    if cv.waitKey(1) & 0xFF == ord("q"):
        break
cap.release()
cv.destroyAllWindows()
