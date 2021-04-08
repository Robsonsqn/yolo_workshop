import cv2
import numpy as np
import time

net = cv2.dnn.readNet("/home/robson/Documents/YOLO_WORKSHOP/yolov3.weights", "/home/robson/Documents/YOLO_WORKSHOP/darknet/cfg/yolov3.cfg")
classes = []

with open("/home/robson/Documents/YOLO_WORKSHOP/darknet/data/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()

output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

colors = np.random.uniform(0, 255, size=(len(classes), 3))
# cap = cv2.VideoCapture('/home/robson/Pictures/carro.mp4')

cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_PLAIN

startin_time = time.time()

frame_id = 0

while(cap.isOpened()):
    ret, frame = cap.read()
    frame_id += 1

    frame = cv2.resize(frame, None, fx=0.8, fy=0.8)

    height, width, channels = frame.shape

    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)

    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)

            confidence = scores[class_id]

            if confidence > 0.8:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.8, 0.3)

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]

            label = str(classes[class_ids[i]])

            confidence = confidences[i]

            color = colors[class_ids[i]]

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

            cv2.putText(frame, label + " " + str(round(confidence, 2)), (x, y + 30), font, 3, color, 3)
    
    elapse_time = time.time() - startin_time

    fps = frame_id / elapse_time

    cv2.putText(frame, "FPS: "+ str(round(fps, 2)), (10, 50), font, 4, (0, 0, 0), 3)

    cv2.imshow("image ", frame)

    key = cv2.waitKey(1)
    if key == 27:
        break

    if cv2.waitKey(25) & 0xff == ord('q'):
        break

cap.release()

cv2.destroyAllWindows()
    
