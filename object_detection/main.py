import cv2

#OpenCv DNN
net = cv2.dnn.readNet("dnn_model\yolov4-tiny.weights", "dnn_model\yolov4-tiny.cfg")
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(320,320),scale=1/255)
cap = cv2.VideoCapture(0)

#Load class lists
classes = []
with open("dnn_model\classes.txt", "r") as file_object:
    for class_name in file_object.readlines():
        class_name = class_name.strip()
        classes.append(class_name)

print("Objects List", classes)
#Initialize camera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

while True:

#Get Frames
    ret, frame = cap.read()

    #Object Detection
    (class_ids, scores, bboxes) = model.detect(frame)
    for class_id, score, bbox in zip(class_ids, scores, bboxes):
        x,y,w,h = bbox
        # print(x,y,w,h)
        class_name = classes[class_id]
        print(class_name)
        cv2.putText(frame, class_name, (x,y -10), cv2.FONT_HERSHEY_PLAIN, 2, (200,0,50), 2 )
        cv2.rectangle(frame, (x,y),(x +w, y+h), (200,0,50), 3)
    # print("class ids", class_ids)
    # print("scores", scores)
    # print("bboxes", bboxes)    
    cv2.imshow("Frame", frame)
    cv2.waitKey(1) 