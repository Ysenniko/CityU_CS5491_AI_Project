from ultralytics import YOLO
 
model = YOLO(model="yolov8s-dcn-cbam.yaml")


 
data = "ultralytics/cfg/datasets/VOC.yaml"
 
model.train(data=data, epochs=100, batch=16)
