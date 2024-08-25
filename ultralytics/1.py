from ultralytics import YOLO


model = YOLO('/repository7403/wangyanjie/project/ultralytics-ma/ultralytics/cfg/models/v8/yolov82.yaml').load('./yolov8s.pt')  # build from YAML and transfer weights


results = model.train(data='/repository7403/wangyanjie/project/ultralytics/ultralytics/cfg/cell.yaml', epochs=100, imgsz=640,lr0=1e-3,device=2,batch=32)