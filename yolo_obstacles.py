# ------------------------- YOLO 장애물 -------------------------
yolo_model = YOLO("yolov8n.pt")
def detect_obstacles(frame):
    results = yolo_model(frame)[0]
    obstacles=[]
    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
        if int(cls)==2:
            x=int((box[0]+box[2])/2)
            y=int((box[1]+box[3])/2)
            obstacles.append((x,y))
    return obstacles
