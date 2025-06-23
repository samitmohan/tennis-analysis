from ultralytics import YOLO

# xyxy: tensor([[1853.7853,   30.9651, 1892.3993,   69.0948]]) {xmin, ymin, xmax, ymax}
model = YOLO("models/last.pt")
res = model.predict("input/input_video.mp4", conf=0.2, save=True)
res = model.track("input/input_video.mp4", conf=0.2, save=True)
print(res)
print("Boxes")
for box in res[0].boxes:
    print(box)
