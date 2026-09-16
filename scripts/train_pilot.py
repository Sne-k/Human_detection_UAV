from ultralytics import YOLO

# Load a lightweight pretrained YOLO model
model = YOLO("yolo26n.pt")

# Pilot training
model.train(
    data="dataset/visdrone_person/data.yaml",

    # Image size
    imgsz=640,

    # Small batch for 6 GB VRAM
    batch=8,

    # Short pilot run
    epochs=3,

    # GPU
    device=0,

    # Data loading
    workers=0,

    # Project results
    project="results/training",
    name="pilot",

    # Save checkpoints
    save=True,

    # Use pretrained weights
    pretrained=True
)