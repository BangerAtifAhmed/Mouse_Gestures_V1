# Run this in Colab or a separate environment where you can install ultralytics
# pip install ultralytics tqdm

from ultralytics import YOLO
from tqdm.auto import tqdm
import threading
import time

# Load your custom PyTorch model
print("Loading model...")
model = YOLO("YOLOv10x_gestures.pt")
print("✅ Model loaded")

# Function to perform export
def export_model():
    global export_result
    export_result = model.export(format="onnx")

# Start export in a separate thread
thread = threading.Thread(target=export_model)
thread.start()

# Display an animated progress bar while exporting
with tqdm(total=100, desc="Exporting to ONNX", ncols=100) as pbar:
    while thread.is_alive():
        if pbar.n < 95:
            pbar.update(1)
        time.sleep(0.2)

    # Finish the progress bar
    pbar.n = 100
    pbar.refresh()

thread.join()

print("\n✅ Export completed successfully!")
print("📁 Generated file: YOLOv10n_gestures.onnx")
print("➡️ Move 'YOLOv10n_gestures.onnx' into your clean capstone folder.")