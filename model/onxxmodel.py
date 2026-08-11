import cv2
import numpy as np

CLASS_NAMES = (
    "grabbing","grip","holy","point","call","three3","timeout",
    "xsign","hand_heart","hand_heart2","little_finger","middle_finger",
    "take_picture","dislike","fist","four","like","mute","ok","one",
    "palm","peace","peace_inverted","rock","stop","stop_inverted",
    "three","three2","two_up","two_up_inverted","three_gun",
    "thumb_index","thumb_index2","no_gesture"
)

net = cv2.dnn.readNet("model/YOLOv10n_gestures.onnx")

cap = cv2.VideoCapture(1)

while True:

    ret, frame = cap.read()

    if not ret:
        break

    blob = cv2.dnn.blobFromImage(
        frame,
        scalefactor=1/255,
        size=(640,640),
        swapRB=True,
        crop=False
    )

    net.setInput(blob)

    output = net.forward()

    print("Output shape:", output.shape)

    preds = output.reshape(-1, output.shape[-1])

    for pred in preds:

        cx,cy,w,h = pred[:4]

        scores = pred[4:]

        cls = np.argmax(scores)
        conf = scores[cls]

        if conf < 0.35:
            continue

        print(CLASS_NAMES[cls], conf)

    cv2.imshow("ONNX Test",frame)

    if cv2.waitKey(1)==27:
        break

cap.release()
cv2.destroyAllWindows()