from tensorflow import keras
import cv2
import numpy as np
import mediapipe as mp
from imutils.object_detection import non_max_suppression

# Load Model
model = keras.models.load_model('Best_CNN_model.h5')
# print(model.layers)
classes = ["Burglary", "Broke", "Normal", "Assault"]

mpPose = mp.solutions.pose
pose = mpPose.Pose()
mpDraw = mp.solutions.drawing_utils

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# cap = cv2.VideoCapture(1)
cap = cv2.VideoCapture('Fight/0.mp4')

while cap.isOpened():
    success, image = cap.read()

    if not success:
        print('Not success')
        break

    orig = image.copy()
    height, width, color = image.shape

    if image.shape[0] > 1080 and image.shape[1] > 1920:
        image = cv2.resize(image, dsize=(1920, 1080), interpolation=cv2.INTER_AREA)

    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)

    # draw the original bounding boxes
    for (x, y, w, h) in rects:
        cv2.rectangle(orig, (x, y), (x + w, y + h), (0, 0, 255), 2)

    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])

    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    # People index
    index = 0

    for (xA, yA, xB, yB) in pick:
        # print("pick")
        resize_size = 20  # rectangles in pick would be added this.
        irs = 150  # Ignore Rectangle Size

        # If the resized rectangle size go out frame
        resize_xA = (xA - resize_size) if xA - resize_size > 0 else 0
        resize_yA = (yA - resize_size) if yA - resize_size > 0 else 0
        resize_xB = (xB + resize_size) if xB + resize_size < width else width
        resize_yB = (yB + resize_size) if yB + resize_size < height else height

        # Ignore rectangle
        if resize_xB - resize_xA < irs or resize_yB - resize_yA < irs:
            # print("Ignore")
            continue

        # Draw Green Rectangle in origin image
        cv2.rectangle(image, (resize_xA, resize_yA), (resize_xB, resize_yB), (0, 255, 0), 2)
        new_img = image[resize_yA:resize_yB, resize_xA:resize_xB].copy()

        # Mediapipe Processing
        imgRGB = cv2.cvtColor(new_img, cv2.COLOR_BGR2RGB)
        results = pose.process(imgRGB)

        # Empty black Matrix
        Empty_black = new_img.copy()
        # shape[0] : height, shape[1] : width
        for i in range(Empty_black.shape[1]):
            for j in range(Empty_black.shape[0]):
                Empty_black[j, i] = (0, 0, 0)

        # Draw Pose point and connections
        if results.pose_landmarks:
            mpDraw.draw_landmarks(Empty_black, results.pose_landmarks, mpPose.POSE_CONNECTIONS)

            # TODO Draw Skeleton in image
            # 1) Black Background
            # image[resize_yA:resize_yB, resize_xA:resize_xB] = Empty_black
            # 2) Transparent Background
            mpDraw.draw_landmarks(image[resize_yA:resize_yB, resize_xA:resize_xB], results.pose_landmarks, mpPose.POSE_CONNECTIONS)

            # Convert RGB -> GRAY
            Empty_black = cv2.cvtColor(Empty_black, cv2.COLOR_RGB2GRAY)

            # Resize image for CNN
            Empty_black = cv2.resize(Empty_black, dsize=(80, 80), interpolation=cv2.INTER_LINEAR)

            # scaled data
            Scaled_data = Empty_black.reshape(1, 80, 80, 1) / 255.0

            # Predict
            predict = model.predict(Scaled_data)
            print(classes[np.argmax(predict)])

            # Text on window
            cv2.putText(image, classes[np.argmax(predict)], (xA, yA), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)

        index += 1

        # Show image through window
    cv2.imshow("After NMS", image)

    # Wait
    if cv2.waitKey(1) & 0xFF == 27:
        break