import cv2 as cv
from tkinter import *
import mediapipe as mp
from mediapipe.tasks.python import vision
from mediapipe.tasks import python
from PIL import Image, ImageTk

model_path = 'pose_landmarker_full.task'

BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE
)

landmarker = PoseLandmarker.create_from_options(options)

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, model_complexity=1, enable_segmentation=False)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles


def addmedia(frame):
    # Convert the frame to RGB as required by MediaPipe
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark  # Retrieve landmark points
        h, w, _ = frame.shape

        # Draw lines connecting landmarks
        for connection in mp_pose.POSE_CONNECTIONS:
            start_idx, end_idx = connection
            start = landmarks[start_idx]
            end = landmarks[end_idx]
            x1, y1 = int(start.x * w), int(start.y * h)
            x2, y2 = int(end.x * w), int(end.y * h)
            cv.line(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # Draw circles on landmarks
        for lm in landmarks:
            x, y = int(lm.x * w), int(lm.y * h)
            cv.circle(frame, (x, y), 5, (0, 0, 255), -1)

    return cv.flip(frame, 1)


def Physical_Read_frame():
    ret, frame = cap.read()
    if ret:
        small_frame = cv.resize(frame, (320, 240))
        small_frame = cv.cvtColor(small_frame, cv.COLOR_BGR2RGB)
        processed_frame = addmedia(small_frame)
        display_frame = cv.resize(processed_frame, (700, 500))
        img = Image.fromarray(display_frame)
        imgtk = ImageTk.PhotoImage(image=img)
        camera_label.imgtk = imgtk
        camera_label.configure(image=imgtk)
    camera_label.after(1, Physical_Read_frame)

def start_camera():
    global cap
    cap = cv.VideoCapture(0)
    Physical_Read_frame()

def cap_frame():
    global cap, camera_label
    cap = cv.VideoCapture(0)
    
    ret, frame = cap.read()
    if ret: 
        small_frame = cv.resize(frame, (320, 240))
        small_frame = cv.cvtColor(small_frame, cv.COLOR_BGR2RGB)
        img = Image.fromarray(small_frame)
        imgtk = ImageTk.PhotoImage(image=img)
        camera_label.imgtk = imgtk
        camera_label.configure(image=imgtk)


def close_cam():
    global cap
    global camera_label
    print("Cam close")
    camera_label.config(text = "", image = None)  
    if cap.isOpened():
        cap.release()  
    landmarker.close()  


main = Tk()
main.title("Yoga AI Assistant")
main.geometry("700x700")

camera_label = Label(main)
camera_label.pack()

start_button = Button(main, text="Start Camera", command=start_camera)
start_button.pack()
start_button = Button(main, text="End Camera", command=close_cam)
start_button.pack()

main.mainloop()
if 'cap' in globals():
    cap.release()
landmarker.close()
