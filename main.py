# import cv2 as cv
# from tkinter import *
# from PIL import Image, ImageTk
# import tensorflow as tf
# import numpy as np  

# class Model(): 
#     def __init__(self, model_path: str):
#         self.model = tf.keras.models.load_model(model_path)
#         self.classes = {
#             0:"Happy", 1:"Neutral", 2:"Sad"
#         }
    
#     def preprocess(self, img_data):
#         img_data = cv.resize(img_data, (128, 128))
#         img_data = img_data / 255.0
#         img_data = np.expand_dims(img_data, axis=0)
#         return img_data
        
#     def predict(self, image_data): 
#         data = self.preprocess(image_data)  
#         return self.model.predict(data)   
    
# class Cam:
#     def __init__(self):
#         self.cap = cv.VideoCapture(0)
#         self.is_live = True
#         self.last_frame = None
    
#     def read_cam(self):
#         ret, frame = self.cap.read()
#         if ret:
#             frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
#             self.last_frame = frame
#             return frame
#         return None

#     def display_image(self, camera_label):
#         if not self.is_live: 
#             return 
#         frame = self.read_cam()
#         if frame is not None:
#             img = ImageTk.PhotoImage(image=Image.fromarray(frame))
#             camera_label.imgtk = img
#             camera_label.configure(image=img)
#         camera_label.after(10, self.display_image, camera_label)
    
#     def capture_frame(self, model: Model):
#         if self.last_frame is not None:
#             self.is_live = False  # Stop the live feed
#             prediction = model.predict(self.last_frame)
#             # print("Live feed stopped. Showing the captured frame.")
#             # print("Model Prediction:", prediction, model.classes[np.argmax(prediction)] )
#             display(model.classes[np.argmax(prediction)])
#             img = ImageTk.PhotoImage(image=Image.fromarray(self.last_frame))
#             camera_label.imgtk = img
#             camera_label.configure(image=img)
#         else:
#             print("No frame available to capture!")

# def display(): 
#     pass 



# # Tkinter setup
# main = Tk()
# main.title("Webcam Feed")

# camera_label = Label(main)
# camera_label.pack()

# # Initialize camera
# cam = Cam()
# cam.display_image(camera_label)

# # Load the model
# m = Model("Model_creation/best_model.h5")

# # Add a capture button
# capture_button = Button(main, text="Capture Frame", command=lambda: cam.capture_frame(m))
# capture_button.pack()

# main.mainloop()

# # Release resources
# cam.cap.release()
# cv.destroyAllWindows()


import cv2 as cv
from tkinter import *
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np  

class Model(): 
    def __init__(self, model_path: str):
        self.model = tf.keras.models.load_model(model_path)
        self.classes = {
            0: "Happy", 
            1: "Neutral", 
            2: "Sad"
        }
        # Yoga poses for each emotion
        self.poses = {
            "Happy": [
                "Warrior Pose II", "Cobra Pose", "Tree Pose", "Child’s Pose with Arms Stretched", "Wild Thing"
            ],
            "Neutral": [
                "Mountain Pose", "Seated Forward Bend", "Corpse Pose", "Cat-Cow Pose", "Staff Pose"
            ],
            "Sad": [
                "Child’s Pose", "Seated Head-to-Knee Forward Bend", "Pigeon Pose", "Low Lunge", "Standing Forward Fold"
            ]
        }
    
    def preprocess(self, img_data):
        img_data = cv.resize(img_data, (128, 128))
        img_data = img_data / 255.0
        img_data = np.expand_dims(img_data, axis=0)
        return img_data
        
    def predict(self, image_data): 
        data = self.preprocess(image_data)  
        return self.model.predict(data)   
    
class Cam:
    def __init__(self):
        self.cap = cv.VideoCapture(0)
        self.is_live = True
        self.last_frame = None
    
    def read_cam(self):
        ret, frame = self.cap.read()
        if ret:
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            self.last_frame = frame
            return frame
        return None

    def display_image(self, camera_label):
        if not self.is_live: 
            return 
        frame = self.read_cam()
        if frame is not None:
            img = ImageTk.PhotoImage(image=Image.fromarray(frame))
            camera_label.imgtk = img
            camera_label.configure(image=img)
        camera_label.after(10, self.display_image, camera_label)
    
    def capture_frame(self, model: Model):
        if self.last_frame is not None:
            self.is_live = False  # Stop the live feed
            prediction = model.predict(self.last_frame)
            predicted_class = model.classes[np.argmax(prediction)]
            # Display the relevant poses
            display(predicted_class, model)
            img = ImageTk.PhotoImage(image=Image.fromarray(self.last_frame))
            camera_label.imgtk = img
            camera_label.configure(image=img)
        else:
            print("No frame available to capture!")

def display(predicted_class, model): 
    poses = model.poses[predicted_class]
    # Update the label with the poses based on the predicted class
    pose_text = "\n".join(poses)  # Join the poses in a readable format
    pose_label.config(text=f"Predicted Emotion: {predicted_class}\n\nPoses:\n{pose_text}")

# Tkinter setup
main = Tk()
main.title("Webcam Feed")

camera_label = Label(main)
camera_label.pack()

# Pose display label
pose_label = Label(main, font=("Helvetica", 12))
pose_label.pack()

# Initialize camera
cam = Cam()
cam.display_image(camera_label)

# Load the model
m = Model("Model_creation/best_model.h5")

# Add a capture button
capture_button = Button(main, text="Capture Frame", command=lambda: cam.capture_frame(m))
capture_button.pack()

main.mainloop()

# Release resources
cam.cap.release()
cv.destroyAllWindows()
