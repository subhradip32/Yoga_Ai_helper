import cv2 as cv
from tkinter import *
from PIL import Image, ImageTk
import tensorflow as tf

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
    
    def capture_frame(self):
        print(self.last_frame)
        if self.last_frame is not None:
            self.is_live = False
            img = ImageTk.PhotoImage(image=Image.fromarray(self.last_frame))
            camera_label.imgtk = img
            camera_label.configure(image=img)
            print("Live feed stopped, showing the captured frame.")
        else:
            print("No frame available to capture!")

class Model(): 
    def __init__(self, model_path: str):
        self.model = tf.keras.models.load_model(model_path)
    
    def preprocess(self, img_data): 
        
    def predict(self,image_data): 
        data = preprocess(image_data)
        return self.model.predict(data) 
    


# Tkinter setup
main = Tk()
main.title("Webcam Feed")


camera_label = Label(main)
camera_label.pack()

cam = Cam()
cam.display_image(camera_label)

capture_button = Button(main, text="Capture Frame", command= cam.capture_frame )
capture_button.pack()


main.mainloop()

# Release resources
cam.cap.release()
cv.destroyAllWindows()
