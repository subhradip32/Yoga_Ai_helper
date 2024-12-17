# Emotion Detection with Suggested Yoga Poses

This project combines **emotion detection** using a deep learning model and suggests **yoga poses** for corresponding emotional states (Happy, Neutral, Sad). The application uses **OpenCV** for live webcam feed, **TensorFlow/Keras** for predictions, and **Tkinter** for the user interface.
Watch this : https://youtu.be/vGgx6OPDq-4

## Table of Contents

1. [Overview](#overview)
2. [Approach](#approach)
3. [Project Structure](#project-structure)
4. [Data Preprocessing](#data-preprocessing)
5. [Model Architecture](#model-architecture)
6. [Application Workflow](#application-workflow)
7. [Results](#results)
8. [Next Steps](#next-steps)
9. [Dependencies](#dependencies)
10. [How to Run](#how-to-run)
11. [Sample Output](#sample-output)

---

## Overview

The application uses a **pre-trained CNN model** to classify emotions into three categories: **Happy, Neutral, and Sad**. For each emotion, it suggests a list of corresponding yoga poses. The live webcam feed allows users to capture their image and predict their emotional state.

---

## Approach

1. **Data Acquisition**: A dataset of facial expressions was used to train a CNN model.
2. **Model Training**: A custom Convolutional Neural Network (CNN) was created and trained for multi-class classification.
3. **Emotion Mapping**: A predefined list of yoga poses is mapped to each emotion.
4. **Application Development**:
   - **Webcam Feed**: Real-time video stream using OpenCV.
   - **Emotion Detection**: The captured frame is passed through the CNN model for prediction.
   - **Pose Display**: Suggested yoga poses are displayed in the application window.

---

## Project Structure

The project files are organized as follows:

```plaintext
Emotion_Yoga_App/
│
├── Model_creation/
│   └── best_model.h5          # Pre-trained CNN model
├── main.py                    # Main application code
├── README.md                  # Documentation file
└── requirements.txt           # List of dependencies
```

---

## Data Preprocessing

The image input undergoes the following preprocessing steps before being fed into the model:

1. **Resize**: Each frame is resized to 128x128 pixels.
2. **Normalization**: Pixel values are normalized to the range [0, 1].
3. **Batch Dimension**: The image is expanded to add a batch dimension for compatibility with the model.

Code snippet:

```python
def preprocess(self, img_data):
    img_data = cv.resize(img_data, (128, 128))
    img_data = img_data / 255.0
    img_data = np.expand_dims(img_data, axis=0)
    return img_data
```

---

## Model Architecture

The CNN model was trained with the following structure:

- **Input Layer**: Image size (128x128x3)
- **Convolutional Layers**: Extract features using filters with ReLU activation.
- **Pooling Layers**: Reduce spatial dimensions using max-pooling.
- **Dense Layers**: Fully connected layers to learn high-level features.
- **Output Layer**: A softmax activation function for multi-class classification.

The model predicts one of the following classes:

- **0**: Happy
- **1**: Neutral
- **2**: Sad

The model file `best_model.h5` is loaded for inference.

---

## Application Workflow

1. **Start Application**:
   - Open a live webcam feed.
2. **Live Display**:
   - The webcam feed is continuously displayed in a Tkinter window.
3. **Capture Frame**:
   - The user clicks the **"Capture Frame"** button.
   - The captured image is processed and passed to the CNN model.
4. **Emotion Detection**:
   - The predicted emotion is displayed.
5. **Yoga Pose Suggestions**:
   - A list of yoga poses corresponding to the detected emotion is displayed in the UI.

---

## Results

The application successfully classifies emotions and suggests appropriate yoga poses. The output includes:

- Predicted Emotion: *Happy*, *Neutral*, or *Sad*.
- Suggested Yoga Poses: A list of poses tailored to the emotional state.

For example:

**Predicted Emotion**: *Sad*  
**Poses**:
- Child’s Pose
- Seated Head-to-Knee Forward Bend
- Pigeon Pose
- Low Lunge
- Standing Forward Fold

---

## Next Steps

- **Enhance Model**: Train the model on a larger and more diverse dataset for improved accuracy.
- **Add More Emotions**: Include more emotional states like Anger, Fear, and Surprise.
- **Pose Visualization**: Display images or animations for each yoga pose.
- **Mobile Application**: Develop a mobile-friendly version of the app.

---

## Dependencies

Ensure the following dependencies are installed. They are listed in the `requirements.txt` file:

```plaintext
tensorflow
numpy
opencv-python
Pillow
tkinter
```

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/emotion-yoga-app.git
   cd emotion-yoga-app
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the application:
   ```bash
   python main.py
   ```

4. The live webcam feed will appear. Capture a frame and see the suggested yoga poses!

---

## Sample Output

### Live Webcam Feed:
A real-time video stream is displayed in the application.

### Captured Frame:
- Emotion: **Happy**
- Suggested Poses:
  - Warrior Pose II
  - Cobra Pose
  - Tree Pose
  - Child’s Pose with Arms Stretched
  - Wild Thing

---

## Screenshots

| **Live Webcam** | **Detected Emotion** |
|-----------------|----------------------|
| ![Webcam](placeholder.png) | ![Result](placeholder.png) |

---

## Contributors

- **Your Name** (Developer and Documenter)

---

## License

This project is licensed under the MIT License.

---

Feel free to add screenshots and replace placeholders with your actual GitHub link and visuals.
