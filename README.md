# About and Features

Human pose estimation is a computer vision task that involves detecting and estimating the spatial positions of key body joints or keypoints in images or videos. The goal is to understand the body's pose, including the positions of limbs and joints, such as shoulders, elbows, hips, and knees. This technology has applications in various fields, including human-computer interaction, virtual reality, augmented reality, sports analysis, and healthcare. Pose estimation algorithms typically use deep learning techniques and convolutional neural networks to analyze and interpret visual data, enabling machines to understand and respond to human movements.

# Explanation of the code linewise

import cv2 as cv
import matplotlib.pyplot as plt
#import cv2 as cv: This line imports the OpenCV library and aliases it as cv. OpenCV (Open Source Computer Vision Library) is a popular computer vision library with various functionalities.
#import matplotlib.pyplot as plt: This line imports the matplotlib library and aliases its submodule pyplot as plt. matplotlib.pyplot is used for creating visualizations and plots. In this context, it might be used to display the image later on.

net = cv.dnn.readNetFromTensorflow("C:\\Users\\CCCIR-003\\Downloads\\Human_Pose_Estimation-main\\Human_Pose_Estimation-main\\graph_opt.pb")
#This line creates a DNN object (net) using the readNetFromTensorflow function from the OpenCV library. The function loads a pre-trained TensorFlow model specified by the file path provided as an argument. In this case, the path is pointing to a file named "graph_opt.pb" in a specific directory.

inWidth = 368
inHeight = 368
thr = 0.2
#inWidth = 368: This line sets the input width for the neural network to 368 pixels. The model expects input images of this width during the pose estimation process.
#inHeight = 368: Similar to the previous line, this one sets the input height for the neural network to 368 pixels. The model expects square input images, so both the width and height are set to 368 pixels.
#thr = 0.2: This line sets a threshold (thr) to 0.2. This threshold is used during the post-processing step to filter out the detected keypoints that have a confidence score lower than 0.2. This helps in removing less reliable keypoints.


BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }
#The next block of code defines a dictionary called BODY_PARTS, which maps body part names to their corresponding indices.

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]
#Each inner list represents a pair of body parts that should be connected to form the skeleton. For example, ["Neck", "RShoulder"] indicates a line connecting the "Neck" and "Right Shoulder" keypoints.


img = cv.imread("C:\\Users\\CCCIR-003\\Downloads\\Human_Pose_Estimation-main\\Human_Pose_Estimation-main\\pos.png")
#This line uses the cv.imread function to read an image file located at the specified file path ("C:\Users\CCCIR-003\Downloads\Human_Pose_Estimation-main\Human_Pose_Estimation-main\pos.png"). The loaded image is then stored in the variable img.

plt.imshow(img)
#This line uses the imshow function from Matplotlib's pyplot module to display the image stored in the variable img. It visualizes the image directly within the Jupyter notebook or in a separate window if you are running the code outside of a notebook environment.


plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
#Here, the cv.cvtColor function is used to convert the color space of the image. The original image is in the BGR (Blue-Green-Red) color space, which is the default color space used by OpenCV. However, Matplotlib's imshow function expects the RGB (Red-Green-Blue) color space.

def pos_estimation(frame):
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    #Set the input for the neural network
    net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    
    #Forward pass through the network
    out = net.forward()
    out = out[:, :19, :, :]

    #Ensure the number of body parts is consistent with the output shape
    assert(len(BODY_PARTS) <= out.shape[1])

    points = []  
    #List to store detected keypoints

    #Iterate over the detected body parts
    for i in range(len(BODY_PARTS)):
        #Slice heatmap of corresponding body part
        heatMap = out[0, i, :, :]

        #Find the global maximum (pose keypoint) in the heatmap
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]

        #Add a point if its confidence is higher than the threshold
        points.append((int(x), int(y)) if conf > thr else None)

    #Connect keypoints to form the pose skeleton
    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

    #Display the processing time on the frame
    t, _ = net.getPerfProfile()
    freq = cv.getTickFrequency() / 1000
    cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))

    return frame
    #Set Input and Forward Pass:The function sets the input to the neural network using cv.dnn.blobFromImage and performs a forward pass to obtain the output.
    #Detect Keypoints:The function extracts keypoints from the output by finding the maximum value in each heatmap and converting it to image coordinates.
    #Connect Keypoints:The function connects keypoints according to predefined pairs in POSE_PAIRS and draws lines on the frame to represent the pose skeleton.
    #Display Processing Time:The processing time of the neural network inference is calculated and displayed on the frame.

    estimated_img =pos_estimation(img)
    #calling the function

    
plt.imshow(cv.cvtColor(img, cv.COLOR_BGR2RGB))
#displaying the original image again using Matplotlib


#for video you have to add the following code and rest of the code will be same
cap = cv.VideoCapture("C:\\Users\\CCCIR-003\\Downloads\\Human_Pose_Estimation-main\\Human_Pose_Estimation-main\\1-Minute Yoga Practice_ Flexibility with Nathan Briner _ Yoga Anytime.mp4")

cap.set(3,800)
cap.set(4,800)

if not cap.isOpened():
    cap=cv.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open video")
while cv.waitKey(1)<0:
    hasFrame,frame=cap.read()
    if not hasFrame:
        cv.waitKey()
        break
        ......
        ......
        cv.imshow('OpenPose using OpenCV', frame)   


#for realtime video capture through web cam you have to add the following code and rest of the code will be same
cap = cv.VideoCapture(0)

cap.set(cv.CAP_PROP_FPS, 10)

cap.set(3,800)
cap.set(4,800)

if not cap.isOpened():
    cap=cv.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open Webcam")
while cv.waitKey(1)<0:
    hasFrame,frame=cap.read()
    if not hasFrame:
        cv.waitKey()
        break
        ......
        ......
        cv.imshow('OpenPose using OpenCV', frame)   



    
