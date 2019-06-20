# LoomoTracking
Custom robotic tracking system using a deep learning approach. Semester project supervised by Yuejiang Liu and Prof. Alexandre Alahi at the visual intelligence for transportation lab of EPFL.

## Most Important files

### client.py

The client is the base of the communication between the robot and the local computer. The communication is based on TCP socket. This file depends on the files final_detector.py and kalman.py . It receives a image from the Loomo robot, detects the object of interest, compute a correction of the position based on a kalman filter, using the position in the past of the object and sends back the position of detected object to the robot. 

### kalman.py

This file implements a simple kalman filter using the OpenCV library.

### detector.py 

The detector itself is implemented in this file. It is implemented as class. When a first detector object is created, the init function loads the pre-trained weights obtained after training. The forward function allows to do a prediction based on the image given as argument. A threshold can also be chosen in order to tune the sensitivity of the detector. 
