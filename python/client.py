# MICRO-453 TP-13
# Visual Navigation: A Deep Learning Perspective
# VITA, EPFL 

import cv2
import socket
import sys
import numpy
import struct
import binascii
import argparse

from PIL import Image
#from detector import Detector
from final_detector import Detector
#import kalman
from kalman import KalmanFilter

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)


parser.add_argument('-i','--ip-address',
                    help='IP Address of robot')
parser.add_argument('--instance-threshold', default=0.2, type=float,
                    help='Defines the threshold of the detection score')
parser.add_argument('-d', '--downscale', default=3, type=int,
                    help=('downscale of the received image'))


args = parser.parse_args()

##### IP Address of server #########
host = args.ip_address #'128.179.150.43'  # The server's hostname or IP address


thresh = args.instance_threshold
downscale = args.downscale


##### IP Address of server #########
#host = '128.179.198.173'  # The server's hostname or IP address
####################################
port = 8081        # The port used by the server

# image data

width = int(640/downscale)
height = int(480/downscale)
channels = 3
sz_image = width*height*channels

# create socket
print('# Creating socket')
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
except socket.error:
    print('Failed to create socket')
    sys.exit()

print('# Getting remote IP address') 
try:
    remote_ip = socket.gethostbyname( host )
except socket.gaierror:
    print('Hostname could not be resolved. Exiting')
    sys.exit()

# Connect to remote server
print('# Connecting to server, ' + host + ' (' + remote_ip + ')')
s.connect((remote_ip , port))
s.setblocking(1)

# Set up detector
detector = Detector()

#Image Receiver 
net_recvd_length = 0
recvd_image = b''

#Test Controller
direction = -1
cnt = 0

packed_data = b''

nb_byte_to_receive = int(sz_image / 8)

# init kalman filter 
kalman = KalmanFilter()
err_in_row = 0
width_bbox = 60 
height_bbox = 60

while True:
   
    #######################
    # Receive data
    #######################
   
    while net_recvd_length < sz_image:
        reply = s.recv(nb_byte_to_receive)
        recvd_image += reply
        net_recvd_length += len(reply)

    pil_image = Image.frombytes('RGB', (width, height), recvd_image)
    opencvImage = cv2.cvtColor(numpy.array(pil_image), cv2.COLOR_RGB2BGR)
    opencvImage = cv2.cvtColor(opencvImage,cv2.COLOR_BGR2RGB)

    pil_image = pil_image.resize((224, 224), Image.ANTIALIAS)
    opencvImage = cv2.cvtColor(numpy.array(pil_image), cv2.COLOR_RGB2BGR)
    
    cv2.imshow('Test window',opencvImage)
    cv2.waitKey(1)

    net_recvd_length = 0
    recvd_image = b''

    #######################
    # Detect
    #######################

    
    bbox, bbox_label = detector.forward(pil_image)
    print(bbox,bbox_label)
    if bbox_label:
        print(bbox)
    else:
        print("False")

    #######################
    # Correction kalman 
    #######################
    prediction = kalman.Estimate(bbox[0],bbox[1])

    detected = 1

    if err_in_row >= 5:
        detected = 0

    if float(bbox_label) == 1 : 
        err_in_row = 0
        width_bbox = bbox[2]/224*width
        height_bbox = bbox[2]/224*width
        values = (prediction[0][0]/224*width, prediction[1][0]/224*height, width_bbox ,height_bbox, detected )
    else: 
        err_in_row += 1
        values = (prediction[0][0]/224*width, prediction[1][0]/224*height, width_bbox ,height_bbox, detected )

    #values = (100, 100,50, 50, True)
    #Test Controller
    # cnt = cnt + 1
    # if cnt > 50:
    #     direction = - direction
    #     cnt = 0
    # values = (40.0 + direction * 20.0, 30.0, 10.0, 20.0, 1.0)
    
    packer = struct.Struct('f f f f f')
    packed_data = packer.pack(*values)

    # Send data
    send_info = s.send(packed_data)
