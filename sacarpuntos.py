import sys
import argparse
import os
import json

from time import time
from jetson_inference import poseNet
from jetson_utils import videoSource, videoOutput, Log

puntos = {}
file_path = "puntos_data.json"
# parse the command line
parser = argparse.ArgumentParser(description="Run pose estimation DNN on a video/image stream.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=poseNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="links,keypoints", help="pose overlay flags (e.g. --overlay=links,keypoints)\nvalid combinations are:  'links', 'keypoints', 'boxes', 'none'")
parser.add_argument("--threshold", type=float, default=0.15, help="minimum detection threshold to use") 

try:
	args = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# load the pose estimation model
net = poseNet("resnet18-body", 0.15)

# create video sources & outputs
input = videoSource("2024-05-16-121658.webm")

# process frames until EOS or the user exits
i = 0
while True:
    # capture the next image
    try: 
        img = input.Capture()
    except:
        break

    if img is None: # timeout
        continue  

    poses = net.Process(img)
    os.system("clear")
    for num_persona, pose in enumerate(poses):
        for keypoint in pose.Keypoints:
            puntos[time()] = {num_persona : {keypoint.ID: (keypoint.x, keypoint.y)}}
                # {timestamp : {id_persona : {id_parte_del_cuerpo : (pos_x , pos_y)}}}

with open(file_path, "w") as json_file:
    json.dump(puntos, json_file)



