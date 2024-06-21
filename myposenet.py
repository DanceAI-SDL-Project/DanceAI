#!/usr/bin/env python3
#
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

import sys
import argparse
import os

from jetson_inference import poseNet
from jetson_utils import videoSource, videoOutput, Log

# parse the command line
parser = argparse.ArgumentParser(description="Run pose estimation DNN on a video/image stream.", 
                                 formatter_class=argparse.RawTextHelpFormatter, 
                                 epilog=poseNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
parser.add_argument("--network", type=str, default="resnet18-body", help="pre-trained model to load (see below for options)")
parser.add_argument("--overlay", type=str, default="links,keypoints", help="pose overlay flags (e.g. --overlay=links,keypoints)\nvalid combinations are:  'links', 'keypoints', 'boxes', 'none'")
parser.add_argument("--threshold", type=float, default=0.15, help="minimum detection threshold to use") 

try:
	args = parser.parse_known_args()[0]
except:
	print("")
	parser.print_help()
	sys.exit(0)

# load the pose estimation model
net = poseNet(args.network, sys.argv, args.threshold)

# create video sources & outputs
input = videoSource(args.input, argv=sys.argv)
output = videoOutput(args.output, argv=sys.argv)

# process frames until EOS or the user exits
while True:
    # capture the next image
    img = input.Capture()

    if img is None: # timeout
        continue  

    # perform pose estimation (with overlay)
    poses = net.Process(img, overlay=args.overlay)

    # print the pose results
    # print("detected {:d} objects in image".format(len(poses)))



    os.system("clear")
    for i, pose in enumerate(poses):
        left_hip = 0
        right_hip = 0
        left_knee = 0
        right_knee = 0
        nose = 0
        print ("\nPersona #" + str(i))

        ids = [keypoint.ID for keypoint in pose.Keypoints]
        no_todas = False
        for i in (0, 11, 12, 13, 14):
            if i not in ids:
                no_todas = True
                break

        if no_todas:
            print("NO ESTAN TODAS LAS PARTES\n\r")
            continue

        for keypoint in pose.Keypoints:
            if keypoint.ID == 0:
                nose = keypoint.y
            if keypoint.ID == 11:
                left_hip = keypoint.y
            elif keypoint.ID == 12:
                right_hip = keypoint.y
            elif keypoint.ID == 13:
                left_knee = keypoint.y
            elif keypoint.ID == 14:
                right_knee = keypoint.y
        
        dif_izq = abs(left_hip-left_knee)
        dif_der = abs(right_hip-right_knee)

        dif_naz_izq = abs(nose - left_hip)
        dif_naz_der = abs(nose - right_hip)

        ratio_izq = dif_naz_izq / dif_izq
        ratio_der = dif_naz_der / dif_der

        MAX_RATIO = 2.5
        
        if ratio_izq > MAX_RATIO and ratio_der > MAX_RATIO:
            print("ESTAS SENTADO!")
        else:
            print("ESTAS EN PIE!")
        
        print(ratio_izq)
        print(str(ratio_der) + "\n\r")

    # render the image
    output.Render(img)

    # update the title bar
    output.SetStatus("{:s} | Network {:.0f} FPS".format(args.network, net.GetNetworkFPS()))

    # print out performance info
    net.PrintProfilerTimes()

    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break
