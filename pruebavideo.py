import sys
import argparse
import os

from jetson_inference import poseNet
import jetson.utils

# Create a videoSource object with the device name
video_source = jetson.utils.videoSource("/dev/video0")
output = jetson.utils.videoOutput("display://0")
font = jetson.utils.cudaFont()

# Loop to read frames from the video source
while True:
    # Capture frame-by-frame
    frame = video_source.Capture()

    #Overlay Text
    font.OverlayText(frame, frame.width, frame.height, "TEXTO", 150, 150, font.White, font.Black)
    # Display the resulting frame
    #jetson.utils.cudaDrawCircle(frame, (1000, 1000), 1000, (255,0,0))

    # Show the frame
    output.Render(frame)

    # Exit loop if 'q' is pressed
    if video_source.IsStreaming() == False:
        break

# Clean up
video_source.Close()
