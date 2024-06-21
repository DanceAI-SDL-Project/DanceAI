
import sys
import argparse
import os
import json
import jetson.utils
import numpy as np
from jetson_inference import poseNet

# Create a videoSource object with the device name
video_source = jetson.utils.videoSource("/dev/video0")
output = jetson.utils.videoOutput("display://0")
font = jetson.utils.cudaFont()

# Funcion  para leer y parsear el archivo JSON
def read_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Ruta al archivo JSON
json_file = 'puntos_data.json'


# Leer y parsear el archivo JSON
circles_data = read_json(json_file)



# Loop to read frames from the video source
while True:
    # Capture frame-by-frame
    frame = video_source.Capture()

    #Overlay Points
    #font.OverlayText(frame, frame.width, frame.height, "TEXTO", 150, 150, font.White, font.Black)
    # Display the resulting frame
    #jetson.utils.cudaDrawCircle(frame, (1000, 1000), 1000, (255,0,0))
	color = (255, 0, 0, 255)  # Color rojo con opacidad completa
	radius = 5  # Raadio fijo para los circulos
	for timestamp, personas in data.items():
    		for id_persona, partes_del_cuerpo in personas.items():
       			 for id_parte_del_cuerpo, (pos_x, pos_y) in partes_del_cuerpo.items():
            			jetson.utils.cudaDrawCircle(image, (pos_x, pos_y, radius, color))

    # Show the frame
    output.Render(frame)

    # Exit loop if 'q' is pressed
    if video_source.IsStreaming() == False:
        break

# Clean up
video_source.Close()
