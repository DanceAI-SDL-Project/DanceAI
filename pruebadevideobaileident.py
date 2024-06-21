
import sys
import argparse
import os
import json
import jetson.utils
import numpy as np
from jetson_inference import poseNet
import time

# Create a videoSource object with the device name
video_source = jetson.utils.videoSource("/dev/video0")
output = jetson.utils.videoOutput("display://0")
font = jetson.utils.cudaFont()

def circle_data_iterator(data):
    for timestamp, personas in data.items():
        yield timestamp, personas

# Funcion  para leer y parsear el archivo JSON
def read_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Ruta al archivo JSON
json_file = 'puntos_data.json'


# Leer y parsear el archivo JSON
circles_data = read_json(json_file)
#last_draw_time = time.time()
color = (255, 0, 0, 255)
radius = 5
draw_interval = 1/1
circle_iterator = circle_data_iterator(circles_data)

while True:
    # Capture frame-by-frame

    frame = video_source.Capture()
 #   current_time = time.time()
    if current_time - last_draw_time >= draw_interval:
        try:
            timestamp, personas = next(circle_iterator)
            for id_persona, partes_del_cuerpo in personas.items():
                for id_parte_del_cuerpo, (pos_x, pos_y) in partes_del_cuerpo.items():
                   # jetson.utils.cudaDeviceSynchronize()
			output.Render(frame)
                   # try:
                   #     del frame
                   # except:
                   #     pass
                    frame = video_source.Capture()
                    jetson.utils.cudaDrawCircle(frame, (pos_x, pos_y), radius, color)

        except StopIteration:
            # Reset the iterator if it reaches the end
            circle_iterator = circle_data_iterator(circles_data)
        
#        last_draw_time = current_time
#    output.Render(frame)


    if video_source.IsStreaming() == False:
        break

video_source.Close()
