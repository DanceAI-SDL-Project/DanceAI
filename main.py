import sys
import argparse
import os


from PoseManager import PoseManager
from Pose import Pose
from jetson_inference import poseNet
from jetson_utils import videoSource, videoOutput, Log
from jetson.utils import cudaDrawCircle, cudaDrawLine

import sys
import select
import tty
import termios

def print_lines(pose):
    for link in pose.get_links():
        p1 = pose.get_point(link[0])
        p2 = pose.get_point(link[1])
        cudaDrawLine(img, (p1.x, p1.y), (p2.x, p2.y), (255, 0, 0, 255), 3)

def isData():
    return select.select([sys.stdin], [], [], 0) == ([sys.stdin], [], [])

if __name__ == "__main__" :
    RECORDING = False
    PLAYING = False
    # parse the command line
    parser = argparse.ArgumentParser(description="Run pose estimation DNN on a video/image stream.", 
                                    formatter_class=argparse.RawTextHelpFormatter, 
                                    epilog=poseNet.Usage() + videoSource.Usage() + videoOutput.Usage() + Log.Usage())

    parser.add_argument("input", type=str, default="", nargs='?', help="URI of the input stream")
    parser.add_argument("output", type=str, default="", nargs='?', help="URI of the output stream")
    parser.add_argument("--network", type=str, default="resnet18-body", help="pre-trained model to load (see below for options)")
    parser.add_argument("--overlay", type=str, default="links,keypoints", help="pose overlay flags (e.g. --overlay=links,keypoints)\nvalid combinations are:  'links', 'keypoints', 'boxes', 'none'")
    parser.add_argument("--threshold", type=float, default=0.15, help="minimum detection threshold to use") 

    aux_ref_array = []
    cached_shadows = []
    cached = False
    score = 0

    try:
        args = parser.parse_known_args()[0]
    except:
        print("")
        parser.print_help()
        sys.exit(0)

    output = videoOutput("")
    net = poseNet(args.network, sys.argv, args.threshold)
    input = videoSource("/dev/video0")
    pose_man = PoseManager()

    tty.setcbreak(sys.stdin.fileno())
    while True:
        img = input.Capture()

        if img is None:
            continue

        poses = net.Process(img, overlay=args.overlay)
        if len(poses) == 0:
            output.Render(img)
            continue
        
        recording_pose = Pose(poses[0])


        if isData():
            c = sys.stdin.read(1)

            if (c == "r"):
                if (not RECORDING):
                    pose_man.ref_array.clear()
                RECORDING = not RECORDING
            if (c == "p"):
                if not RECORDING:
                    PLAYING = not PLAYING
                    if PLAYING:
                        aux_ref_array = pose_man.ref_array.copy()
                        score = 0
        
        if RECORDING:
            pose_man.ref_array.append(recording_pose)
            print("RECORDING")

        if PLAYING:

            ref_pose = aux_ref_array.pop(0)

            # for rel_p in ref_pose.get_relative_points():
            #     print_lines(ref_pose)

            for point in ref_pose.get_points():
                rel_point = ref_pose.get_relative_point(point)
                point = recording_pose.get_absolute_point(rel_point)
                cudaDrawCircle(img, (point[0], point[1]), 5, (255, 0, 0, 255))
            
            for link in ref_pose.get_links():
                if (len(link) == 1):
                    continue

                p1 = ref_pose.pose.Keypoints[link[0]]
                p2 = ref_pose.pose.Keypoints[link[1]]

                rel_point = ref_pose.get_relative_point(p1)
                p1 = recording_pose.get_absolute_point(rel_point)

                rel_point = ref_pose.get_relative_point(p2)
                p2 = recording_pose.get_absolute_point(rel_point)
                cudaDrawLine(img, (p1[0], p1[1]), (p2[0], p2[1]), (255, 0, 0, 255), 3)
            

            
            shadow_num = 15
            diffusing = 1.2
            
            shadows = [i for i in range(1, shadow_num, 2)]
            for i, shadow in enumerate(shadows):
                if len(aux_ref_array) > shadow:
                    shadow_pose = aux_ref_array[shadow]
                    red_color = 255/diffusing**(i/3)

                    for point in shadow_pose.get_points():
                        rel_point = shadow_pose.get_relative_point(p1)
                        point = recording_pose.get_absolute_point(rel_point)
                        cudaDrawCircle(img, (point[0], point[1]), 5, (0, 0, 0, 255/diffusing**(i)))
                    
                    for link in shadow_pose.get_links():
                        if (len(link) == 1):
                            continue

                        p1 = shadow_pose.pose.Keypoints[link[0]]
                        p2 = shadow_pose.pose.Keypoints[link[1]]

                        rel_point = shadow_pose.get_relative_point(p1)
                        p1 = recording_pose.get_absolute_point(rel_point)

                        rel_point = shadow_pose.get_relative_point(p2)
                        p2 = recording_pose.get_absolute_point(rel_point)
                        cudaDrawLine(img, (p1[0], p1[1]), (p2[0], p2[1]), (0, 0, 0, 255/diffusing**(i)), 3)
                        
            
            score += pose_man.calculate_score(ref_pose, recording_pose)
            print(f"SCORE: {score}")

            if len(aux_ref_array) == 0: 
                PLAYING = False 


        output.Render(img)
