for pose in poses:
    # find the keypoint index from the list of detected keypoints
    # you can find these keypoint names in the model's JSON file, 
    # or with net.GetKeypointName() / net.GetNumKeypoints()
    left_wrist_idx = pose.FindKeypoint('left_wrist')
    left_shoulder_idx = pose.FindKeypoint('left_shoulder')


    # if the keypoint index is < 0, it means it wasn't found in the image
    if left_wrist_idx < 0 or left_shoulder_idx < 0:
        continue

    left_wrist = pose.Keypoints[left_wrist_idx]
    left_shoulder = pose.Keypoints[left_shoulder_idx]

cudaDrawLine(img, (left_shoulder.x, left_shoulder.y), (left_wrist.x, left_wrist.y), 3, (255, 0, 0))

