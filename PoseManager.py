from Pose import Pose

class PoseManager:
    ref_array = []

    def __init__(self):
        pass

    def compare(self, pose1, pose2):
        pose1.compare(pose2)

    def set_ref(self, ref_array):
        self.reference = ref_array

    def calculate_score(self, ref_pose: Pose, pose_to_test: Pose): 
        score = 0

        for i, ref_point in enumerate(ref_pose.get_points()):
            ref_x, ref_y = ref_pose.get_relative_point(ref_point.ID)
            p_x, p_y = pose_to_test.get_relative_point(ref_point.ID)
            
            score += 100 * (1 - abs(ref_x - p_x) + 1 - ref_y - p_y)
        return score / len(ref_pose.get_points())

    def get_score(self, pose_array):
        if len(self.ref_array) != len(pose_array):
            raise RuntimeError("Reference array is not same length as pose array")

        score = 0 

        for i, ref_pose in enumerate(self.ref_array):
            pose_to_test = pose_array[i]

            score += self.calculate_score(ref_pose, pose_to_test)
    



            
         
