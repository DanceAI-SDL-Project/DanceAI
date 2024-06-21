from jetson_inference import poseNet

class Pose:
    relative_points=list()

    def __init__(self, pose):
        # Store pose object
        self.pose=pose

    def check_point(self, point):
        if point in self.pose.Keypoints:
            return True
        else:
            return False

    def get_point_name(self, point):
        if not self.check_point(point): return -1
        return poseNet.GetKeypointName(point.ID)

    def get_points(self):
        return self.pose.Keypoints
    

    def compare(self, pose):
        for p1 in pose.Keypoints:
            for p2 in self.pose.Keypoints:
                if p1.x == p2.x and p1.y == p2.y:
                    found = True
                    break
            if not found: return False
        return True

    def get_point(self, identifier):
        for p in self.pose.Keypoints:
            if p.ID == identifier:
                return p
        return -1

    def get_relative_x(self, point):
        if not self.check_point(point): return -1
        anchura = abs(self.pose.Left - self.pose.Right)
        if anchura == 0: return 1
        relativeX = point.x - self.pose.Left
        ratioX = relativeX / anchura
        return ratioX

    def get_relative_y(self, point):
        if not self.check_point(point): return -1
        altura = abs(self.pose.Top - self.pose.Bottom)
        if altura == 0: return 1
        relativeY = point.y - self.pose.Top
        ratioY = relativeY / altura
        return ratioY
    
    def get_absolute_x(self, ratioX):
        anchura = abs(self.pose.Left - self.pose.Right)
        relativeX = ratioX * anchura
        absoluteX = relativeX + self.pose.Left
        return absoluteX

    def get_absolute_y(self, ratioY):
        altura = abs(self.pose.Top - self.pose.Bottom)
        relativeY = ratioY * altura
        absoluteY = relativeY + self.pose.Top
        return absoluteY
    
    def get_relative_point(self, p):
        # p = self.get_point(identifier)
        return (self.get_relative_x(p), self.get_relative_y(p))
    
    def get_absolute_point(self, relative_coords):
        relativeX = relative_coords[0]
        relativeY = relative_coords[1]
        return (self.get_absolute_x(relativeX), self.get_absolute_y(relativeY))
    
    def get_links(self):
        return self.pose.Links
        
    def get_relative_points(self):
        if len(self.relative_points) == 0:
            for p in self.pose.Keypoints:
                self.relative_points.append(self.get_relative_point(p))
        return self.relative_points