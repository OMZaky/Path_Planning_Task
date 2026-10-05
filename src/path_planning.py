from __future__ import annotations

from typing import List

from src.models import CarPose, Cone, Path2D


class PathPlanning:
    """Student-implemented path planner.

    You are given the car pose and an array of detected cones, each cone with (x, y, color)
    where color is 0 for yellow (right side) and 1 for blue (left side). The goal is to
    generate a sequence of path points that the car should follow.

    Implement ONLY the generatePath function.
    """

    def __init__(self, car_pose: CarPose, cones: List[Cone]):
        self.car_pose = car_pose
        self.cones = cones

    def generatePath(self) -> Path2D:
        import math

        num_points = 20
        step = 0.5
        safe_offset = 1.25  # 2.5m track width
        blend_dist = 2.0   

        yaw = self.car_pose.yaw
        cx, cy = self.car_pose.x, self.car_pose.y
        cos_yaw, sin_yaw = math.cos(-yaw), math.sin(-yaw)

        # transform cones to the car's local frame
        blue_local = []
        yellow_local = []
        for cone in self.cones:
            dx, dy = cone.x - cx, cone.y - cy
            lx = dx * cos_yaw - dy * sin_yaw
            ly = dx * sin_yaw + dy * cos_yaw
            if cone.color == 1:
                blue_local.append((lx, ly))
            else:
                yellow_local.append((lx, ly))

        # interpolate Y given X in local frame
        def interpolate_y(x: float, cones_local: list):
            if not cones_local:
                return None
            if len(cones_local) == 1:
                return cones_local[0][1]
            
            # sort cones by local X
            cones_local = sorted(cones_local, key=lambda c: c[0])
            
            # find the best line segment
            c1, c2 = cones_local[0], cones_local[1]
            for i in range(len(cones_local) - 1):
                if cones_local[i][0] <= x:
                    c1, c2 = cones_local[i], cones_local[i+1]
                else:
                    break
                    
            dx_val = c2[0] - c1[0]
            if abs(dx_val) < 1e-3:
                return c1[1]
                
            slope = (c2[1] - c1[1]) / dx_val
            return c1[1] + slope * (x - c1[0])

        # generate path in local frame
        local_path = []
        for i in range(1, num_points + 1):
            x = i * step
            
            y_blue = interpolate_y(x, blue_local)
            y_yellow = interpolate_y(x, yellow_local)
            
            if y_blue is not None and y_yellow is not None:
                y_center = (y_blue + y_yellow) / 2.0
            elif y_blue is not None:
                y_center = y_blue - safe_offset
            elif y_yellow is not None:
                y_center = y_yellow + safe_offset
            else:
                y_center = 0.0

            # blend the path to start from the car's current heading
            blend = min(1.0, max(0.0, x / blend_dist))
            blend = blend * blend * (3 - 2 * blend)  # smoothing
            
            local_path.append((x, y_center * blend))

        # smooth corners using a moving average
        smoothed_path = []
        window = 3
        half_w = window // 2
        n = len(local_path)
        for i in range(n):
            start = max(0, i - half_w)
            end = min(n, i + half_w + 1)
            avg_y = sum(p[1] for p in local_path[start:end]) / (end - start)
            smoothed_path.append((local_path[i][0], avg_y))

        # transform back to world frame
        world_path: Path2D = []
        inv_cos, inv_sin = math.cos(yaw), math.sin(yaw)
        for lx, ly in smoothed_path:
            wx = cx + lx * inv_cos - ly * inv_sin
            wy = cy + lx * inv_sin + ly * inv_cos
            world_path.append((wx, wy))

        return world_path
