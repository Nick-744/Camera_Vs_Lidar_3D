# Camera_Vs_Lidar_3D
3Δ Υπολογιστική Γεωμετρία και Όραση (EE844)

---

This project [my_project_4] implements the tasks outlined in the provided assignment, focused on Computer Graphics, 2D images, and 3D point clouds.

## Overview

The goal of this project is to process visual and spatial data to detect roads, identify obstacles, estimate motion vectors, and construct visual projections. The project is divided into two parts:

---

### - Part A: 2D Image Processing 

1. **Road Boundary Detection**  
   Detect the road boundaries and color the two traffic lanes differently from the off-road areas.

2. **Obstacle Detection**  
   Identify obstacles on the road (e.g., cars, pedestrians, and other objects). Classification is not required.

3. **Motion Vector Estimation**  
   Compute the motion vector of a car in the correct direction. If obstacles are detected, draw a circular detour.

4. **Vertical Plane Projection**  
   Construct a vertical plane perpendicular to the road surface, on which a projection of the view (a few meters before the wall) is rendered. Hints are provided in the original description.

---

### - Part B: 3D Point Cloud Processing

1. **Road Detection from 3D Point Cloud**  
   Use the 3D point cloud to detect the road and its boundaries. Classify and color different regions accordingly.

2. **Obstacle Detection**  
   Identify road obstacles (e.g., cars, pedestrians, and other objects). No classification is required.

3. **Motion Vector Estimation**  
   As in Part A, compute the car’s correct movement vector and draw a detour if obstacles exist.
   
   DEMO: https://youtu.be/DXkgFZ9ZrB4

4. **Environment Rendering**  
   Using the setup from Part A.iv, perform steps ii), iii), vi), and vii) in this 3D context.

---

### 🔹 Bonus: CARLA Simulation

As an extension, the core parts of the project have been integrated into the [CARLA simulator](https://carla.org/) to demonstrate real-time performance in a virtual driving environment.

**Video Demonstrations:**
- [Demo 1 – CARLA Simulation - LiDAR road + obstacles detection](https://youtu.be/pzfL9UhPmQs?si=nfasl8OCLi1Re6O4)  
- [Demo 2 – Camera Road/Obstacle detection - CARLA](https://youtu.be/frrVAOl6k38)

---

🔸 Extras:
This repo also includes complete answers to all lab exercises from the semester [Lab1 -> Lab7].
