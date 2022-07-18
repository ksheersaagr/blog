---
layout: post
title: "Graph SLAM: A Noob’s Guide to Simultaneous Localisation and Mapping"  
author: krunal kshirsagar
---

Simultaneous localisation and mapping (SLAM) used in the concurrent construction of a model of the environment (the map), and the estimation of the state of the robot moving within it. In other words, SLAM gives you a way to track the location of a robot in the world in real-time and identify the locations of landmarks such as buildings, trees, rocks, and other world features. In addition to localisation, we also want to build up a model of the robot’s environment so that we have an idea of objects, and landmarks that surround it and so that we can use this map data to ensure that we are on the right path as the robot moves through the world. So the key insight in building a map is that the robot itself might lose track of where it is by virtue of its motion uncertainty since there is no presence of an existing map because we are building the map simultaneously. That’s where SLAM comes into play.

# Working of SLAM:

The basis for simultaneous localisation and mapping (SLAM) is to gather information from a robot’s sensors and motions over time, and then use information about measurements and motion to reconstruct a map of the world. In this case, we’ll be localizing a robot in a 2D grid world and therefore, a **[graph-based SLAM approach constructs a simplified estimation problem by abstracting the raw sensor measurements. These raw measurements are replaced by the edges in the graph which can then be seen as virtual measurements](http://www2.informatik.uni-freiburg.de/~stachnis/pdf/grisetti10titsmag.pdf).**  
Let’s assume we have a robot and the initial location, **x0=0 & y0=0.** For this example, we don’t care about heading direction just to keep things simple. Let’s assume the robot moves to the right in the X-direction by **10**. So, In a perfect world, you would know that **x1**, the location after motion is the same as **x0+10** in other words, **x1=x0+10**, and **y1** is the same as **y0.**

<img src="{{ site.baseurl }}/images/2020-03-31-graph-slam-a-noobs-guide-to-simultaneous-localisation-and-mapping-displacement-by-10.png
">

But according to **[Kalman filters](https://noob-can-compile.github.io/home/2020/02/28/the-curious-case-of-kalman-filters.html)** and various other robotic techniques, we have learned that the **location is actually uncertain**. So, rather than assuming in our X-Y coordinate system the robot moved to the right by 10 exactly, it’s better to understand that the actual location of the robot after the x1= x0+10 motion update is a Gaussian centered around (10,0), but it’s possible that the robot is somewhere else.

<img src="{{ site.baseurl }}/images/2020-03-31-graph-slam-a-noobs-guide-to-simultaneous-localisation-and-mapping-gaussian-centered-around-the-location-of-robot-after-motion-update.png
">
   
# Here’s the math for the Gaussian of x variable:  

Rather than setting x1 to x0+10, let’s express it in Gaussian that peaks when these two things are the same. So, if you subtract x1-x0-10, put this into a square format, and turn this into a Gaussian, we get a probability distribution that relates x1 and x0. We can do the same for y. Since there is no change in y according to our motion, y1 & y0 are as close together as possible.

<img src="{{ site.baseurl }}/images/2020-03-31-graph-slam-a-noobs-guide-to-simultaneous-localisation-and-mapping-likelihood.png
">

The product of these two Gaussian is now our constraint. The goal is to maximize the likelihood of the position x1 given the position x0 is (0,0). **So, what graph-SLAM does is, it defines the probabilities using a sequence of such constraints.** Say we have a robot that moves in some space, GRAPH SLAM collects its initial location which is (0,0) initially, also called as **Initial Constraints**, then collects lots of relative constraints that relate each robot pose to the previous robot pose also called as **Relative Motion Constraints.** As an example, let’s use landmarks that can be seen by the robot at various locations which would be **Relative Measurement Constraints** every time a robot sees a landmark. **So, Graph SLAM collects those constraints in order to find the most likely configuration of the robot path along with the location of landmarks, and that is the mapping process.**

<img src="{{ site.baseurl }}/images/2020-03-31-graph-slam-a-noobs-guide-to-simultaneous-localisation-and-mapping-update-position.png
">


## Implementation

# Generating an environment:  
We will generate a 2D world grid with landmarks in it and then generate data by placing a robot in that world and moving and sensing over some number of time steps. The data is collected as an instantiated robot moves and senses in a world. Our SLAM function will take in this data as input. So, let’s first create this data and explore how it represents the movement and sensor measurements that our robot takes.

# SLAM inputs:  

#### In addition to data, our slam function takes in:  

- **N:** The number of time steps that a robot will be moving and sensing.  
- **num_landmarks:** The number of landmarks in the world.   
- **world_size:** The size (w/h) of your world.  
- **motion_noise:** The noise associated with motion; the update confidence for motion should be `1.0/motion_noise`.    
- **measurement_noise:** The noise associated with measurement/sensing; the update weight for measurement should be `1.0/measurement_noise`.  

```python
import numpy as np
from helpers import make_data

# your implementation of slam should work with the following inputs
# feel free to change these input values and see how it responds!

# world parameters
num_landmarks      = 5        # number of landmarks
N                  = 20       # time steps
world_size         = 100.0    # size of world (square)

# robot parameters
measurement_range  = 50.0     # range at which we can sense landmarks
motion_noise       = 2.0      # noise in robot motion
measurement_noise  = 2.0      # noise in the measurements
distance           = 20.0     # distance by which robot (intends to) move each iteratation 


# make_data instantiates a robot, AND generates random landmarks for a given world size and number of landmarks
data = make_data(N, num_landmarks, world_size, measurement_range, motion_noise, measurement_noise, distance)
```

# Let’s write our two main functions that move our robot around, help locate landmarks and measure the range between them on a 2D map:  

- **Move:** attempts to move the robot by dx, dy.
Sense: returns x and y distances to landmarks within the visibility range.  
- **Sense:** returns x and y distances to landmarks within the visibility range.

```python
class robot:
    
    #move function
    def move(self, dx, dy):
        
        x = self.x + dx + self.rand() * self.motion_noise
        y = self.y + dy + self.rand() * self.motion_noise
        
        if x < 0.0 or x > self.world_size or y < 0.0 or y > self.world_size:
            return False
        else:
            self.x = x
            self.y = y
            return True
    
    
    #sense function
    def sense(self):
        measurements = []

        for landmark_index, landmark in enumerate(self.landmarks):
            landmark_distance_x = landmark[0]
            landmark_distance_y = landmark[1]
            random_noise = self.rand()
            cal_dx = self.x - landmark_distance_x + random_noise * self.measurement_noise
            cal_dy = self.y - landmark_distance_y + random_noise * self.measurement_noise
            is_not_in_measurement_range = self.measurement_range == -1
            if(is_not_in_measurement_range) or ((abs(cal_dx) <= self.measurement_range) and (abs(cal_dy) <= self.measurement_range)):
                measurements.append([landmark_index, cal_dx, cal_dy])
        return measurements
```

# Omega and Xi:  
To implement Graph SLAM, a matrix and a vector (omega and xi, respectively) are introduced. The matrix is square, labeled with all the robot poses (xi) and all the landmarks. Every time you make an observation, for example, as you move between two poses by some distance dx and can relate those two positions, you can represent this as a numerical relationship in these matrices. let’s write the function such that it returns omega and xi constraints for the starting position of the robot. Any values that we do not yet know should be initialized with the value 0. we may assume that our robot starts out in exactly the middle of the world with 100% confidence.

```python
def initialize_constraints(N, num_landmarks, world_size):
    ''' This function takes in a number of time steps N, number of landmarks, and a world_size,
        and returns initialized constraint matrices, omega and xi.'''
    
    middle_of_the_world = world_size / 2
    
    ## Recommended: Define and store the size (rows/cols) of the constraint matrix in a variable
    rows, cols = 2*(N + num_landmarks), 2*(N + num_landmarks)
    ## TODO: Define the constraint matrix, Omega, with two initial "strength" values
    omega = np.zeros(shape = (rows, cols))
    ## for the initial x, y location of our robot
    #omega = [0]
    
    omega[0][0], omega[1][1] = 1,1
    
    ## TODO: Define the constraint *vector*, xi
    ## you can assume that the robot starts out in the middle of the world with 100% confidence
    #xi = [0]
    xi = np.zeros(shape = (rows, 1))
    xi[0][0] = middle_of_the_world
    xi[1][0] = middle_of_the_world
    
    return omega, xi
```

# Updating with motion and measurements:

```python
## slam takes in 6 arguments and returns mu, 
## mu is the entire path traversed by a robot (all x,y poses) *and* all landmarks locations
def slam(data, N, num_landmarks, world_size, motion_noise, measurement_noise):
    
    ## TODO: Use your initilization to create constraint matrices, omega and xi
    omega, xi = initialize_constraints(N, num_landmarks, world_size)
    ## TODO: Iterate through each time step in the data
    for time_step in range(len(data)):
        
        ## get all the motion and measurement data as you iterate through each time step
        measurement = data[time_step][0]
        motion = data[time_step][1]
        
        dx = motion[0]         # distance to be moved along x in this time_step
        dy = motion[1]         # distance to be moved along y in this time_step
        
        #Consider the robot moves from (x0,y0) to (x1,y1) in this time_step
        
        #even numbered columns of omega correspond to x values
        x0 = (time_step * 2)   #x0 = 0,2,4,...
        x1 = x0 + 2            #x1 = 2,4,6,...
        
        #odd numbered columns of omega correspond to y values
        y0 = x0 + 1            #y0 = 1,3,5,...
        y1 = y0 + 2            #y1 = 3,5,7,...
        
        actual_m_noise = 1.0/measurement_noise
        actual_n_noise = 1.0/motion_noise
    ## TODO: update the constraint matrix/vector(omega/xi) to account for all *measurements*
    ## this should be a series of additions that take into account the measurement noise
        for landmark in measurement:
            lM = landmark[0]            # landmark id
            dx_lM = landmark[1]         # separation along x from current position
            dy_lM = landmark[2]         # separation along y from current position
            
            L_x0 = (N*2) + (lM*2)       # even-numbered columns have x values of landmarks
            L_y0 = L_x0 + 1             # odd-numbered columns have y values of landmarks

            # update omega values corresponding to measurement between x0 and Lx0
            omega[x0][x0] += actual_m_noise
            omega[L_x0][L_x0] += actual_m_noise
            omega[x0][L_x0] += -actual_m_noise
            omega[L_x0][x0] += -actual_m_noise
            
            # update omega values corresponding to measurement between y0 and Ly0
            omega[y0][y0] += actual_m_noise
            omega[L_y0][L_y0] += actual_m_noise
            omega[y0][L_y0] += -actual_m_noise
            omega[L_y0][y0] += -actual_m_noise
            
            # update xi values corresponding to measurement between x0 and Lx0
            xi[x0]  -= dx_lM/measurement_noise
            xi[L_x0]  += dx_lM/measurement_noise
            
            # update xi values corresponding to measurement between y0 and Ly0
            xi[y0]  -= dy_lM/measurement_noise
            xi[L_y0] += dy_lM/measurement_noise
            
            
        ## TODO: update the constraint matrix/vector(omega/xi) to account for all *motion* from from (x0,y0) to (x1,y1) and motion noise
        omega[x0][x0] += actual_n_noise
        omega[x1][x1] += actual_n_noise
        omega[x0][x1] += -actual_n_noise
        omega[x1][x0] += -actual_n_noise
        
        omega[y0][y0] += actual_n_noise
        omega[y1][y1] += actual_n_noise
        omega[y0][y1] += -actual_n_noise
        omega[y1][y0] += -actual_n_noise
        
        xi[x0] -= dx/motion_noise
        xi[y0] -= dy/motion_noise
        
        xi[x1] += dx/motion_noise
        xi[y1] += dy/motion_noise
    
    ## TODO: After iterating through all the data
    ## Compute the best estimate of poses and landmark positions
    ## using the formula, omega_inverse * Xi
    inverse_of_omega = np.linalg.inv(np.matrix(omega))
    mu = inverse_of_omega * xi
    
    return mu
```

# Robot Poses & Landmarks:

Let’s print the estimated pose and landmark locations that our function has produced. We define a function that extracts the poses and landmarks locations and returns those as their own separate lists.

```python
def get_poses_landmarks(mu, N):
    # create a list of poses
    poses = []
    for i in range(N):
        poses.append((mu[2*i].item(), mu[2*i+1].item()))

    # create a list of landmarks
    landmarks = []
    for i in range(num_landmarks):
        landmarks.append((mu[2*(N+i)].item(), mu[2*(N+i)+1].item()))

    # return completed lists
    return poses, landmarks
  
def print_all(poses, landmarks):
    print('\n')
    print('Estimated Poses:')
    for i in range(len(poses)):
        print('['+', '.join('%.3f'%p for p in poses[i])+']')
    print('\n')
    print('Estimated Landmarks:')
    for i in range(len(landmarks)):
        print('['+', '.join('%.3f'%l for l in landmarks[i])+']')

# call your implementation of slam, passing in the necessary parameters
mu = slam(data, N, num_landmarks, world_size, motion_noise, measurement_noise)

# print out the resulting landmarks and poses
if(mu is not None):
    # get the lists of poses and landmarks
    # and print them out
    poses, landmarks = get_poses_landmarks(mu, N)
    print_all(poses, landmarks)
```
#### Estimated Robot poses and landmarks

<img src="{{ site.baseurl }}/images/2020-03-31-graph-slam-a-noobs-guide-to-simultaneous-localisation-and-mapping-estimated-robot-poses-and-landmarks.png
">

# Visualise the constructed world:

```python
# import the helper function
from helpers import display_world

# Display the final world!

# define figure size
plt.rcParams["figure.figsize"] = (20,20)

# check if poses has been created
if 'poses' in locals():
    # print out the last pose
    print('Last pose: ', poses[-1])
    # display the last position of the robot *and* the landmark positions
    display_world(int(world_size), poses[-1], landmarks)
```

#### Output:

<img src="{{ site.baseurl }}/images/2020-03-31-graph-slam-a-noobs-guide-to-simultaneous-localisation-and-mapping-lastpose.png
">

Check out the code on **[Github.](https://github.com/Noob-can-Compile/Landmark_Detection_Robot_Tracking_SLAM-)** 