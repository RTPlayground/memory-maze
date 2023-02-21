""" Test maze is a minimal working example that shows the environment rendering 
    on the screen.
"""

# https://stackoverflow.com/questions/71010343/cannot-load-swrast-and-iris-drivers-in-fedora-35/71421355#71421355

#!pip install gym
import random
import gymnasium as gym
import mujoco
import cv2
from memory_maze import tasks
# Set this if you are getting "Unable to load EGL library" error:
#  os.environ['MUJOCO_GL'] = 'glfw'  

#env = gym.make('memory_maze:MemoryMaze-9x9-v0')
#env = gym.make('memory_maze:MemoryMaze-11x11-v0')
#env = gym.make('memory_maze:MemoryMaze-13x13-v0')
#env = gym.make('memory_maze:MemoryMaze-15x15-v0')

# Cannot make the observation visualize the image
env = tasks.memory_maze_9x9(
    #global_observables=True,
    image_only_obs=True,
    #top_camera=False,
    #camera_resolution=64,
    #control_freq=4.0,
    #discrete_actions=False,
    )
# This is working
env = gym.make('memory_maze:MemoryMaze-15x15-v0', camera_resolution=64, top_camera=False, discrete_actions=True, control_freq=20)
env.reset()

for _ in range(1000):
    #action = env.action_space.sample()
    action = random.randint(0, 5)
    print('action:{}'.format(action))
    obs, reward, done, trunc, info = env.step(action)  # type: ignore
    img = env.render()
    #print('obs:{}'.format(obs))
    cv2.imshow('img', img)
    cv2.waitKey(1)

