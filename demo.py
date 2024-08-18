import time

import numpy as np

import pybullet as p
import pybullet_data as pd

DURATION = 10000

client = p.connect(p.GUI)

p.setAdditionalSearchPath(pd.getDataPath())

p.setGravity(0, 0, -10)
planeId = p.loadURDF("plane.urdf")


sphereStartPos = [-2, 0, 0.25]
sphereStartOrientation = p.getQuaternionFromEuler([0, 0, 0])

sphereId = p.loadURDF("sphere2.urdf", 
                      sphereStartPos, sphereStartOrientation,globalScaling=0.5)
pos_y = 2
cubeTwoStartPos = [2, pos_y, 0.25]
cubeTwoStartOrientation = p.getQuaternionFromEuler([0, 0, 0])

cubeTwoId = p.loadURDF("cube.urdf", 
                       cubeTwoStartPos, 
                       cubeTwoStartOrientation,
                       globalScaling=0.5)

cubeOneStartPos = [2.25, 0, 0.0025]
cubeOneStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
cubeOneId = p.loadURDF("cube.urdf", 
                       cubeOneStartPos, 
                       cubeOneStartOrientation,
                       globalScaling=0.005)

p.resetBaseVelocity(sphereId, [5, 0, 0])

for i in range(DURATION):
    p.stepSimulation()
    time.sleep(1./240.)
    spherePos, sphereOrn = p.getBasePositionAndOrientation(sphereId)

    cubeOnePos, cubeOneOrn = p.getBasePositionAndOrientation(cubeOneId)
    # 
    distance = np.linalg.norm(np.array(cubeOnePos) - np.array(spherePos))

    if distance < 0.5:
        p.resetBaseVelocity(sphereId, [0, 5, 0])
    # print(f"{distance=}")