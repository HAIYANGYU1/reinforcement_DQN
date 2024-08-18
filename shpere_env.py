import time
import random
from copy import deepcopy

from rich.console import Console
import numpy as np

import pybullet as p
import pybullet_data as pd


console = Console()

class SphereEnv:
    def __init__(self,player_pos,target_pos) -> None:
        self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pd.getDataPath())
        p.setGravity(0, 0, -10)

        planeId = p.loadURDF("plane.urdf")

        self.player_pos_x = player_pos[0]
        self.player_pos_y = player_pos[1]
        self.initial_pos = [self.player_pos_x,self.player_pos_y]

        self.target_pos_x = target_pos[0]
        self.target_pos_y = target_pos[1]


        self.targetId = p.loadURDF("cube.urdf", 
                        [self.target_pos_x,self.target_pos_y,0], 
                        p.getQuaternionFromEuler([0, 0, 0]),
                        globalScaling=0.5)
            
        self.count = 0
        self.num_actions = 2
        self.is_done = False

    def reset(self):
        self.cur_state = deepcopy(self.initial_pos)
        return self.cur_state

    def _get_state_dim(self):
        return len(self.initial_pos)

    def _get_action_dim(self):
        return self.num_actions

    def transition(self, state, action):
        if self.is_done:
            return 0, state, True
        
        sphereStartPos = [state[0],state[1], 0.25]
        sphereStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
        sphereId = p.loadURDF("sphere2.urdf", 
                      sphereStartPos, sphereStartOrientation,globalScaling=0.5)
        distance = -1
        # [0,0.5,0]
        action = [action[0],action[1],0]
        p.resetBaseVelocity(sphereId, action)
        console.print(f"action={action}",style="white on green")
        for step_id in range(100):
            p.stepSimulation()
            time.sleep(1./240.)
            if step_id == 99:

                spherePos, _ = p.getBasePositionAndOrientation(sphereId)
                targetPos, _ = p.getBasePositionAndOrientation(self.targetId)
                distance = np.linalg.norm(np.array(targetPos) - np.array(spherePos))
        reward = -distance * 10

        self.count += 1

        if(distance < 0.5) or distance >10:
            self.is_done = True

        if distance < 0.5:
            reward = 100
            
        if distance >20:
            self.is_done = True
            reward =-50

        if self.count >= 10:
            self.is_done = True
            
        console.print(f"distance={distance}",style="white on green")
        console.print(f"spherePos={spherePos}",style="white on green")
        console.print(f"reward={reward}",style="white on green")
        return reward,[spherePos[0],spherePos[1]],self.is_done





