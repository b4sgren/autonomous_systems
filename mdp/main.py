import car_params as params 
from mdp import MDPPlanner
import matplotlib.pyplot as plt
import numpy as np

def drawArrows(ax, map, policy):
    arrow_len = 0.75
    arrow_x0 = .125
    for i in range(1, params.r-1):
        for j in range(1, params.c-1):
            if np.isnan(policy[i,j]):
                continue 
            elif policy[i,j] == 0: # North
                debug = 1
            elif policy[i,j] == 1: #South
                debug = 1
            elif policy[i,j] == 2: #East
                debug = 1
            else: #West
                debug = 1
    return ax
            


if __name__ == "__main__":
    planner = MDPPlanner()

    planner.createPolicy()
    planner.correcStaticCells()
    print("Created the policy")

    plt.figure(1)
    ax = plt.imshow(planner.map * 255)
    ax = drawArrows(ax, planner.map, planner.policy)
    plt.colorbar()

    plt.show()