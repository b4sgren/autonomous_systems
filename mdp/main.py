import car_params as params 
from mdp import MDPPlanner
import matplotlib.pyplot as plt
import numpy as np

def drawArrows(ax, map, policy):
    arrow_len = 0.55
    x0 = .375
    w = 0.25
    l = 0.15
    for i in range(1, params.r-1):
        for j in range(1, params.c-1):
            if np.isnan(policy[i,j]):
                continue 
            elif policy[i,j] == 0: # North
                plt.arrow(j, i + x0, 0, -arrow_len, head_width=w, head_length=l)
            elif policy[i,j] == 1: #South
                plt.arrow(j, i - x0, 0, arrow_len, head_width=w, head_length=l)
            elif policy[i,j] == 2: #East
                plt.arrow(j-x0, i, arrow_len, 0, head_width=w, head_length=l)
            else: #West
                plt.arrow(j + x0, i, -arrow_len, 0, head_width=w, head_length=l)
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