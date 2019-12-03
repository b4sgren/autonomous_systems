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
            elif policy[i,j] == 3: # North
                plt.arrow(j, i + x0, 0, -arrow_len, head_width=w, head_length=l)
            elif policy[i,j] == 2: #South
                plt.arrow(j, i - x0, 0, arrow_len, head_width=w, head_length=l)
            elif policy[i,j] == 0: #East
                plt.arrow(j-x0, i, arrow_len, 0, head_width=w, head_length=l)
            else: #West
                plt.arrow(j + x0, i, -arrow_len, 0, head_width=w, head_length=l)
    return ax

def drawPath(ax, y, x, policy):
    pp_x = x 
    pp_y = y 
    p_x = x 
    p_y = y
    while not np.isnan(policy[y,x]):
        if policy[y,x] == 3: #North
            plt.plot([x, x], [y, y-1], 'r')
            pp_y = p_y
            p_y = y
            y -= 1
        elif policy[y,x] == 2: #South
            plt.plot([x, x], [y, y+1], 'r')
            pp_y = p_y
            p_y = y
            y += 1
        elif policy[y,x] == 0: #East
            plt.plot([x, x+1], [y, y], 'r')
            pp_x = p_x
            p_x = x
            x += 1
        else:
            plt.plot([x, x-1], [y, y], 'r')
            pp_x = p_x
            p_x = x
            x -= 1
        if pp_x == x and pp_y == y:
            break
    return ax
            
if __name__ == "__main__":
    planner = MDPPlanner()

    planner.createPolicy()
    planner.correcStaticCells()
    print("Created the policy")

    plt.figure(1)
    ax = plt.imshow(planner.map * 255)
    ax = drawArrows(ax, planner.map, planner.policy)
    ax = drawPath(ax, params.x0, params.y0, planner.policy)
    plt.colorbar()

    plt.show()