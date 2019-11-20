import car_params as params 
from mdp import MDPPlanner
import matplotlib.pyplot as plt

if __name__ == "__main__":
    planner = MDPPlanner()

    planner.createPolicy()
    print("Created the policy")

    plt.imshow(planner.map * 255)
    plt.show()