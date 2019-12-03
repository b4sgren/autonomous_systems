import car_params as params 
from pomdp import POMDPPlanner
import matplotlib.pyplot as plt
import numpy as np
            
if __name__ == "__main__":
    planner = POMDPPlanner()

    planner.createPolicy()
    print("Created the policy")