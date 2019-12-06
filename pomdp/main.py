import car_params as params 
from pomdp import POMDPPlanner
import matplotlib.pyplot as plt
import numpy as np
plt.ion()
            
if __name__ == "__main__":
    planner = POMDPPlanner()

    planner.createPolicy()
    print("Created the policy")
    print("Alpha Vectors: ")
    print(planner.Y)

    V, P = planner.getMax()

    plt.figure(1)
    plt.plot(P,V)
    plt.draw()
    plt.xlim([0,1])
    plt.ylim([-100, 100])
    
    for i in range(1):
        print("Simulating")
        bel = 0.6
        true_state = 0
        planner.simulate(bel, true_state)
        input("Verify the simulation is correct. Then press enter to continue")