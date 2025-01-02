#Notes
#
#End env after fixed num steps ex 10000
#Rearrange input tensor to have agent and agents teamates data first and second respectively
#Randomized starting position (rndm soccer ball start mayb)
#Crowd
#Graphics
#Improved user interface
#Goal improvements (Stop phasing)


import concurrent.futures
from multiprocessing import freeze_support
import Environment
import PolicyNetwork

import random
from collections import defaultdict

def initialize_agents(num_agents=100):
    agent_list = []
    for i in range(num_agents):
        model = PolicyNetwork.NueralNetwork()
        agent_list.append(model)

    return agent_list

from itertools import combinations


def generate_matches(num_agents=100, games_per_agent=20, agents_per_match=4):
    # Initialize a dictionary to track how many games each agent has played
    matches_played = {agent: 0 for agent in range(num_agents)}
    matches = []

    # Create a list of agents (can shuffle it for randomness)
    available_agents = list(range(num_agents))

    # Keep looping until all agents have played the required number of games
    while len(matches) < (num_agents * games_per_agent) // agents_per_match:
        # Shuffle the available agents randomly
        random.shuffle(available_agents)

        # Try to form a match
        for i in range(0, len(available_agents), agents_per_match):
            match = available_agents[i:i + agents_per_match]

            # Ensure that the match contains only agents who haven't played too many games
            valid_match = all(matches_played[agent] < games_per_agent for agent in match)

            if valid_match:
                matches.append(tuple(match))
                for agent in match:
                    matches_played[agent] += 1

        # Filter out agents who have already played enough games
        available_agents = [agent for agent in available_agents if matches_played[agent] < games_per_agent]

    return matches

print(len(generate_matches()))

# Define the environment instance and simulation function
def run_simulation(model1, model2, model3, model4, match):
    env = Environment.Environment()  # Initialize environment inside the function
    result = env.run(model1, model2, model3, model4)
    return result, match

def simulate_generation(agent_list, matches):
    random.shuffle(agent_list)

    freeze_support()  # Required for multiprocessing on Windows

    # Parameters
    num_workers = 20  # Number of worker processes

    results = []
    agent_wins = [0] * 100  # Initialize a list of 100 agents with 0 wins

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = []
        for match in matches:
            idx1, idx2, idx3, idx4 = match

            model1 = agent_list[idx1]
            model2 = agent_list[idx2]
            model3 = agent_list[idx3]
            model4 = agent_list[idx4]

            # Submit the simulation task to the executor
            futures.append(executor.submit(run_simulation, model1, model2, model3, model4, match))

        # Process results from the futures
        for future in concurrent.futures.as_completed(futures):
            match_result, match = future.result()  # result is a tuple (result, match)

            print(match_result)

            # Update agent_wins based on match result
            idx1, idx2, idx3, idx4 = match

            if match_result == 1:
                # If agents 1 and 2 win
                agent_wins[idx1] += 1
                agent_wins[idx2] += 1
            elif match_result == -1:
                # If agents 3 and 4 win
                agent_wins[idx3] += 1
                agent_wins[idx4] += 1
            else:
                # If it's a draw, both sets of agents get half a win
                agent_wins[idx1] += 0.5
                agent_wins[idx2] += 0.5
                agent_wins[idx3] += 0.5
                agent_wins[idx4] += 0.5

    return agent_list, agent_wins


if __name__ == "__main__":
    agent_list = initialize_agents()
    matches = generate_matches()
    agent_list, agent_wins = simulate_generation(agent_list, matches)

    print("---------")
    print(agent_wins)


