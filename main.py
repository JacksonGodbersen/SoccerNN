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
import os
from multiprocessing import freeze_support
import Environment
import PolicyNetwork
import torch
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

            #print(match_result)

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


def mutate_model(model, mutation_rate=0.1, mutation_strength=0.05):
    # Clone the model
    new_model = PolicyNetwork.NueralNetwork()

    # Get the state dict of the original model and apply mutation
    state_dict = model.state_dict()
    new_state_dict = {}

    for name, param in state_dict.items():
        # Apply mutation only on some parameters based on mutation_rate
        if random.random() < mutation_rate:  # If mutation occurs on this parameter
            # Apply random noise (mutation_strength controls how much to change)
            noise = torch.randn_like(param) * mutation_strength
            new_state_dict[name] = param + noise
        else:
            new_state_dict[name] = param

    # Load the modified parameters into the new model
    new_model.load_state_dict(new_state_dict)
    return new_model

def save_model(model, generation_num):
    # Save the model's state_dict
    filepath = os.path.join("models", f"model_{generation_num}.pth")
    torch.save(model.state_dict(), filepath)


def generation_step(agent_list, matches, generation_num):
    # Simulate the generation to get the win list
    agent_list, win_list = simulate_generation(agent_list, matches)

    next_generation = []

    # Step 1: Keep agents that have won more than half of their games
    winners = [i for i in range(100) if win_list[i] > 10]

    # Step 2: Add winners to the next generation
    for i in winners:
        next_generation.append(agent_list[i])

    # Step 3: Replace losers with mutated versions of winners
    num_winners = len(winners)
    for i in range(100):
        if win_list[i] <= 10:  # If the agent is a loser
            # Randomly select a winner and mutate their model
            winner_idx = random.choice(winners)
            next_generation.append(mutate_model(agent_list[winner_idx]))

    max_idx = win_list.index(max(win_list))
    best_agent = agent_list[max_idx]
    save_model(best_agent, generation_num)

    print("Generation " + str(generation_num) + " is complete")

    return next_generation


if __name__ == "__main__":
    agent_list = initialize_agents()
    matches = generate_matches()

    generation_num = 0
    while True:
        agent_list = generation_step(agent_list, matches, generation_num)
        generation_num += 1
