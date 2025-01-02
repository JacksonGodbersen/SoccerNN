import concurrent.futures
import Environment
import PolicyNetwork

# Create the environment instance
env = Environment.Environment()

def run_simulation():
    model1 = PolicyNetwork.NueralNetwork()
    model2 = PolicyNetwork.NueralNetwork()
    model3 = PolicyNetwork.NueralNetwork()
    model4 = PolicyNetwork.NueralNetwork()
    result = env.run(model1, model2, model3, model4)
    return result

# Use a ProcessPoolExecutor for multiprocessing
num_simulations = 10000
num_workers = 4  # Number of processes to use

results = []
with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
    futures = [executor.submit(run_simulation) for _ in range(num_simulations)]
    for future in concurrent.futures.as_completed(futures):
        results.append(future.result())
        print(future.result())
