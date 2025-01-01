import Environment
import PolicyNetwork

env = Environment.Environment()

model1 = PolicyNetwork.NueralNetwork()
model2 = PolicyNetwork.NueralNetwork()
model3 = PolicyNetwork.NueralNetwork()
model4 = PolicyNetwork.NueralNetwork()

result = env.run(model1, model2, model3, model4)

print(result)