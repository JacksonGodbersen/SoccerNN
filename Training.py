import Environment
import PolicyNetwork

env = Environment.Environment()



for i in range(10000):
    model1 = PolicyNetwork.NueralNetwork()
    model2 = PolicyNetwork.NueralNetwork()
    model3 = PolicyNetwork.NueralNetwork()
    model4 = PolicyNetwork.NueralNetwork()
    print("ready")

    result = env.run(model1, model2, model3, model4)
    print(result)

print(result)