import Environment
import main

env = Environment.Environment()

model1 = main.NueralNetwork()
model2 = main.NueralNetwork()
model3 = main.NueralNetwork()
model4 = main.NueralNetwork()

env.run(model1, model2, model3, model4)

env.run()