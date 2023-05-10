import numpy as np

sizes = [1, 2, 3, 4]
biases = [np.random.randn(y) for y in sizes[1:]] 
weights = [np.random.randn(y, x)
    for x, y in zip(sizes[:-1], sizes[1:])] 
nabla_b = [np.zeros(b.shape) for b in biases]
nabla_w = [np.zeros(w.shape) for w in weights]
    # x: number of links to one neuron in the 2nd layer
    # y: number of neurons
for b, w in zip(biases, weights):
    print(sizes[-3])

# print("w = ", weights,"\n" ,"b = ", biases)
# print("nabla_w = ", nabla_w,"\n", "nabla_b = ", nabla_b)