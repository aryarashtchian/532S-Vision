import numpy as np
import numpy.matlib

class nn_MSECriterion:
    def forward(self, predictions, labels):
        return np.sum(np.square(predictions - labels))

    def backward(self, predictions, labels):
        num_samples = labels.shape[0]
        return num_samples * 2 * (predictions - labels)

# This is referred above as g(v).
class nn_Sigmoid:
    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def backward(self, x, gradOutput):
        # It is usually a good idea to use gv from the forward pass and not recompute it again here.
        gv = 1 / (1 + np.exp(-x))
        return np.multiply(np.multiply(gv, (1 - gv)), gradOutput)

# This is referred above as h(W, b)
class nn_Linear:
    def __init__(self, input_dim, output_dim):
        # Initialized with random numbers from a gaussian N(0, 0.001)
        self.weight = np.matlib.randn(input_dim, output_dim) * 0.01
        self.bias = np.matlib.randn((1, output_dim)) * 0.01
        self.gradWeight = np.zeros_like(self.weight)
        self.gradBias = np.zeros_like(self.bias)

    def forward(self, x):
        return np.dot(x, self.weight) + self.bias

    def backward(self, x, gradOutput):
        # dL/dw = dh/dw * dL/dv
        self.gradWeight = np.dot(x.T, gradOutput)
        # dL/db = dh/db * dL/dv
        self.gradBias = np.copy(gradOutput)
        # return dL/dx = dh/dx * dL/dv
        return np.dot(gradOutput, self.weight.T)

    def getParameters(self):
        params = [self.weight, self.bias]
        gradParams = [self.gradWeight, self.gradBias]
        return params, gradPar



dataset_size = 1000

# Generate random inputs within some range.
x = np.random.uniform(0, 6, (dataset_size, 4))
# Generate outputs based on the inputs using some function.
y1 = np.sin(x.sum(axis = 1))
y2 = np.sin(x[:, 1] * 6)
y3 = np.sin(x[:, 1] + x[:, 3])
y = np.array([y1, y2, y3]).T

print(x.shape)
print(y.shape)


hidden_state_size = 5;
learningRate = 0.1

model = {}
model['linear1'] = nn_Linear(4, hidden_state_size)
model['linear2'] = nn_Linear(hidden_state_size, 3)
model['sigmoid1'] = nn_Sigmoid()
model['sigmoid2'] = nn_Sigmoid()
model['loss'] = nn_MSECriterion()

for epoch in range(0, 400):
    loss = 0
    for i in range(0, dataset_size):
        xi = x[i:i+1, :]
        yi = y[i:i+1, :]

        # Forward.
        a0 = model['linear1'].forward(xi)
        a1 = model['sigmoid1'].forward(a0)
        a2 = model['linear2'].forward(a1)
        a3 = model['sigmoid2'].forward(a2)
        loss += model['loss'].forward(a3, yi)

        # Backward.
        da3 = model['loss'].backward(a3, yi)
        da2 = model['sigmoid2'].backward(a2,da3)
        da1 = model['linear2'].backward(a1,da2)
        da0 = model['sigmoid1'].backward(a0, da1)
        model['linear1'].backward(xi, da0)

        model['linear1'].weight = model['linear1'].weight - learningRate * model['linear1'].gradWeight
        model['linear1'].bias = model['linear1'].bias - learningRate * model['linear1'].gradBias
        model['linear2'].weight = model['linear2'].weight - learningRate * model['linear2'].gradWeight
        model['linear2'].bias = model['linear2'].bias - learningRate * model['linear2'].gradBias



    if epoch % 10 == 0: print('epoch[%d] = %.8f' % (epoch, loss / dataset_size))


# We will compute derivatives with respect to a single data pair (x,y)
x = np.array([[2.34, 3.8, 34.44, 5.33]])
y = np.array([[3.2, 4.2, 5.3]])

# Define the operations.
model = {}
model['linear1'] = nn_Linear(4, hidden_state_size)
model['linear2'] = nn_Linear(hidden_state_size, 3)
model['sigmoid1'] = nn_Sigmoid()
model['sigmoid2'] = nn_Sigmoid()
model['loss'] = nn_MSECriterion()


# Forward.
a0 = model['linear1'].forward(xi)
a1 = model['sigmoid1'].forward(a0)
a2 = model['linear2'].forward(a1)
a3 = model['sigmoid2'].forward(a2)
loss += model['loss'].forward(a3, yi)

# Backward.
da3 = model['loss'].backward(a3, yi)
da2 = model['sigmoid2'].backward(a2,da3)
da1 = model['linear2'].backward(a1,da2)
da0 = model['sigmoid1'].backward(a0, da1)
model['linear1'].backward(xi, da0)

gradWeight1 = model['linear1'].gradWeight
gradBias1 = model['linear1'].gradBias

gradWeight2 = model['linear2'].gradWeight
gradBias2 = model['linear2'].gradBias


approxGradWeight1 = np.zeros_like(model['linear1'].weight)
approxGradWeight2 = np.zeros_like(model['linear2'].weight)


# We will verify here that gradWeights are correct and leave it as an excercise
# to verify the gradBias.
epsilon = 0.0001

for i in range(0, model['linear1'].weight.shape[0]):
    for j in range(0, model['linear1'].weight.shape[1]):
        # Compute f(w)
        # Forward.
        a0 = model['linear1'].forward(xi)
        a1 = model['sigmoid1'].forward(a0)
        a2 = model['linear2'].forward(a1)
        a3 = model['sigmoid2'].forward(a2)
        fw = model['loss'].forward(a3, yi)


        # Compute f(w + eps)
        shifted_weight = np.copy(model['linear1'].weight)
        shifted_weight[i, j] = shifted_weight[i, j] + epsilon

        shifted_linear = nn_Linear(4, 3)
        shifted_linear.bias = model['linear1'].bias
        shifted_linear.weight = shifted_weight

        a0 = shifted_linear.forward(xi)
        a1 = model['sigmoid1'].forward(a0)
        a2 = model['linear2'].forward(a1)
        a3 = model['sigmoid2'].forward(a2)
        fw_epsilon = model['loss'].forward(a3, yi)

        # Compute (f(w + eps) - f(w)) / eps
        approxGradWeight1[i, j] = (fw_epsilon - fw) / epsilon

# These two outputs should be similar up to some precision.
print('gradWeight: ' + str(gradWeight1))
print('\napproxGradWeight: ' + str(approxGradWeight1))


for i in range(0, model['linear2'].weight.shape[0]):
    for j in range(0, model['linear2'].weight.shape[1]):
        # Compute f(w)
        # Forward.
        a0 = model['linear1'].forward(xi)
        a1 = model['sigmoid1'].forward(a0)
        a2 = model['linear2'].forward(a1)
        a3 = model['sigmoid2'].forward(a2)
        fw = model['loss'].forward(a3, yi)

        print(fw)

        # Compute f(w + eps)
        shifted_weight = np.copy(model['linear2'].weight)
        shifted_weight[i, j] = shifted_weight[i, j] + epsilon

        shifted_linear = nn_Linear(4, 3)
        shifted_linear.bias = model['linear2'].bias
        shifted_linear.weight = shifted_weight

        a0 = model['linear1'].forward(xi)
        a1 = model['sigmoid1'].forward(a0)
        a2 = shifted_linear.forward(a1)
        a3 = model['sigmoid2'].forward(a2)
        fw_epsilon = model['loss'].forward(a3, yi)

        print(fw_epsilon)
        print('------------------------')
        # Compute (f(w + eps) - f(w)) / eps
        approxGradWeight2[i, j] = (fw_epsilon - fw) / epsilon

# These two outputs should be similar up to some precision.
print('gradWeight: ' + str(gradWeight2))
print('\napproxGradWeight: ' + str(approxGradWeight2))
