import random
from micrograd.engine import Value

class Module:
    """
    Base class for all neural network modules. 
    Provides methods to reset gradients and retrieve parameters.
    """

    def zero_grad(self):
        """Sets the gradient of all parameters to zero."""
        for p in self.parameters():
            p.grad = 0.0
    
    def parameters(self):
        """Returns a list of parameters (weights and biases) of the module."""
        return []

class Neuron(Module):
    """
    Represents a single neuron in a neural network layer.

    Parameters:
    - nin: Number of inputs to the neuron.
    - nonlin: Boolean indicating if a non-linear activation function (ReLU) is applied.
    """

    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(nin)]  # Weights initialization
        self.b = Value(0)  # Bias initialization
        self.nonlin = nonlin  # Whether to apply ReLU activation
    
    def __call__(self, x):
        """
        Computes the output of the neuron for the given input.

        Parameters:
        - x: List of input values.

        Returns:
        - The output after applying the activation function.
        """
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)  # Weighted sum plus bias
        return act.relu() if self.nonlin else act  # Apply ReLU if nonlin is True

    def parameters(self):
        """Returns the parameters (weights and bias) of the neuron."""
        return self.w + [self.b]
    
    def __repr__(self):
        """Returns a string representation of the neuron."""
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):
    """
    Represents a layer of neurons in a neural network.

    Parameters:
    - nin: Number of inputs to each neuron in the layer.
    - nout: Number of neurons in the layer.
    """

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]  # Initialize neurons
    
    def __call__(self, x):
        """
        Computes the output of the layer for the given input.

        Parameters:
        - x: List of input values.

        Returns:
        - The output of the layer as a list of neuron outputs.
        """
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out  # Return a single output if only one neuron
    
    def parameters(self):
        """Returns the parameters of all neurons in the layer."""
        return [p for n in self.neurons for p in n.parameters()]
    
    def __repr__(self):
        """Returns a string representation of the layer."""
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):
    """
    Represents a Multi-Layer Perceptron (MLP) neural network.

    Parameters:
    - nin: Number of inputs to the network.
    - nouts: List of integers representing the number of neurons in each layer.
    """

    def __init__(self, nin, nouts):
        sz = [nin] + nouts  # Layer sizes
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i != len(nouts) - 1) for i in range(len(nouts))]

    def __call__(self, x):
        """
        Computes the output of the MLP for the given input.

        Parameters:
        - x: List of input values.

        Returns:
        - The final output of the MLP.
        """
        for layer in self.layers:
            x = layer(x)  # Pass the input through each layer sequentially
        return x
    
    def parameters(self):
        """Returns the parameters of all layers in the MLP."""
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        """Returns a string representation of the MLP."""
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"
