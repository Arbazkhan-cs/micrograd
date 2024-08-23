import math

class Value:
    """
    A class representing a scalar value in a computation graph, 
    which supports automatic differentiation.
    """

    def __init__(self, data, _children=(), _op=""):
        """
        Initialize the Value object.

        Parameters:
        - data: The scalar value.
        - _children: The set of parent nodes in the computation graph.
        - _op: The operation that produced this value.
        """
        self.data = data
        self.grad = 0.0  # Gradient of the value, initialized to zero
        self._backward = lambda: None  # Function to backpropagate gradients
        self._prev = set(_children)  # Set of parent nodes
        self._op = _op  # Operation that produced this value

    def __repr__(self):
        """Return a string representation of the Value object."""
        return f"Value(data={self.data}, grad={self.grad})"

    def __add__(self, other):
        """
        Add two Value objects (or a Value and a scalar).

        Parameters:
        - other: The other value to add.

        Returns:
        - A new Value object representing the sum.
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), _op="+")

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out
    
    def __mul__(self, other):
        """
        Multiply two Value objects (or a Value and a scalar).

        Parameters:
        - other: The other value to multiply.

        Returns:
        - A new Value object representing the product.
        """
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), _op="*")

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward

        return out
    
    def __pow__(self, other):
        """
        Raise the value to the power of a scalar.

        Parameters:
        - other: The exponent (must be an int or float).

        Returns:
        - A new Value object representing the power.
        """
        assert isinstance(other, (int, float)), "only supporting int/float powers for now"
        out = Value(self.data ** other, (self,), _op=f"**{other}")

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad
        out._backward = _backward

        return out
    
    def relu(self):
        """
        Apply the ReLU activation function.

        Returns:
        - A new Value object after applying ReLU.
        """
        out = Value(max(0, self.data), (self,), _op="ReLU")

        def _backward():
            self.grad += (out.data > 0) * out.grad
        out._backward = _backward

        return out
    
    def tanh(self):
        """
        Apply the tanh activation function.

        Returns:
        - A new Value object after applying tanh.
        """
        x = self.data
        t = (math.exp(2*x) - 1) / (math.exp(2*x) + 1)
        out = Value(t, (self,), _op="tanh")

        def _backward():
            self.grad += (1 - t**2) * out.grad
        out._backward = _backward

        return out

    def backward(self):
        """
        Perform backpropagation to compute the gradients of all 
        preceding nodes in the computation graph.
        """
        # Build the topological order of the computation graph
        topo = []
        visited = set()
        
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # Initialize the gradient of the output to 1
        self.grad = 1.0
        for v in reversed(topo):
            v._backward()
    
    def __neg__(self):
        """Return the negation of the Value object."""
        return self * -1
    
    def __sub__(self, other):
        """Subtract another Value object or scalar from this Value."""
        return self + (-other)
    
    def __radd__(self, other):
        """Right-hand addition to support scalar + Value."""
        return self + other
    
    def __rsub__(self, other):
        """Right-hand subtraction to support scalar - Value."""
        return other + (-self)
    
    def __rmul__(self, other):
        """Right-hand multiplication to support scalar * Value."""
        return self * other
    
    def __truediv__(self, other):
        """Divide this Value by another Value or scalar."""
        return self * other**-1
    
    def __rtruediv__(self, other):
        """Right-hand division to support scalar / Value."""
        return other * self**-1
    