# imports 
import numpy as np
import matplotlib.pyplot as plt
import math
from graphviz import Digraph
import torch
import random

def draw(o):
     dot = draw_dot(o)
     path = dot.render(filename="value_graph", cleanup=True, view=True)
     print(path)

# defining a function : (expects a scalar and return a scalar too)
def f(x):
    return 3*x**2 -4*x + 5

print(f(2))

# let's just plot it 
xs = np.arange(-10 , 10 , 0.5)
ys = f(xs)
plt.plot(xs,ys)
plt.grid(True)
# plt.show()

# let's use basic diff principle
h = 0.000001
x = 3.0
grad = (f(x+h)-f(x))/h
print(grad)


# now we will move to more complex case (function) 
a = 3.0
b = -2.0
c = 11.0

d1 = a*b + c
# a+=h
# b+=h
c+=h
d2 = a*b+c
print(f"d1 : {d1}")
print(f"d2 : {d2}")
print(f"slope : {(d2-d1)/h}")

# now we want to wrap these scalars inside some object 

import math

class Value:
    def __init__(self, data, _ancestors=(), _op='', label=''):
        self.data = data
        self._prev = set(_ancestors)
        self._op = _op
        self.label = label
        self.grad = 0.0
        self._backward = lambda: None 

    def __repr__(self):
        return f"Value(data={self.data})"

    def __rmul__(self, other): 
         return self * other

    def __radd__(self, other): 
        return self + other

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
             self.grad += 1.0 * out.grad
             other.grad += 1.0 * out.grad
        out._backward = _backward 
        return out 
    
    def __neg__(self):
         return self * -1
    
    def __sub__(self, other):
         # Fixed: Added 'return' so it passes the resulting Value object back
         return self + (-other)

    def __rsub__(self, other):
         # Fixed: Added support for expression like: number - Value
         return Value(other) - self

    def __mul__(self, other):
         other = other if isinstance(other, Value) else Value(other)
         out = Value(self.data * other.data, (self, other), '*')
         def _backward():
              self.grad += other.data * out.grad
              other.grad += self.data * out.grad
         out._backward = _backward 
         return out

    def tanh(self):
         n = self.data
         out = Value((math.exp(2*n) - 1)/(math.exp(2*n)+1), (self,), 'tanh')
         def _backward():
              self.grad += (1 - out.data * out.data) * out.grad
         out._backward = _backward
         return out

    def exp(self):
         x = self.data
         out = Value(math.exp(x), (self,), 'exp')
         def _backward():
              self.grad += out.data * out.grad 
         out._backward = _backward
         return out

    def __truediv__(self, other): 
         other = other if isinstance(other, Value) else Value(other)
         return self * (other ** -1)

    def __pow__(self, other):
         assert isinstance(other, (int, float)), "only supporting int/float powers for now"
         out = Value(self.data**other, (self,), f"**{other}")
         def _backward():
              self.grad += (other * (self.data ** (other - 1))) * out.grad
         out._backward = _backward
         return out

    def backward(self):
         topo = []
         visited = set()
         def build_topo(v):
              if v not in visited:
                   visited.add(v)  # Fixed: Keep track of visited nodes to avoid loops
                   for child in v._prev:
                        build_topo(child)
                   topo.append(v)
         build_topo(self)
         
         self.grad = 1.0
         for node in reversed(topo):
              node._backward()


        
a = Value(2.0 , label='a')
b = Value(-3.0, label='b')
c = Value(10.0, label='c')
e = a*b ; e.label = 'e'
d = e+c ; d.label='d'
f = Value(-2.0, label='f')
L = d*f ; L.label='l'
print(d._prev)
print(d._op)
print(d.label)
# print(a.__add__(b))
# print(a.__mul__(b))

# print(a*b + c)

#now we will try to visualize it via some code ( copied from micrograd original one)
# basically dfs 
def trace(root):
     # builds a set of 
     nodes, edges = set() , set()
     def build(v):
          if v not in nodes:
               nodes.add(v)
               for child in v._prev:
                    edges.add((child,v))
                    build(child)
     build(root)
     return nodes , edges

def draw_dot(root):
     dot = Digraph(format='svg', graph_attr={'rankdir': 'LR'}) #LR left to right 

     nodes, edges = trace(root)
     for n in nodes :
          uid = str(id(n))
          # for any graph create a rectangular ('record') node for it
          dot.node(name = uid , label = "{%s | data %.4f | grad %.4f} " % (n.label, n.data, n.grad), shape='record')
          if n._op:
               #if this value is a result of some operation , create an op node for it
               dot.node(name=uid+n._op, label = n._op)
               # connect this node to it
               dot.edge(uid+n._op, uid)
     for n1,n2 in edges :
          dot.edge(str(id(n1)), str(id(n2))+n2._op)
     return dot

# dot = draw_dot(L)
# path = dot.render(filename="value_graph", cleanup=True, view=True)
# print(path)


# ------------------------------
# Manual back propagation 
L.grad = 1.0 # manually 
# dL/dd = f = -2.0
# dL / df = d = 4.0
d.grad = -2.0
f.grad = 4.0

# now we will dervive dL/dc
# d = c+e
# dd / dc = 1.0 (derivative rules)
# dd / de = 1.0 (derivative rules)
# by partial derivative ( chain rule )
# now how do c impact L : we need to see all it's contribution to next layer to it 
# here we c just impact d but in neural net it could impact many next layer node and we need to use
# partial derivative rule there too
# dL/dc = (dd/dc) * (dL/dd) = (1.0)*(-2.0) = -2.0
# similary dL/de = (dd/de) * (dL/dd) = (1.0)*(-2.0) = -2.0
c.grad = -2.0
e.grad = -2.0

# de/da = b
# de/db = a

# we need to apply all the way through the graph 
# dL/da = (dL/de)*(de/da) = (-2.0)*(-3.0) = 6.0
# dL/db = (dL/de)*(de/db) = (-2.0)*(2.0) = -4.0
a.grad = 6.0
b.grad = -4.0

# recursive application of chain rule in the computation graph : back propagation 

# now how does a single step optimization looks like ?
# now that we know the gradients so we know how nudging the inputs can influence the
# final outcome i.e, L in our case  , so we will increase that by tiny amonunt in 
# proportional to gradients
a.data += 0.01*a.grad
b.grad += 0.01*b.grad
c.grad += 0.01*c.grad
d.grad += 0.01*d.grad
# since we increase in the direction of the gradient we expect L to increase/go in 
# positive direction 
# forward pass again 
h = 0.0001
e = a*b 
d = e+c 
L = d*f ; #prev it was -8.0000... now its -7.6400000...1

# print(L.data)

# inline gradient check : derviated check ( numerical gradient : estimating using small step size)
def lol():
     h = 0.0001
     a = Value(2.0 , label='a')
     b = Value(-3.0, label='b')
     c = Value(10.0, label='c')
     e = a*b ; e.label = 'e'
     d = e+c ; d.label='d'
     f = Value(-2.0, label='f')
     L = d*f ; L.label='l'
     L1 = L.data
    
     a = Value(2.0+h , label='a')
     b = Value(-3.0, label='b')
     c = Value(10.0, label='c')
     e = a*b ; e.label = 'e'
     d = e+c ; d.label='d'
     f = Value(-2.0, label='f')
     L = d*f ; L.label='l'
     L2 = L.data

     print((L2-L1)/h)

# lol()




# dot = draw_dot(L)
# path = dot.render(filename="value_graph", cleanup=True, view=True)
# print(path)

# ---------------------------------------
# now trying to do backpropagation for each operation in a part of neural net

# inputs , weights and bias
x1 = Value (2.0, label='x1')
x2 = Value (0.0, label='x2')
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
b = Value(6.8813735870195432, label='b')
# x1*w1 + x2*w2 + b
x1w1 = x1*w1; x1w1.label = "x1w1"
x2w2 = x2*w2; x2w2.label = "x2w2" # Fixed: Changed x1 to x2, and updated string label
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
n = x1w1x2w2 + b; n.label = 'n'
o = n.tanh(); o.label = 'o'


#  back prop for gradients manually 
o.grad = 1.0
# d(tanhx)/dx = 1 - tanh^2x
# do/dn = 1 - o**2 
n.grad = 1 - o.data**2
# similary go back  ( and for plus(+) operation the gradient just passes as it is)
x1w1x2w2.grad = 0.5
b.grad = 0.5
x1w1.grad = 0.5
x2w2.grad = 0.5 
# now for rest 
x2.grad = w2.data*x2w2.grad
w2.grad = x2.data*x2w2.grad
x1.grad = w1.data*x1w1.grad
w1.grad = x1.data*x1w1.grad

# dot = draw_dot(o)
# path = dot.render(filename="value_graph", cleanup=True, view=True)
# print(path)


# now we want to do these back prop automatically not manually like we did for two examples
# so we need to update Value class ( we added backward function in the opertaion methods)
# and upgraded grad as per chain rule for those operations 

# forward pass 
# inputs , weights and bias
x1 = Value (2.0, label='x1')
x2 = Value (0.0, label='x2')
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
b = Value(6.8813735870195432, label='b')
# x1*w1 + x2*w2 + b
x1w1 = x1*w1; x1w1.label = "x1w1"
x2w2 = x2*w2; x2w2.label = "x2w2" # Fixed: Changed x1 to x2, and updated string label
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'
n = x1w1x2w2 + b; n.label = 'n'
o = n.tanh(); o.label = 'o'

# back_prop
# base set to one the last node 
# calling _backwards manually 
# o.grad = 1.0
# o._backward() # this should call that method and calculates and propagates derivative to prev layer i.e, n
# n._backward()
# b._backward()
# x1w1x2w2._backward()
# x2w2._backward()
# x1w1._backward()

# # drawing
# dot = draw_dot(o)
# path = dot.render(filename="value_graph", cleanup=True, view=True)
# print(path)

# now we want to get rid of us calling _backward manually too
# and this can be achieved via toposort algorithm used in graph 
# we will be wrting that in the Value class so that we can use it directly to call once
# rather than manually calling is sequence which is absurb in larger network 
o.backward()
# draw(o)
# drawing
# dot = draw_dot(o)
# path = dot.render(filename="value_graph", cleanup=True, view=True)
# print(path)



#----------------------------------------------
# everything works fine until we use the same variable more than once in forward pass
# things like : 
# a = Value(1.0) ; b = a+a
# or 
# a = Value(); b = Value()
# d = a*b ; e = a+a ; f = d*e 
# because of our function the gradient overrides 
# solution is : 
# we need to accumulate those gradients if a variable is involved in multiple positions
# basically chain rule 
# so : for example from : self.grad = 1.0*out.grad  ---> self.grad += 1.0*out.grad

a = Value(3.0, label='a')
b = a+a ; b.label = 'b'
b.backward()
# draw(b)


# ---------------------------

x1 = Value (2.0, label='x1')
x2 = Value (0.0, label='x2')
w1 = Value(-3.0, label='w1')
w2 = Value(1.0, label='w2')
b = Value(6.8813735870195432, label='b')

x1w1 = x1*w1; x1w1.label = "x1w1"
x2w2 = x2*w2; x2w2.label = "x2w2" 
x1w1x2w2 = x1w1 + x2w2; x1w1x2w2.label = 'x1*w1 + x2*w2'

n = x1w1x2w2 + b; n.label = 'n'
#--- breaking tanh ----
e = (2*n).exp()
o = (e-1)/(e+1)
# ----
o.label = 'o'
e.lable = 'e'
o.backward()
# draw(o)


# we can do the same thing but via pytorch api 
x1 = torch.tensor([2.0],dtype=torch.float64) ; x1.requires_grad = True
x2 = torch.tensor([0.0],dtype=torch.float64) ; x2.requires_grad = True
w1 = torch.tensor([-3.0],dtype=torch.float64) ; w1.requires_grad = True
w2 = torch.tensor([1.0],dtype=torch.float64) ; w2.requires_grad = True
b = torch.tensor([6.8813735870195432],dtype=torch.float64) ; b.requires_grad = True
n = x1*w1+x2*w2+b
o = torch.tanh(n)
print(o.data)
o.backward()
print("-------")
print (f"x2 grad : {x2.grad.item()}")
print (f"x1 grad : {x1.grad.item()}")
print (f"w2 grad : {w2.grad.item()}")
print (f"w1 grad : {w1.grad.item()}")

# ----------------------------------------
# building a single nueron 
class Neuron :
     def __init__(self, nin):
          self.w = [Value(random.uniform(-1,1))for _ in range(nin)]
          self.b = Value(random.uniform(-1,1))

     def __call__(self, x):
          # w*x +b 
          act = sum((wi*xi for wi, xi in zip(self.w,x)),self.b)
          out = act.tanh()
          return out
     def parameters(self):
          return self.w+[self.b]
     

# a single layer
class Layer :
     def __init__(self,nin,nout):
          self.neurons = [Neuron(nin)for _ in range(nout)]

     def __call__(self, x):
          outs = [n(x) for n in self.neurons]
          return outs[0] if len(outs)==1 else outs
     def parameters(self):
          params = []
          for nueron in self.neurons:
               ps = nueron.parameters()
               params.extend(ps)
          return params
     
# entire layer mlp
class MLP :
     def __init__(self, nin , nouts):
          sz = [nin] + nouts
          self.layers = [Layer(sz[i],sz[i+1]) for i in range(len(nouts))]
     def __call__(self, x):
          for layer in self.layers:
               x = layer(x)
          return x
     def parameters(self):
          return [p for layer in self.layers for p in layer.parameters()]
        #   params = []
        #   for layer in self.layers:
        #        ps = layer.parameters()
        #        params.extend(ps)
        #   return params
          

# for MLP code understanding 
# =====================================================================
# MICROGRAD MLP (MULTI-LAYER PERCEPTRON) REFERENCE NOTE
# =====================================================================

# 1. THE ARCHITECTURE CONFIGURATION (__init__)
# Goal: Chain layers sequentially, ensuring Layer(Inputs, Outputs) match.
# Example: MLP(nin=3, nouts=[4, 4, 1])
#   - sz = [3] + [4, 4, 1] -> [3, 4, 4, 1] (Complete sizing map)
#   - Loop creates layers using adjacent sizes (sz[i] to sz[i+1]):
#     - i = 0: Layer(3, 4) -> Accepts 3 inputs, contains 4 neurons
#     - i = 1: Layer(4, 4) -> Accepts 4 inputs, contains 4 neurons
#     - i = 2: Layer(4, 1) -> Accepts 4 inputs, contains 1 output neuron

# 2. THE FORWARD PROPAGATION FLOW (__call__)
# Goal: Pass inputs sequentially through the network ("Assembly Line").
#   - Uses a rolling variable update rule: x = layer(x)
#   - Step 1: Input features [x1, x2, x3] enter Layer 0 -> returns 4 values
#   - Step 2: Those 4 values immediately enter Layer 1 -> returns 4 new values
#   - Step 3: Those 4 new values enter Layer 2 -> returns final predictions
# =====================================================================


# ----------------------
x = [2.0,3.0,-1.0]
n = MLP(3,[4,4,1])
# n = Neuron(2)
# n = Layer(2,3)

# print(n(x))
# print(n.parameters())
# print(f"size : {len(n.parameters())}")

# draw(n(x))

# now lets creating a tiny dataset
xs = [
     [2.0,3.0,-1.0],
     [3.0,-1.0,0.5],
     [0.5,1.0,1.0],
     [1.0,1.0,-1.0]
]
ys = [1.0, -1.0, -1.0, 1.0]

# ypred = [n(x) for x in xs]
# print(f"ypred: {ypred}")

# loss = sum([(yout-ygt)**2 for ygt,yout in zip(ys,ypred)])
# print(f"loss: {loss}")

# loss.backward()

# # print(n.layers[0].neurons[0].w[0].grad)
# print(n.layers[0].neurons[0].w[0].data)

# draw(loss)

# for p in n.parameters():
#      p.data += -0.01*p.grad # - minus as we want to decrease the loss

# print(n.layers[0].neurons[0].w[0].data)

# ypred = [n(x) for x in xs]
# print(f"ypred: {ypred}")

# loss = sum([(yout-ygt)**2 for ygt,yout in zip(ys,ypred)])
# print(f"loss: {loss}")


# now we have to iterate the process (forward pass , backward pass and nudge the value)
loss_history = []
epochs = 100
learning_rate = 0.01
for k in range(epochs):
     # forward pass
     ypred = [n(x) for x in xs]
     loss = sum([(yout-ygt)**2 for ygt, yout in zip(ys,ypred)])

     # .zero_grad()(in pytorch) before .backward() ( to avoid accumulation from prev training loop)
     for parameter in n.parameters():
          parameter.grad = 0.0

     # backward pass
     loss.backward()

     # gradient descent 
     for p in n.parameters():
          p.data += -learning_rate*p.grad

     loss_history.append(loss.data)


# plt.figure(figsize=(8, 5))

# plt.plot(loss_history)

# plt.xlabel("Epoch")
# plt.ylabel("Binary Cross Entropy Loss")
# plt.title("Training Loss Curve")

# plt.grid(True)
# plt.show()

ypred = [n(x) for x in xs]
print(ypred)

# -------------------------------

# nueral net mistakes : 
# 1. we forget to fit a single batch first
# 2. we forget to toggle train/eval mode for the net
# 3. we forget to .zero_grad()(in pytorch) before .backward()
# 4. we pass the softmaxed outputs to a loss that expects a raw logits 