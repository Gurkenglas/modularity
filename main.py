# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
vmap = lambda f, x: torch.stack([f(x) for x in x.unbind()])

# Record all outputs of linear modules as f(x) runs.
accumulator = []
def activations(f, x):
  global accumulator
  accumulator = []
  f(x)
  return torch.hstack(accumulator)

class Lambda(nn.Module):
  def __init__(self, func):
    super().__init__()
    self.func=func
  def forward(self, x):
    return self.func(x)

def trace(input):  # eeeeevil
  global accumulator
  accumulator.append(input)
  return input

layer = lambda dims: nn.Sequential(
  nn.GELU(),
  nn.Linear(dims[0], dims[1]),
  Lambda(trace)
)

mlp = lambda dims: nn.Sequential(
  layer(dims[0:2]),
  layer(dims[1:3]),
  layer(dims[2:4]),
  layer(dims[3:5]),
)

def correlation(covariance):
  inv_std = (1 / torch.sqrt(torch.diag(covariance))).expand([covariance.shape[0], -1])
  return inv_std * covariance * inv_std.T

index = 0
def show(matrix):
  global index
  plt.figure(index, figsize=(3,3), dpi=900)
  plt.matshow(matrix, fignum = index)
  plt.colorbar()
  index += 1
  plt.show()

import matplotlib.pyplot as plt
from torch.autograd.functional import jacobian

inputdims = 2
outputdims = 3
teachers = 4
studentinner = 15
teacherinner = 16
student = mlp([inputdims, studentinner,studentinner,studentinner, outputdims * teachers])
teachers = [mlp([inputdims, teacherinner,teacherinner,teacherinner, outputdims]) for _ in range(teachers)]
label = lambda input: torch.cat([t(input) for t in teachers], -1)
optimizer = torch.optim.Adam(student.parameters(),lr=0.02)
student.train()
for t in teachers:
  t.train()
log = []
for _ in range(100):
  input = torch.rand([50,inputdims], requires_grad=True)
  loss = nn.MSELoss()(label(input),student(input))
  loss.backward()
  optimizer.step()
  optimizer.zero_grad()
N = 6
x = torch.rand([N*N, inputdims], requires_grad=True)
# j describes what student does to a neighborhood of x
j = vmap(lambda x: jacobian(lambda x:activations(student, x),x), x)
# student sends a normal distribution around x with covariance matrix 1
           # to a normal distribution with this     covariance matrix:
cov = vmap(lambda j: j.matmul(j.T),j)
corr = vmap(correlation,cov)
yyxx = corr.view(N, N, *corr.shape[1:]).permute([2,0,3,1])
show(yyxx.reshape(yyxx.shape[0]*yyxx.shape[1], yyxx.shape[2]*yyxx.shape[3]))