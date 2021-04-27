# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
vmap = lambda f, x: torch.stack([f(x) for x in x.unbind()])

# Record all module outputs as f(x) runs.
def activations(f, x):
  accumulator = []
  def hook(module, input, output):
    nonlocal accumulator
    accumulator.append(output)
  h = f.register_forward_hook(hook)
  f(x)
  h.remove()
  return torch.hstack(accumulator)

layer = lambda dims: nn.Sequential(
  nn.GELU(),
  nn.Linear(dims[0], dims[1]),
)

mlp = lambda dims: nn.Sequential(
  layer(dims[0:2]),
  layer(dims[1:3]),
  layer(dims[2:4]),
  layer(dims[3:5]),
)

class Vmap(nn.Sequential):
  forward = lambda self,x: torch.stack([m.forward(x) for m,x in zip(self,x)])

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
nteachers = 4
studentinner = 15
teacherinner = 16
sqrtnsamples = 3
student = mlp([inputdims, studentinner, studentinner, studentinner, outputdims * nteachers])
teachers = [mlp([inputdims, teacherinner, teacherinner, teacherinner, outputdims]) for _ in range(nteachers)]
label = lambda input: torch.cat([t(input) for t in teachers], -1)
optimizer = torch.optim.Adam(student.parameters(), lr=0.02)
log = []
for _ in range(100):
  input = torch.rand([50, inputdims])
  loss = nn.MSELoss()(label(input), student(input))
  loss.backward()
  optimizer.step()
  optimizer.zero_grad()
x = torch.rand([sqrtnsamples * sqrtnsamples, inputdims])
# each j describes what student does to a neighborhood of x
js = vmap(lambda x: jacobian(lambda x: activations(student, x), x), x)
# student sends a normal distribution around x with covariance matrix 1
# to a normal distribution with this     covariance matrix:
covs = vmap(lambda j: j @ j.T, js)
squaredcorrs = vmap(correlation, covs).pow(2)
yyxx = squaredcorrs.view(sqrtnsamples, sqrtnsamples, *squaredcorrs.shape[1:]).permute([2, 0, 3, 1])
show(yyxx.reshape(yyxx.shape[0] * yyxx.shape[1], yyxx.shape[2] * yyxx.shape[3]))