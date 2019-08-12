---
title: "pytorch"
date: 2019-01-06T17:44:32+01:00
draft: false
categories: ["scratchpad"]
---

<center>
# This is not a proper blog post yet, just my notes.

pytorch (TODO)
</center>

[pytorch vs tensorflow](https://towardsdatascience.com/pytorch-vs-tensorflow-spotting-the-difference-25c75777377b)

```
"""
examples of using pytorch borrowed from
https://towardsdatascience.com/pytorch-vs-tensorflow-spotting-the-difference-25c75777377b
"""
# one example
import torch
from torch.autograd import Variable
import numpy as np


def rmse(y, y_hat):
    """Compute root mean squared error"""
    return torch.sqrt(torch.mean((y - y_hat).pow(2).sum()))


def forward(x, e):
    """Forward pass for our fuction"""
    return x.pow(e.repeat(x.size(0)))


# Let's define some settings
n = 100  # number of examples
learning_rate = 5e-6
target_exp = 2.0  # real value of the exponent will try to find

# Model definition
x = Variable(torch.rand(n) * 10, requires_grad=False)

# Model parameter and it's true value
exp = Variable(torch.FloatTensor([target_exp]), requires_grad=False)
# just some starting value, could be random as well
exp_hat = Variable(torch.FloatTensor([4]), requires_grad=True)
y = forward(x, exp)

# a couple of buffers to hold parameter and loss history
loss_history = []
exp_history = []

# Training loop
for i in range(0, 200):

    # Compute current estimate
    y_hat = forward(x, exp_hat)

    # Calculate loss function
    loss = rmse(y, y_hat)

    # Compute gradients
    loss.backward()

    # Update model parameters
    exp_hat.data -= learning_rate * exp_hat.grad.data
    exp_hat.grad.data.zero_()
    print(exp_hat.data)
```
