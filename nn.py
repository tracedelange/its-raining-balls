import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



#convert the above code into a function
def get_input_array(balls):
    N = 8
    DIM = 800

    input_array = np.zeros(N)
    for ball in balls:
        try:
            x, y = ball
            index = min(int(x // (DIM / N)), N-1)
            input_array[index] = max(input_array[index], y) / DIM
        except IndexError:
            print('Index error', ball)
    return input_array