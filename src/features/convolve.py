import numpy as np
import torch


input = np.arange(81).reshape(9,9)

kernel = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])

def convolve(input, kernel):
    input = torch.tensor(input).float()
    kernel = torch.tensor(kernel).float()
    input = input.unsqueeze(0).unsqueeze(0)
    kernel = kernel.unsqueeze(0).unsqueeze(0)
    output = torch.nn.functional.conv2d(input, kernel, padding=1)
    return output.numpy().squeeze()

print(convolve(input, kernel))


class convolve_mine():
    def __init__(self, input, kernel, padding="none") -> None:
        self.input = input
        self.kernel = kernel
        self.padding = padding
        self.h = self.input.shape[0]
        self.w = self.input.shape[1]
        self.kh = self.kernel.shape[0]
        self.kw = self.kernel.shape[1]
        
    def convolve(self):
        if self.padding == "none":
            self.output = np.zeros((self.h - self.kh + 1, self.w - self.kw + 1))
        if self.padding == "same":
            self.output = np.zeros((self.h, self.w))
            self.input = np.pad(self.input, ((1, 1), (1, 1)))

        for i in range(self.output.shape[0]):
            for j in range(self.output.shape[0]):
                self.output[i, j] = np.sum(self.input[i:i+self.kh, j:j+self.kw] * self.kernel)
                    
        return self.output

conv = convolve_mine(input, kernel, padding="same")
print(conv.convolve())
