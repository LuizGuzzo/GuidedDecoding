import torch
import torch.nn as nn
import time
from tqdm import tqdm

class NormalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(NormalConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.conv(x))


class DepthwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(DepthwiseConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.pointwise(self.depthwise(x)))


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def compute_latency(model, input_shape=(1, 3, 224, 224), warmup=100, repeats=1, use_cuda=True):
    inputs = torch.randn(input_shape)
    
    if use_cuda and torch.cuda.is_available():
        model = model.cuda()
        inputs = inputs.cuda()
    
    # Warm up
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(inputs)
    
    if use_cuda:
        torch.cuda.synchronize()
    
    # Measurement
    times = []
    for _ in tqdm(range(repeats)):
        if use_cuda:
            start_time = torch.cuda.Event(enable_timing=True)
            end_time = torch.cuda.Event(enable_timing=True)
            start_time.record()
        else:
            start_time = time.time()
        
        with torch.no_grad():
            _ = model(inputs)
        
        if use_cuda:
            end_time.record()
            torch.cuda.synchronize()
            times.append(start_time.elapsed_time(end_time)) # milliseconds
        else:
            times.append(time.time() - start_time)
    
    # Average over the measurement iterations.
    latency = sum(times) / repeats
    return latency


def main(use_cuda=True):
    # Create the models
    normal_conv = NormalConv(3, 64, 3, padding=1)
    depthwise_conv = DepthwiseConv(3, 64, 3, padding=1)

    # Count the parameters
    normal_conv_params = count_parameters(normal_conv)
    depthwise_conv_params = count_parameters(depthwise_conv)

    print(f'Normal Conv Parameters: {normal_conv_params}')
    print(f'Depthwise Conv Parameters: {depthwise_conv_params}')

    # Compute the latency
    normal_conv_latency = compute_latency(normal_conv, use_cuda=use_cuda)
    depthwise_conv_latency = compute_latency(depthwise_conv, use_cuda=use_cuda)

    print(f'Normal Conv Latency: {normal_conv_latency} ms')
    print(f'Depthwise Conv Latency: {depthwise_conv_latency} ms')


if __name__ == "__main__":
    main(use_cuda=False)
