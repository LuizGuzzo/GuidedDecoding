import torch.nn as nn
from model.GAUnet.graphToBlock import GraphConvBlock
from model.GAUnet.binToUnet import binary_to_unet
import torch

class GAUNet(nn.Module):
    def __init__(self, num_classes=1, input_channels=3, mid_channels = 20, bin_genotype = None ):
        super().__init__()

        # conversor de binario para o genotype transcrito
        genotype = binary_to_unet(bin_genotype)
        
        # encoder
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv2d do codigo original
        self.init_conv = nn.Conv2d(in_channels=input_channels, out_channels=mid_channels, kernel_size=3, stride=1, padding=1)

        encoder_size = 4
        self.genotype_sequence_encoder = nn.ModuleList()
        for i in range(encoder_size):
            dict = genotype[i]
            op_sequence = dict["operation_sequence"]
            connections = dict["connections"]
            self.genotype_sequence_encoder.append(
                GraphConvBlock(graph=connections, operation_id=op_sequence, in_ch=mid_channels, md_ch=mid_channels, out_ch=mid_channels)
            )

        self.genotype_sequence_encoder = nn.Sequential(*self.genotype_sequence_encoder)

        # decoder
        self.up_list = nn.ModuleList()
        
        self.genotype_sequence_decoder = nn.ModuleList()
        for i in range(encoder_size-1):
            dict = genotype[encoder_size+i]
            op_sequence = dict["operation_sequence"]
            connections = dict["connections"]
            self.genotype_sequence_decoder.append(
                GraphConvBlock(graph=connections, operation_id=op_sequence, in_ch=mid_channels, md_ch=mid_channels, out_ch=mid_channels)
            )
            self.up_list.append(nn.ConvTranspose2d(in_channels=mid_channels, out_channels=mid_channels, kernel_size=2, stride=2))
        
        self.genotype_sequence_decoder = nn.Sequential(*self.genotype_sequence_decoder)

        self.final_conv = nn.Conv2d(in_channels=mid_channels, out_channels=num_classes, kernel_size=1, stride=1)
        self.mish = nn.Mish()

        # rgb > convblock > maxpool > ... > maxpool > convblock > up > convblock > ... > up > pred

        # self.sigmoid = nn.Sigmoid()


        
    def forward(self, input):
        
        x = self.init_conv(input)
        encode_outputs = [None for _ in range(len(self.genotype_sequence_encoder))]

        for i, convBlock in enumerate(self.genotype_sequence_encoder):
            if i == 0:
                encode_outputs[i] = convBlock(x)
            else:
                encode_outputs[i] = convBlock(self.maxpool(encode_outputs[i - 1]))

        for i, convBlock in enumerate(self.genotype_sequence_decoder):
            if i == 0:
                out = convBlock(self.up_list[i](encode_outputs[-1]) + encode_outputs[-(2 + i)])
            else:
                out = convBlock(self.up_list[i](out) + encode_outputs[-(2 + i)])
            
        out = self.final_conv(out)
        out = self.mish(out)
        # out = self.sigmoid(out)
        return out

# # Exemplo de uso
# graph = {3: [2], 4: [1], 5: [1, 4], 1: [0], 2: [0], 6: [3, 5]}
# operation_id = 1
# graph_conv_block = GraphConvBlock(graph, operation_id, in_ch=3, md_ch=20, out_ch=20)

# # Teste Bloco
# input_tensor = torch.rand(1, 3, 32, 32)
# print(input_tensor.shape)
# output = graph_conv_block(input_tensor)
# print(output.shape)

# # Teste GAUnet
# bin_list = [0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1]
# unet = GAUNet(num_classes=1,input_channels=3,mid_channels=20,bin_genotype=bin_list)

# input_tensor = torch.rand(1, 3, 480, 640)
# print(input_tensor.shape)
# output = unet(input_tensor)
# print(output.shape)