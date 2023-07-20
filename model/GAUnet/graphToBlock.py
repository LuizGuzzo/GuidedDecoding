import torch
import torch.nn as nn
from copy import deepcopy
from model.GAUnet.operationSequences import get_func
# from operationSequences import get_func

class GraphConvBlock(nn.Module):
    def __init__(self, graph, operation_id, in_ch, md_ch, out_ch):
        super(GraphConvBlock, self).__init__()
        self.graph = graph

        # self.operation = get_func(operation_id, in_channel=in_ch, out_channel=md_ch)
        self.layers = nn.ModuleDict()
        self.layers["0"] = get_func(operation_id, in_channel=in_ch, out_channel=md_ch)
        for node in graph.keys():
            self.layers[str(node)] = deepcopy(get_func(operation_id, in_channel=md_ch, out_channel=out_ch))

    def forward(self, x):
        # Determinar o nó raiz
        root = max(self.graph.keys())
        return self.compute(root, x)
        
    
    def compute(self,node, x):
        if node not in self.graph:  # o nó não tem filhos
            return self.layers[str(node)](x)
        
        # Calcule a ativação somando as ativações dos nós filhos
        activations = sum(self.compute(child, x) for child in self.graph[node])
        return self.layers[str(node)](activations)
    

# # Exemplo de uso
# graph = {3: [2], 4: [1], 5: [1, 4], 1: [0], 2: [0], 6: [3, 5]}
# operation_id = 1
# graph_conv_block = GraphConvBlock(graph, operation_id, in_ch=3, md_ch=20, out_ch=20)

# # Teste
# input_tensor = torch.rand(1, 3, 32, 32)
# print(input_tensor.shape)
# output = graph_conv_block(input_tensor)
# print(output.shape)
