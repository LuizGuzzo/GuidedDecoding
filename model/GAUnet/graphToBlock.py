import torch
import torch.nn as nn
from copy import deepcopy
from model.GAUnet.operationSequences import get_func



class GraphConvBlock(nn.Module):
    def __init__(self, graph, operation_id, in_ch, md_ch, out_ch):
        super(GraphConvBlock, self).__init__()
        self.graph = graph
        self.layers = nn.ModuleDict()

        self.layers["0"] = get_func(operation_id, in_channel=in_ch, out_channel=md_ch)

        # criando os nós com os blocos convolucionais
        for node in graph.keys():
            self.layers[str(node)] = deepcopy(get_func(operation_id, in_channel=md_ch, out_channel=out_ch))



    def dfs(self, graph, node, activations, layers):
        if node in activations:
            return activations[node]

        # print(f"Calculando a ativação do nó {node}")

        # Inicialize a ativação do nó com zeros
        activations[node] = torch.zeros_like(activations[0])
        
        # Calcule a ativação do nó somando as ativações dos nós filhos
        for child in graph[node]:
            activations[node] = activations[node] + layers[str(node)](self.dfs(graph, child, activations, layers))

        # print(f"Ativação do nó {node} calculada")  # Imprime quando a ativação do nó atual é calculada

        return activations[node]


    def forward(self, x):
        
        activations = {}

        # o nó inicial passa pela convolução
        activations[0] = self.layers["0"](x)

        # Encontre o nó de saída (não é filho de nenhum outro nó)
        # all_children = [child for children in self.graph.values() for child in children]
        # output_node = [node for node in self.graph.keys() if node not in all_children][0]
        output_node = 6 

        # Execute a busca em profundidade a partir do nó de saída
        self.dfs(self.graph, output_node, activations, self.layers)

        return activations[output_node]


# # Exemplo de uso
# graph = {3: [2], 4: [1], 5: [1, 4], 1: [0], 2: [0], 6: [3, 5]}
# operation_id = 1
# graph_conv_block = GraphConvBlock(graph, operation_id, in_ch=3, md_ch=20, out_ch=20)

# # Teste
# input_tensor = torch.rand(1, 3, 32, 32)
# print(input_tensor.shape)
# output = graph_conv_block(input_tensor)
# print(output.shape)
