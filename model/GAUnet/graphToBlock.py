import torch
import torch.nn as nn
from copy import deepcopy
from model.GAUnet.operationSequences import get_func
# from operationSequences import get_func

def topological_sort(graph):
    """Retorna uma lista dos nós do grafo em ordem topológica."""
    visited = set()
    order = []

    def dfs(node):
        """Visita todos os nós alcançáveis a partir do nó atual usando busca em profundidade."""
        visited.add(node)
        for neighbour in graph.get(node, []):
            if neighbour not in visited:
                dfs(neighbour)
        order.insert(0, node)

    # order = dfs("6")
    for node in graph:
        if node not in visited:
            dfs(node)

    return order

#Recursividade, sem armazenamento de tensores (acreditava ser mais leve)
class GraphConvBlock(nn.Module):
    def __init__(self, graph, operation_id, in_ch, md_ch, out_ch):
        super(GraphConvBlock, self).__init__()
        self.graph = graph
        self.operation = get_func(operation_id, in_channel=in_ch, out_channel=md_ch)

    def forward(self, x):
        # Determinar o nó raiz
        root = max(self.graph.keys())
        return self.compute(root, x)
        
    
    def compute(self,node, x):
        if node not in self.graph:  # o nó não tem filhos
            return self.operation(x)
        
        # Calcule a ativação somando as ativações dos nós filhos
        activations = sum(self.compute(child, x) for child in self.graph[node])
        return self.operation(activations)
    

# ordenação DFS com armazenamento de tensores
"""
class GraphConvBlock(nn.Module):
    def __init__(self, graph, operation_id, in_ch, md_ch, out_ch):
        super(GraphConvBlock, self).__init__()
        self.graph = graph
        # self.layers = nn.ModuleDict()
        self.operation = get_func(operation_id, in_channel=in_ch, out_channel=md_ch)

        # TODO: alterar o DFS
        # self.order = list(reversed(topological_sort(graph)))

        pass
    
    def forward(self, x):
        activations = {}

        # o nó inicial passa pela convolução
        activations[0] = self.operation(x)

        # Ordem dos nós. Isso deve ser alterado para uma função que calcula a ordenação topológica
        # No caso, já sei qual é a ordem correta, então posso simplesmente escrevê-la

        #return activations[0] #teste, a execução foi pra 622FPS..
        
        # Calcule as ativações dos nós em ordem
        for node in self.order[1:]:  # Comece do segundo nó, porque nó 0 nao existe no grafo
            # Calcule a ativação do nó somando as ativações dos nós filhos
            for i,child in enumerate(self.graph[node]):
                if i == 0:
                    activations[node] = self.operation(activations[child])
                else:
                    activations[node] += self.operation(activations[child])          

        # A saída é a ativação do último nó
        output = activations[self.order[-1]]
        return output
"""

# # Exemplo de uso
# graph = {3: [2], 4: [1], 5: [1, 4], 1: [0], 2: [0], 6: [3, 5]}
# operation_id = 1
# graph_conv_block = GraphConvBlock(graph, operation_id, in_ch=3, md_ch=20, out_ch=20)

# # Teste
# input_tensor = torch.rand(1, 3, 32, 32)
# print(input_tensor.shape)
# output = graph_conv_block(input_tensor)
# print(output.shape)
