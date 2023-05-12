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


class GraphConvBlock(nn.Module):
    def __init__(self, graph, operation_id, in_ch, md_ch, out_ch):
        super(GraphConvBlock, self).__init__()
        self.graph = graph
        self.layers = nn.ModuleDict()

        self.layers["0"] = get_func(operation_id, in_channel=in_ch, out_channel=md_ch)

        # TODO: alterar o DFS
        self.order = list(reversed(topological_sort(graph)))

        # criando os nós com os blocos convolucionais
        for node in graph.keys():
            self.layers[str(node)] = deepcopy(get_func(operation_id, in_channel=md_ch, out_channel=out_ch))
    
    def forward(self, x):
        activations = {}

        # o nó inicial passa pela convolução
        activations[0] = self.layers["0"](x)

        # Ordem dos nós. Isso deve ser alterado para uma função que calcula a ordenação topológica
        # No seu caso, você já sabe qual é a ordem correta, então pode simplesmente escrevê-la

        # Calcule as ativações dos nós em ordem
        for node in self.order[1:]:  # Comece do segundo nó, pois já calculamos a ativação do nó 0
            # Inicialize a ativação do nó com zeros
            activations[node] = torch.zeros_like(activations[0])
            
            # Calcule a ativação do nó somando as ativações dos nós filhos
            for child in self.graph[node]:
                activations[node] = activations[node] + self.layers[str(node)](activations[child])

        # A saída é a ativação do último nó
        output = activations[self.order[-1]]
        return output


# # Exemplo de uso
# graph = {3: [2], 4: [1], 5: [1, 4], 1: [0], 2: [0], 6: [3, 5]}
# operation_id = 1
# graph_conv_block = GraphConvBlock(graph, operation_id, in_ch=3, md_ch=20, out_ch=20)

# # Teste
# input_tensor = torch.rand(1, 3, 32, 32)
# print(input_tensor.shape)
# output = graph_conv_block(input_tensor)
# print(output.shape)
