import networkx as nx
import matplotlib.pyplot as plt


def binary_array_to_graph(binary_array):
    n = int(len(binary_array)/2)
    matrix = []
    index = 0

    # gerando matriz apartir do array binario
    for i in range(n-1):
        row = []
        for j in range(n-1):
            if j <= i:
                row.append(int(binary_array[index]))
                index += 1
            else:
                continue               
        matrix.append(row)

    # conexões
    # (12)
    # (13)(23)
    # (14)(24)(34)
    # (15)(25)(35)(45)

    # 0 
    # 0 0
    # 0 0 0
    # 1 1 1 1
    
    # {5: [1,2,3,4]} - o 5 recebe de 1,2,3,4

    # Criando o grafo de conexões
    graph = {}
    for i in range(len(matrix)):
        for j in range(len(matrix[i])):
            if matrix[i][j] == 1:
                if i+2 not in graph:
                    graph[i+2] = []
                graph[i+2].append(j+1)

    return graph


def create_connections(binary_str, max_nodes=7):
    num_intermediate_nodes = max_nodes - 2

    connections = binary_array_to_graph(binary_str)

    nodes = set(range(1, num_intermediate_nodes + 1))

    # Os que não tem antecessor é oque não esta na key do connection
    no_pred = set(nodes.copy() - set(connections.keys()))

    # Os que não tem sucessor são os nós que não estao no connection
    no_succ = nodes.copy()
    for list_node in connections.values():
        no_succ.difference_update(set(list_node))

    # Os que tem sucessor são os nós do connection, a data das key;
    with_succ = set()
    for list_node in connections.values():
        with_succ.update(set(list_node))

    # Os que tem antecessor é as key do connection;
    with_pred = set(connections.keys())

    # Encontre nós que não têm antecessor, mas têm sucessor.
    ini_inter_nodes = no_pred.intersection(with_succ)
    # Esses nós são os que linkam com o nó inicial branco

    # Encontre nós que não tem sucessor, mas que têm antecessor.
    end_inter_nodes = no_succ.intersection(with_pred)
    # Esses nós são os que linkam com o nó final verde

    # adiciona nos nós intermediarios iniciais, que ele esta conectado pelo nó final
    for node in ini_inter_nodes:
        if node not in connections:
            connections[node] = []
        connections[node].append(0)

    # cria o nó final os nós intermediarios finais
    for node in end_inter_nodes:
        if num_intermediate_nodes+1 not in connections:
            connections[num_intermediate_nodes+1] = []
        connections[num_intermediate_nodes+1].append(node)

    return connections


def binary_to_unet(bin_list):

    # Separando os genes de operação e conexão dos 7 blocos
    block_genes = [bin_list[i:i+14] for i in range(0, len(bin_list), 14)]

    # Mapeando os IDs das operações para suas sequências
    op_seq_map = {
        0: "3x3 conv → ReLU",
        1: "3x3 conv → Mish",
        2: "3x3 conv → IN → ReLU",
        3: "3x3 conv → IN → Mish",
        4: "5x5 conv → ReLU",
        5: "5x5 conv → Mish",
        6: "5x5 conv → IN → ReLU",
        7: "5x5 conv → IN → Mish",
        8: "ReLU → 3x3 conv",
        9: "Mish → 3x3 conv",
        10: "IN → ReLU → 3x3 conv",
        11: "IN → Mish → 3x3 conv",
        12: "ReLU → 5x5 conv",
        13: "Mish → 5x5 conv",
        14: "IN → ReLU → 5x5 conv",
        15: "IN → Mish → 5x5 conv"
    }

    unet_architecture = []

    for block_gene in block_genes:
        # Extraindo o gene de operação e o gene de conexão
        operation_gene = block_gene[:4]
        connection_gene = block_gene[4:]

        # Convertendo o gene de operação para a sequência de operações correspondente
        operation_id = int("".join(map(str, operation_gene)), 2)
        operation_sequence = operation_id #op_seq_map[operation_id]

        # Processando o gene de conexão
        connections = create_connections(connection_gene)

        # Adicionando o bloco atual à arquitetura U-Net
        unet_architecture.append({"operation_sequence":operation_sequence, "connections":connections})

    return unet_architecture

def unetToGraph(unet):

    op_sequence_list = []
    connections_list = []
    for dict in unet:
        connections_list.append(dict["connections"])
        op_sequence_list.append(dict["operation_sequence"])

    graphs = []
    for connections in connections_list:
        G = nx.DiGraph()
        for key, listNode in connections.items():
            for node in listNode:
                G.add_edge(str(node),str(key))
        graphs.append(G)

    for G in graphs:
        # nx.draw_planar(G, with_labels=True, font_weight="bold")
        pos = nx.planar_layout(G)
        nx.draw(G, pos=pos, with_labels=True, font_weight="bold")
        plt.title("teste")
        plt.show()



# bin_list = [0, 1, 1, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1]
# unet_architecture = binary_to_unet(bin_list)

# for dict in unet_architecture:
#     print(dict["operation_sequence"])
#     print(dict["connections"])

# unetToGraph(unet_architecture)
# print(create_connections("0100000011"))


"""
5x5 conv → IN → Mish
{3: [2], 4: [1], 5: [1, 4], 1: [0], 2: [0], 6: [3, 5]}
3x3 conv → IN → Mish
{2: [1], 3: [2], 4: [1], 5: [2, 3], 1: [0], 6: [4, 5]}
5x5 conv → IN → Mish
{2: [1], 3: [1], 4: [1], 5: [1, 2, 3, 4], 1: [0], 6: [5]}
5x5 conv → IN → Mish
{4: [1], 5: [2, 3, 4], 1: [0], 2: [0], 3: [0], 6: [5]}
5x5 conv → IN → Mish
{2: [1], 4: [3], 5: [3, 4], 1: [0], 3: [0], 6: [2, 5]}
Mish → 5x5 conv
{2: [1], 4: [1, 2, 3], 1: [0], 3: [0], 6: [4]}
ReLU → 3x3 conv
{2: [1], 3: [1], 4: [3], 5: [1, 2, 4], 1: [0], 6: [5]}
"""