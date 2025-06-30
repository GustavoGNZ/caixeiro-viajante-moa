import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

# --- Carregar o arquivo XML ---
tree = ET.parse("burma14.xml")  # Substitua pelo caminho do seu arquivo, se necessário
root = tree.getroot()

# --- Extrair vértices e matriz de custos ---
vertices = root.find("graph").findall("vertex")
n = len(vertices)
cost_matrix = np.zeros((n, n))

for i, vertex in enumerate(vertices):
    for edge in vertex.findall("edge"):
        j = int(edge.text)
        cost = float(edge.attrib["cost"])
        cost_matrix[i][j] = cost

# --- Mostrar matriz de custos em tabela ---
cost_df = pd.DataFrame(cost_matrix.astype(int), columns=[f"To {j}" for j in range(n)], index=[f"From {i}" for i in range(n)])
print("📋 Matriz de Custos (de cada cidade para cada cidade):")
print(cost_df)

# --- Criar grafo não direcionado (bidirecional) ---
G = nx.Graph()
G.add_nodes_from(range(n))

for i in range(n):
    for j in range(n):
        if i != j and cost_matrix[i][j] > 0:
            G.add_edge(i, j, weight=cost_matrix[i][j])

# --- Layout com mais espaçamento ---
pos = nx.spring_layout(G, seed=42, k=2.5)

# --- Desenho do grafo ---
plt.figure(figsize=(16, 12))
nx.draw_networkx_nodes(G, pos, node_size=1200, node_color='skyblue')
nx.draw_networkx_labels(G, pos, font_size=14, font_weight='bold')
nx.draw_networkx_edges(G, pos, connectionstyle='arc3,rad=0.1')  # Sem setas

# Rótulos de arestas (em vermelho)
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(
    G,
    pos,
    edge_labels={k: f"{v:.0f}" for k, v in edge_labels.items()},
    font_size=10,
    font_color='black'
)

plt.title("Grafo  da Instância BURMA14 (TSPLIB)", fontsize=18)
plt.axis("off")
plt.tight_layout()
plt.show()
