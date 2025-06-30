import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random

# --- Leitura do XML ---
tree = ET.parse("burma14.xml")
root = tree.getroot()

vertices = root.find("graph").findall("vertex")
n = len(vertices)
cost_matrix = np.zeros((n, n))

for i, vertex in enumerate(vertices):
    for edge in vertex.findall("edge"):
        j = int(edge.text)
        cost = float(edge.attrib["cost"])
        cost_matrix[i][j] = cost

# --- Dados iniciais (exemplo) ---
# Demanda em cada porto (exemplo simples, pode adaptar)
demanda_portos = np.array([0, 2.5, 1.8, 3.0, 2.0, 1.5, 2.2, 1.9, 3.1, 2.4, 1.7, 2.6, 2.8, 1.4])
# Limite de calado em metros para cada porto
calado_portos_base = np.array([9.0, 8.0, 7.5, 9.0, 7.0, 8.2, 8.5, 7.3, 8.0, 8.1, 7.8, 7.9, 8.3, 7.2])

# Carga inicial total do navio (soma das demandas)
carga_inicial = demanda_portos.sum()
calado_navio_max = 9.0  # calado máximo inicial do navio (com carga total)

# --- Função para aplicar variação no limite de calado ---
def aplicar_variacao_calado(calado_base, percentual_reducao, reducao_metros):
    calado_modificado = calado_base.copy()
    n_portos = len(calado_base)
    n_reduzir = int(np.ceil(n_portos * percentual_reducao))
    portos_para_reduzir = random.sample(range(n_portos), n_reduzir)
    for p in portos_para_reduzir:
        calado_modificado[p] = max(0, calado_modificado[p] - reducao_metros)
    return calado_modificado, portos_para_reduzir

# --- Função que calcula calado do navio baseado na carga atual (linear) ---
def calcular_calado_navio(carga_atual, carga_total, calado_max):
    # Exemplo simples: calado decresce proporcionalmente à carga
    return (carga_atual / carga_total) * calado_max

# --- Função para construir rota respeitando calado variável e visita todos portos ---
def construir_rota_calado_variavel(cost_matrix, calado_portos, demanda_portos, calado_navio_max):
    n = len(calado_portos)
    carga_total = demanda_portos.sum()
    carga_atual = carga_total
    porto_atual = 0
    rota = [porto_atual]
    portos_restantes = set(range(1, n))
    
    while portos_restantes:
        calado_navio = calcular_calado_navio(carga_atual, carga_total, calado_navio_max)
        
        # Portos onde navio pode atracar com calado atual
        portos_validos = [p for p in portos_restantes if calado_navio <= calado_portos[p]]
        if not portos_validos:
            print("Não há portos válidos para calado atual:", calado_navio)
            break
        
        # Escolhe o porto válido mais próximo
        proximo = min(portos_validos, key=lambda p: cost_matrix[porto_atual][p])
        rota.append(proximo)
        portos_restantes.remove(proximo)
        
        # Descarrega carga neste porto
        carga_atual -= demanda_portos[proximo]
        if carga_atual < 0:
            carga_atual = 0
        porto_atual = proximo
    
    rota.append(0)  # retorna ao porto inicial
    return rota

# --- Exemplo: aplicar variação de calado em 10% dos portos, reduzindo 0.5m ---
percentual_reducao = 0.10
reduz_metros = 0.5
calado_portos, portos_reduzidos = aplicar_variacao_calado(calado_portos_base, percentual_reducao, reduz_metros)

print(f"Portos com calado reduzido ({percentual_reducao*100:.0f}%): {portos_reduzidos}")
print(f"Limites de calado após variação:\n{calado_portos}")

# --- Construir rota considerando calado variável ---
rota = construir_rota_calado_variavel(cost_matrix, calado_portos, demanda_portos, calado_navio_max)

# --- Calcular custo total da rota ---
custo_total = 0
for i in range(len(rota)-1):
    custo_total += cost_matrix[rota[i]][rota[i+1]]

print("\nRota construída respeitando calado variável:")
print(rota)
print(f"Custo total da rota: {custo_total:.2f}")

# --- Plotar o grafo e a rota ---
G = nx.Graph()
G.add_nodes_from(range(n))
for i in range(n):
    for j in range(n):
        if i != j and cost_matrix[i][j] > 0:
            G.add_edge(i, j, weight=cost_matrix[i][j])

pos = nx.spring_layout(G, seed=42, k=2.5)

plt.figure(figsize=(16, 12))
nx.draw_networkx_nodes(G, pos, node_size=1200,
                       node_color=['lightgreen' if i in portos_reduzidos else 'skyblue' for i in range(n)])
nx.draw_networkx_labels(G, pos, font_size=14, font_weight='bold')
nx.draw_networkx_edges(G, pos, edge_color='lightgray')

edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(
    G, pos,
    edge_labels={k: f"{v:.0f}" for k, v in edge_labels.items()},
    font_size=10,
    font_color='black'
)

edges_rota = [(rota[i], rota[i+1]) for i in range(len(rota)-1)]
nx.draw_networkx_edges(G, pos, edgelist=edges_rota, edge_color='red', width=3)

plt.title(f"Rota PCVLC com {percentual_reducao*100:.0f}% portos reduzidos em calado", fontsize=18)
plt.axis("off")
plt.tight_layout()
plt.show()
