import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
from pulp import *

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

# --- Dados do problema ---
demanda_portos = np.array([0, 2.5, 1.8, 3.0, 2.0, 1.5, 2.2, 1.9, 3.1, 2.4, 1.7, 2.6, 2.8, 1.4])

# --- Ajuste dos calados para garantir viabilidade ---
carga_total = demanda_portos.sum()
calado_portos_base = np.full(n, carga_total)
calado_portos_base[0] = carga_total  # Porto 0 aceita carga total

# --- Função para aplicar variação de calado ---
def aplicar_variacao_calado(calado_base, percentual_reducao, reducao_metros, seed=None):
    calado_modificado = calado_base.copy()
    n_portos = len(calado_base)
    n_reduzir = int(np.ceil(n_portos * percentual_reducao))
    if seed is not None:
        random.seed(seed)
    portos_para_reduzir = random.sample(range(1, n_portos), n_reduzir) if n_reduzir > 0 else []
    for p in portos_para_reduzir:
        calado_modificado[p] = max(0, calado_modificado[p] - reducao_metros)
    return calado_modificado, portos_para_reduzir

# --- Escolha do cenário de variação de calado ---

percentual_reducao = 0.1   # 10% dos portos
reducao_metros = 5        # Redução de 10 unidades 
seed = 42                   # Para reprodutibilidade

calado_portos, portos_reduzidos = aplicar_variacao_calado(calado_portos_base, percentual_reducao, reducao_metros, seed=seed)

print(f"Carga total: {carga_total:.2f}")
print(f"Portos com calado reduzido ({percentual_reducao*100:.0f}%): {portos_reduzidos}")
print(f"Calado dos portos: {calado_portos}")

# --- MODELO PuLP ---
prob = LpProblem("PCVLC", LpMinimize)

x = LpVariable.dicts("x", ((i, j) for i in range(n) for j in range(n) if i != j), cat='Binary')
y = LpVariable.dicts("y", ((i, j) for i in range(n) for j in range(n) if i != j), lowBound=0, cat='Continuous')

# Objetivo
prob += lpSum(cost_matrix[i][j] * x[i, j] for i in range(n) for j in range(n) if i != j)

# Grau de entrada e saída
for k in range(n):
    prob += lpSum(x[i, k] for i in range(n) if i != k) == 1
    prob += lpSum(x[k, j] for j in range(n) if j != k) == 1

# Fluxo de carga
for j in range(1, n):
    prob += lpSum(y[i, j] for i in range(n) if i != j) - lpSum(y[j, k] for k in range(n) if k != j) == demanda_portos[j]

# Carga inicial e final
prob += lpSum(y[0, j] for j in range(1, n)) == carga_total
prob += lpSum(y[i, 0] for i in range(1, n)) == 0

# Restrição de calado
for i in range(n):
    for j in range(n):
        if i != j:
            prob += y[i, j] <= calado_portos[j] * x[i, j]

# Resolver
prob.solve(PULP_CBC_CMD(timeLimit=300))

if prob.status == LpStatusOptimal:
    rota = [(i, j) for i in range(n) for j in range(n) if i != j and x[i, j].varValue > 0.5]
    custo_total = value(prob.objective)

    # Reconstruir a rota sequencial a partir das arestas
    rota_dict = {i: j for i, j in rota}
    caminho = [0]
    atual = 0
    while True:
        proximo = rota_dict.get(atual)
        if proximo is None or proximo == 0:
            break
        caminho.append(proximo)
        atual = proximo
    caminho.append(0)  # Fecha o ciclo

    print("Rota ótima sequencial:", caminho)
    print(f"Custo total da rota: {custo_total:.2f}")

    print("Etapas da rota e custos:")
    custo_acumulado = 0
    for i in range(len(caminho) - 1):
        origem = caminho[i]
        destino = caminho[i + 1]
        custo = cost_matrix[origem][destino]
        # Mostra também o calado e a carga transportada
        carga = y[(origem, destino)].varValue if (origem, destino) in y else 0
        print(f"{origem} -> {destino} | custo: {custo} | calado destino: {calado_portos[destino]:.2f} | carga: {carga:.2f}")
        custo_acumulado += custo
    print(f"Custo acumulado calculado: {custo_acumulado}")

    # --- Plot ---
    G = nx.DiGraph()
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

    # Todas as arestas possíveis em cinza claro
    all_edges = [(i, j) for i in range(n) for j in range(n) if i != j and cost_matrix[i][j] > 0]
    nx.draw_networkx_edges(G, pos, edgelist=all_edges, edge_color='lightgray', style='dotted', arrows=True, alpha=0.5)

    # Rótulos de custo para todas as arestas
    all_edge_labels = {(i, j): f"{cost_matrix[i][j]:.0f}" for i, j in all_edges}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=all_edge_labels, font_size=8, font_color='gray', label_pos=0.7)

    # Destacar as arestas da solução ótima em vermelho
    nx.draw_networkx_edges(G, pos, edgelist=rota, edge_color='red', width=3, arrows=True)

    plt.title("Rota Otimizada via PLI (PCVLC) com PuLP", fontsize=18)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

else:
    print("\nSolução não encontrada.")