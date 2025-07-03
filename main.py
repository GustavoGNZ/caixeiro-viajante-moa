import os
import json
import time
import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import random
from pulp import *
import matplotlib.patches as mpatches

def ler_matriz_custos_do_xml(caminho_arquivo):
    tree = ET.parse(caminho_arquivo)
    root = tree.getroot()
    vertices = root.find("graph").findall("vertex")
    n = len(vertices)
    cost_matrix = np.zeros((n, n))
    for i, vertex in enumerate(vertices):
        for edge in vertex.findall("edge"):
            j = int(edge.text)
            cost = float(edge.attrib["cost"])
            cost_matrix[i][j] = cost
    return cost_matrix

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

def criar_e_resolver_modelo(cost_matrix, demanda_portos, calado_portos):
    n = len(cost_matrix)
    carga_total = demanda_portos.sum()

    prob = LpProblem("PCVLC", LpMinimize)
    x = LpVariable.dicts("x", ((i, j) for i in range(n) for j in range(n) if i != j), cat='Binary')
    y = LpVariable.dicts("y", ((i, j) for i in range(n) for j in range(n) if i != j), lowBound=0, cat='Continuous')

    prob += lpSum(cost_matrix[i][j] * x[i, j] for i in range(n) for j in range(n) if i != j)

    for k in range(n):
        prob += lpSum(x[i, k] for i in range(n) if i != k) == 1
        prob += lpSum(x[k, j] for j in range(n) if j != k) == 1

    for j in range(1, n):
        prob += lpSum(y[i, j] for i in range(n) if i != j) - lpSum(y[j, k] for k in range(n) if k != j) == demanda_portos[j]

    prob += lpSum(y[0, j] for j in range(1, n)) == carga_total
    prob += lpSum(y[i, 0] for i in range(1, n)) == 0

    for i in range(n):
        for j in range(n):
            if i != j:
                prob += y[i, j] <= calado_portos[j] * x[i, j]

    prob.solve(PULP_CBC_CMD(timeLimit=300))
    return prob, x, y

def reconstruir_rota(x, n):
    rota = [(i, j) for i in range(n) for j in range(n) if i != j and x[i, j].varValue > 0.5]
    rota_dict = {i: j for i, j in rota}
    caminho = [0]
    atual = 0
    while True:
        proximo = rota_dict.get(atual)
        if proximo is None or proximo == 0:
            break
        caminho.append(proximo)
        atual = proximo
    caminho.append(0)
    return rota, caminho

def plotar_e_salvar_grafo(nome_instancia, n, cost_matrix, rota, caminho, calado_portos, y, portos_reduzidos, pasta_imagens, percentual_reducao):
    G_completo = nx.DiGraph()
    G_otimo = nx.DiGraph()

    all_edges = [(i, j) for i in range(n) for j in range(n) if i != j and cost_matrix[i][j] > 0]
    G_completo.add_nodes_from(range(n))
    G_completo.add_edges_from(all_edges)

    G_otimo.add_nodes_from(range(n))
    G_otimo.add_edges_from(rota)

    pos = nx.kamada_kawai_layout(G_completo)
    node_colors = ['lightgreen' if i in portos_reduzidos else 'skyblue' for i in range(n)]
    node_labels = {i: f"{i}\nC:{calado_portos[i]:.1f}" for i in range(n)}

    plt.figure(figsize=(24, 10), dpi=120)

    # Grafo da Solução Ótima
    plt.subplot(1, 2, 1)
    nx.draw_networkx_nodes(G_otimo, pos, node_color=node_colors, node_size=1000, edgecolors='black')
    nx.draw_networkx_labels(G_otimo, pos, labels=node_labels, font_size=9, font_weight='bold')

    edge_colors = []
    edge_widths = []
    for (i, j) in rota:
        carga = y[(i, j)].varValue if (i, j) in y else 0
        edge_colors.append('red')
        edge_widths.append(1 + carga / 10)

    nx.draw_networkx_edges(
        G_otimo, pos, edgelist=rota, edge_color=edge_colors, width=edge_widths,
        arrows=True, arrowsize=15, arrowstyle='->', connectionstyle='arc3,rad=0.1'
    )
    plt.title(f"{nome_instancia} - Solução Ótima")

    # Grafo Completo
    plt.subplot(1, 2, 2)
    nx.draw_networkx_nodes(G_completo, pos, node_color=node_colors, node_size=1000, edgecolors='black')
    nx.draw_networkx_labels(G_completo, pos, labels=node_labels, font_size=9, font_weight='bold')
    nx.draw_networkx_edges(
        G_completo, pos, edge_color='lightgray', width=0.6, style='dotted',
        arrows=True, arrowsize=10, arrowstyle='->', connectionstyle='arc3,rad=0.1'
    )
    plt.title(f"{nome_instancia} - Grafo Completo")

    normal_patch = mpatches.Patch(color='skyblue', label='Calado normal')
    reduzido_patch = mpatches.Patch(color='lightgreen', label='Calado reduzido')
    plt.legend(handles=[normal_patch, reduzido_patch], loc='lower center', bbox_to_anchor=(-0.1, -0.15), ncol=2, fontsize=10, frameon=True)

    plt.tight_layout()

    os.makedirs(pasta_imagens, exist_ok=True)
    nome_arquivo_imagem = os.path.join(
        pasta_imagens,
        f"{nome_instancia}_grafico_{int(percentual_reducao * 100)}.png"
    )
    plt.savefig(nome_arquivo_imagem)
    plt.close()

import time
import json
import os

if __name__ == "__main__":
    pasta_instancias = "instancias"
    pasta_imagens = "imagens"
    pasta_resultados = "resultados"

    os.makedirs(pasta_imagens, exist_ok=True)
    os.makedirs(pasta_resultados, exist_ok=True)

    arquivos_xml = [f for f in os.listdir(pasta_instancias) if f.endswith(".xml")]

    for arquivo in arquivos_xml:
        nome_instancia = os.path.splitext(arquivo)[0]
        caminho_arquivo = os.path.join(pasta_instancias, arquivo)
        cost_matrix = ler_matriz_custos_do_xml(caminho_arquivo)
        n = cost_matrix.shape[0]

        np.random.seed(42)
        demandas = np.random.uniform(1.5, 3.5, size=n)
        demandas[0] = 0  # depósito
        demanda_portos = demandas
        carga_total = demanda_portos.sum()
        calado_portos_base = np.full(n, carga_total)
        calado_portos_base[0] = carga_total

        percentual_reducao = 0.5 
        reducao_metros = 5
        seed = 42

        calado_portos, portos_reduzidos = aplicar_variacao_calado(calado_portos_base, percentual_reducao, reducao_metros, seed=seed)

        start = time.perf_counter()
        prob, x, y = criar_e_resolver_modelo(cost_matrix, demanda_portos, calado_portos)
        end = time.perf_counter()
        tempo_resolucao = end - start

        resultado = {
            "status": LpStatus[prob.status],
            "rota_otima": [],
            "custo_total": None,
            "portos_calado_reduzido": portos_reduzidos,
            "calado_portos": calado_portos.tolist(),
            "tempo_resolucao_segundos": tempo_resolucao
        }

        if prob.status == LpStatusOptimal:
            rota, caminho = reconstruir_rota(x, n)
            custo_total = value(prob.objective)

            resultado["rota_otima"] = caminho
            resultado["custo_total"] = custo_total

            # Nome base com percentual para arquivos
            nome_base_com_percentual = f"{nome_instancia}_grafico_{int(percentual_reducao * 100)}"

            plotar_e_salvar_grafo(
                nome_instancia, n, cost_matrix, rota, caminho, calado_portos, y, portos_reduzidos, pasta_imagens, percentual_reducao
            )
            print(f"Instância {nome_instancia}: solução ótima encontrada, custo = {custo_total:.2f}, tempo = {tempo_resolucao:.2f}s")
        else:
            print(f"Instância {nome_instancia}: solução não encontrada, status = {LpStatus[prob.status]}")

            # Mesmo nome base mesmo que sem solução
            nome_base_com_percentual = f"{nome_instancia}_grafico_{int(percentual_reducao * 100)}"

        # Salvar JSON com nome igual da imagem (com percentual)
        caminho_json = os.path.join(pasta_resultados, f"{nome_base_com_percentual}.json")
        with open(caminho_json, "w") as f:
            json.dump(resultado, f, indent=4)
