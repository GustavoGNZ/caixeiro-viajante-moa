import pulp
import csv
import networkx as nx
import matplotlib.pyplot as plt

# ----- Parâmetros FIXOS para evitar problemas -----

num_portos = 5  # 0 = origem

# Custo fixo entre portos
custo = [
    [0, 20, 40, 25, 30],
    [20, 0, 15, 35, 10],
    [40, 15, 0, 20, 25],
    [25, 35, 20, 0, 30],
    [30, 10, 25, 30, 0]
]

# Demandas: portos 1 a 4 recebem carga, origem (0) não
demanda = [0, 10, 15, 20, 5]
carga_total = sum(demanda)  # 50

# Calado: todos os portos podem receber o navio com 50 de carga
calado_limite = [100, 50, 50, 50, 50]

# ----- Modelo de Otimização -----

modelo = pulp.LpProblem("PCVLC", pulp.LpMinimize)

# Variáveis
x = [[pulp.LpVariable(f"x_{i}_{j}", cat="Binary") for j in range(num_portos)] for i in range(num_portos)]
y = [[pulp.LpVariable(f"y_{i}_{j}", lowBound=0, cat="Continuous") for j in range(num_portos)] for i in range(num_portos)]

# Objetivo
modelo += pulp.lpSum(custo[i][j] * x[i][j] for i in range(num_portos) for j in range(num_portos) if i != j)

# Visita única
for j in range(num_portos):
    modelo += pulp.lpSum(x[i][j] for i in range(num_portos) if i != j) == 1
    modelo += pulp.lpSum(x[j][i] for i in range(num_portos) if i != j) == 1

# Fluxo de carga
for j in range(1, num_portos):
    modelo += (
        pulp.lpSum(y[i][j] for i in range(num_portos) if i != j) -
        pulp.lpSum(y[j][i] for i in range(num_portos) if i != j)
        == demanda[j]
    )

# Carga inicial e final
modelo += pulp.lpSum(y[0][i] for i in range(1, num_portos)) == carga_total
modelo += pulp.lpSum(y[i][0] for i in range(1, num_portos)) == 0

# Calado
for i in range(num_portos):
    for j in range(num_portos):
        if i != j:
            modelo += y[i][j] <= calado_limite[j] * x[i][j]

# Resolver
modelo.solve(pulp.PULP_CBC_CMD(msg=False))

# Verificar resultado
if pulp.LpStatus[modelo.status] != "Optimal":
    print("❌ Modelo não tem solução viável.")
    print("Demanda:", demanda)
    print("Calado:", calado_limite)
    print("Carga total:", carga_total)
    exit()

# ----- Resultado -----
print("✅ Status:", pulp.LpStatus[modelo.status])
print("Custo total da rota:", pulp.value(modelo.objective))

rotas = []
for i in range(num_portos):
    for j in range(num_portos):
        if i != j and pulp.value(x[i][j]) > 0.5:
            origem = i
            destino = j
            carga = pulp.value(y[i][j])
            custo_ij = custo[i][j]
            rotas.append([origem, destino, carga, custo_ij])
            print(f"De {origem} para {destino} - Carga: {carga:.1f}, Custo: {custo_ij}")

# ----- Exportar para CSV -----
with open("resultado_rotas.csv", "w", newline="") as arquivo:
    writer = csv.writer(arquivo)
    writer.writerow(["Origem", "Destino", "Carga", "Custo"])
    for rota in rotas:
        writer.writerow(rota)

print("\n✅ Arquivo 'resultado_rotas.csv' gerado com sucesso!")

# ----- Desenhar Grafo -----
G = nx.DiGraph()

# Adicionar nós
for i in range(num_portos):
    G.add_node(i)

# Adicionar todas as arestas possíveis (exceto laços)
for i in range(num_portos):
    for j in range(num_portos):
        if i != j:
            G.add_edge(i, j, weight=custo[i][j], all=True)

# Layout circular para visualização
pos = nx.circular_layout(G)

# Desenhar nós
nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue')

# Rótulos dos nós (número dos portos)
nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')

# Desenhar todas as arestas possíveis em cinza claro
all_edges = [(i, j) for i in range(num_portos) for j in range(num_portos) if i != j]
nx.draw_networkx_edges(G, pos, edgelist=all_edges, edge_color='lightgray', style='dotted', arrowsize=10, alpha=0.5)

# Rótulos de custo para todas as arestas possíveis
all_edge_labels = {(i, j): f"{custo[i][j]}" for i, j in all_edges}
nx.draw_networkx_edge_labels(G, pos, edge_labels=all_edge_labels, font_size=8, font_color='gray', label_pos=0.7)

# Adicionar arestas da solução ótima com rótulo da carga
sol_edges = []
sol_edge_labels = {}
for origem, destino, carga, custo_ij in rotas:
    sol_edges.append((origem, destino))
    sol_edge_labels[(origem, destino)] = f"{carga:.1f}"

# Destacar as arestas da solução ótima em azul
nx.draw_networkx_edges(G, pos, edgelist=sol_edges, edge_color='blue', width=2, arrowstyle="->", arrowsize=20)

# Rótulos das arestas da solução ótima (carga)
nx.draw_networkx_edge_labels(G, pos, edge_labels=sol_edge_labels, font_size=10, font_color='blue')

plt.title("Rota Ótima do Navio (PCVLC) e Todas as Conexões", fontsize=14)
plt.axis("off")
plt.tight_layout()
plt.show()