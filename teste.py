import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

# --- Carregar o arquivo XML ---
# Substitua pelo caminho do seu arquivo, se necessário
tree = ET.parse("burma14.xml")
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

# --------------------------------------------------------------------------
# ETAPA 1: ADAPTAR A ESTRUTURA DE DADOS PARA O PCVLC
# --------------------------------------------------------------------------


def adaptar_para_pcvlc(num_portos, restriction_level=0.25):
    """
    Cria a estrutura de dados dos portos com demanda e limite de calado.
    - [cite_start]Demanda de carga para cada porto de destino[cite: 56].
    - [cite_start]Limite de calado para uma porcentagem dos portos[cite: 134].
    """
    portos = []
    # Gera demandas aleatórias para os portos de destino (1 a n-1)
    demandas = np.random.randint(10, 41, size=num_portos)
    demandas[0] = 0  # Porto de origem não tem demanda

    carga_total = np.sum(demandas)

    for i in range(num_portos):
        portos.append({
            "id": i,
            "demanda": demandas[i],
            "limite_calado": float('inf')  # Padrão: sem limite
        })

    # Define o limite de calado para uma porcentagem dos portos
    num_portos_restritos = int((num_portos - 1) * restriction_level)
    portos_destino_ids = list(range(1, num_portos))
    portos_restritos_ids = np.random.choice(
        portos_destino_ids, size=num_portos_restritos, replace=False)

    print(f"Adaptação para PCVLC ({restriction_level*100}% de restrição):")
    print(f"Portos com limite de calado: {sorted(list(portos_restritos_ids))}")

    for porto_id in portos_restritos_ids:
        # O limite deve ser um desafio: menor que a carga total, mas não trivial.
        limite = carga_total * np.random.uniform(0.7, 0.95)
        portos[porto_id]["limite_calado"] = int(limite)

    return portos

# --------------------------------------------------------------------------
# ETAPA 2: CRIAR A FUNÇÃO DE VALIDAÇÃO DA ROTA
# --------------------------------------------------------------------------


def validar_e_calcular_custo(rota, portos, cost_matrix):
    """
    Verifica se uma rota é válida segundo as regras do PCVLC e calcula seu custo.
    - [cite_start]O navio sai com a carga total e vai descarregando[cite: 57].
    - [cite_start]A carga atual não pode exceder o limite do próximo porto[cite: 58].
    """
    # Validação básica da rota
    if rota[0] != 0 or rota[-1] != 0 or len(set(rota[1:-1])) != len(portos) - 1:
        print("ERRO: Rota inválida. Deve ser um circuito Hamiltoniano começando e terminando em 0.")
        return float('inf'), False

    carga_atual = sum(p["demanda"] for p in portos)
    custo_total = 0

    print(f"\nValidando Rota: {rota} | Carga Inicial: {carga_atual}")

    for i in range(len(rota) - 1):
        porto_origem_id = rota[i]
        porto_destino_id = rota[i+1]

        limite_porto_destino = portos[porto_destino_id]["limite_calado"]

        # A verificação CRUCIAL do PCVLC
        if carga_atual > limite_porto_destino:
            print(
                f"  ❌ ROTA INVÁLIDA no trecho {porto_origem_id} -> {porto_destino_id}")
            print(
                f"     Carga Atual ({carga_atual}) > Limite do Porto {porto_destino_id} ({limite_porto_destino})")
            return float('inf'), False

        # Se a rota é válida até aqui, atualiza custo e carga
        custo_etapa = cost_matrix[porto_origem_id][porto_destino_id]
        custo_total += custo_etapa
        demanda_descarregada = portos[porto_destino_id]["demanda"]
        carga_atual -= demanda_descarregada

        print(
            f"  ✅ Trecho {porto_origem_id} -> {porto_destino_id} (Custo: {custo_etapa:.0f}). Carga restante: {carga_atual}")

    return custo_total, True


# --- EXECUTANDO A ADAPTAÇÃO E VALIDAÇÃO ---
# 1. Adaptar a instância `burma14` para PCVLC com 25% de portos restritos
portos_pcvlc = adaptar_para_pcvlc(n, restriction_level=0.25)
print("\n📋 Dados dos Portos (PCVLC):")
print(pd.DataFrame(portos_pcvlc).set_index('id'))

# 2. Definir rotas de teste
# Rota que provavelmente será válida (visita portos em ordem, descarregando aos poucos)
rota_valida_provavel = list(range(n)) + [0]
# Rota que provavelmente será inválida (tenta visitar um porto restrito no início)
portos_restritos = [p['id']
                    for p in portos_pcvlc if p['limite_calado'] != float('inf')]
if portos_restritos:
    primeiro_restrito = portos_restritos[0]
    outros_portos = [p for p in range(1, n) if p != primeiro_restrito]
    rota_invalida_provavel = [0, primeiro_restrito] + outros_portos + [0]
else:
    rota_invalida_provavel = None  # Caso não haja portos restritos

# 3. Validar as rotas
custo, eh_valida = validar_e_calcular_custo(
    rota_valida_provavel, portos_pcvlc, cost_matrix)
if eh_valida:
    print(f"\n🎉 Rota Válida! Custo Total: {custo:.2f}\n")

if rota_invalida_provavel:
    custo, eh_valida = validar_e_calcular_custo(
        rota_invalida_provavel, portos_pcvlc, cost_matrix)
    if not eh_valida:
        print("\n🎉 Teste de Rota Inválida bem-sucedido!\n")

# --------------------------------------------------------------------------
# SEU CÓDIGO ORIGINAL DE VISUALIZAÇÃO (sem alterações)
# --------------------------------------------------------------------------

# --- Mostrar matriz de custos em tabela ---
cost_df = pd.DataFrame(cost_matrix.astype(int), columns=[
                       f"To {j}" for j in range(n)], index=[f"From {i}" for i in range(n)])
print("\n📋 Matriz de Custos Original (de cada cidade para cada cidade):")
print(cost_df)

# --- Criar e desenhar o grafo ---
G = nx.from_numpy_array(cost_matrix)
pos = nx.spring_layout(G, seed=42, k=2.5)

plt.figure(figsize=(16, 12))
nx.draw_networkx_nodes(G, pos, node_size=1200, node_color='skyblue')
nx.draw_networkx_labels(G, pos, font_size=14, font_weight='bold')
nx.draw_networkx_edges(G, pos, connectionstyle='arc3,rad=0.1')
edge_labels = nx.get_edge_attributes(G, 'weight')
nx.draw_networkx_edge_labels(G, pos, edge_labels={
                             k: f"{v:.0f}" for k, v in edge_labels.items()}, font_size=10, font_color='black')
plt.title("Grafo da Instância BURMA14 (TSPLIB)", fontsize=18)
plt.axis("off")
plt.tight_layout()
plt.show()
