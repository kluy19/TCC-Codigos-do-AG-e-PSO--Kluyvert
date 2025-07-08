import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from geneticalgorithm import geneticalgorithm as ga
from pyswarms.single.global_best import GlobalBestPSO
import time
import networkx as nx

# === Parâmetros reais do TCC ===
num_linhas = 25  # 25 linhas para simular mais faltas
custo_fusivel = 1000.0
custo_religador = 15000.0
tarifa_energia = 736  # R$/kWh
tempo_planejamento = 10  # anos

# Fatores reais
taxa_falha_permanente = 0.072
taxa_falha_temporaria = 0.98
tempo_reparo_permanente = 4.0  # horas
tempo_reparo_temporario = 0.17  # horas

# Comprimento e carga por linha (uniforme para simplificar)
comprimento_linhas = np.ones(num_linhas) * 0.5  # km
print(comprimento_linhas)
carga_linhas = np.ones(num_linhas) * 1.0        # kW
print(carga_linhas)
# Redução das falhas pelos dispositivos
fator_fusivel = 0.8
fator_religador = 0.5

# Falhas por linha - 25 linhas, valores variados simulando casos reais
falhas_linha = np.array([
    5, 4, 6, 3, 80, 5, 6, 4, 5, 6,
    3, 4, 0, 8, 10, 12, 15, 9, 11, 13,
    7, 8, 9, 5, 6
])

# Nomes das linhas
nomes_linhas = [f"Line.{1 + i}" for i in range(num_linhas)]

# === Função de custo ===
def custo_total(solucao):
    total = 0.0
    custos_por_linha = []

    for i in range(num_linhas):
        tipo = int(np.clip(solucao[i], 0, 2))
        carga = carga_linhas[i]
        falhas = falhas_linha[i]

        if tipo == 0:
            custo_fixo = 0
            fator = 1.0
        elif tipo == 1:
            custo_fixo = custo_fusivel
            fator = fator_fusivel
        elif tipo == 2:
            custo_fixo = custo_religador
            fator = fator_religador

        custo_falhas = falhas * fator * carga * tarifa_energia * (
            taxa_falha_permanente * tempo_reparo_permanente +
            taxa_falha_temporaria * tempo_reparo_temporario
        ) * tempo_planejamento

        total += custo_fixo + custo_falhas
        custos_por_linha.append((custo_fixo, custo_falhas))

    return total, custos_por_linha

# === Criação de planilhas Excel com melhor organização ===
def criar_planilhas_excel(nome_arquivo_resumo, nome_arquivo_detalhes,
                          solucao_ag, custos_ag,
                          solucao_pso, custos_pso):

    # Custo sem dispositivo (baseline)
    custo_falha_sem = []
    for i in range(num_linhas):
        carga = carga_linhas[i]
        falhas = falhas_linha[i]
        custo = falhas * carga * tarifa_energia * (
            taxa_falha_permanente * tempo_reparo_permanente +
            taxa_falha_temporaria * tempo_reparo_temporario
        ) * tempo_planejamento
        custo_falha_sem.append(custo)
    custo_falha_sem = np.array(custo_falha_sem)

    # Resumo por algoritmo
    custo_fixo_ag = np.array([x[0] for x in custos_ag])
    custo_falhas_ag = np.array([x[1] for x in custos_ag])
    com_dispositivo_ag = custo_fixo_ag + custo_falhas_ag

    custo_fixo_pso = np.array([x[0] for x in custos_pso])
    custo_falhas_pso = np.array([x[1] for x in custos_pso])
    com_dispositivo_pso = custo_fixo_pso + custo_falhas_pso

    # Planilha Resumo - Custo total global
    df_resumo = pd.DataFrame({
        'Método': ['Sem dispositivo', 'Com Fusível/Religador (AG)', 'Com Fusível/Religador (PSO)'],
        'Custo Total (R$)': [
            custo_falha_sem.sum(),
            com_dispositivo_ag.sum(),
            com_dispositivo_pso.sum()
        ]
    })

    df_resumo.to_excel(nome_arquivo_resumo, index=False)

    # Planilha Detalhada - por linha, custos e dispositivo escolhido
    dispositivos = ["Nenhum", "Fusível", "Religador"]

    linhas = []
    for i in range(num_linhas):
        linhas.append({
            'Linha': nomes_linhas[i],
            'Falhas': falhas_linha[i],
            'Custo sem dispositivo (R$)': custo_falha_sem[i],

            'Dispositivo AG': dispositivos[int(solucao_ag[i])],
            'Custo fixo AG (R$)': custo_fixo_ag[i],
            'Custo falhas AG (R$)': custo_falhas_ag[i],
            'Custo total AG (R$)': com_dispositivo_ag[i],

            'Dispositivo PSO': dispositivos[int(solucao_pso[i])],
            'Custo fixo PSO (R$)': custo_fixo_pso[i],
            'Custo falhas PSO (R$)': custo_falhas_pso[i],
            'Custo total PSO (R$)': com_dispositivo_pso[i],
        })

    df_detalhes = pd.DataFrame(linhas)
    df_detalhes.to_excel(nome_arquivo_detalhes, index=False)
    print(f"Arquivos Excel '{nome_arquivo_resumo}' e '{nome_arquivo_detalhes}' gerados com sucesso!")

# === Algoritmo Genético ===
evolucao_ag = []

def func_ag(X):
    custo, _ = custo_total(X)
    evolucao_ag.append(custo)
    return custo

varbound = np.array([[0, 2]] * num_linhas)
algorithm_param = {
    'max_num_iteration': 300,
    'population_size': 80,
    'mutation_probability': 0.1,
    'elit_ratio': 0.01,
    'crossover_probability': 0.5,
    'parents_portion': 0.3,
    'crossover_type': 'uniform',
    'max_iteration_without_improv': None
}

print("=== Rodando AG ===")
start_ag = time.time()
model = ga(function=func_ag,
           dimension=num_linhas,
           variable_type='int',
           variable_boundaries=varbound,
           algorithm_parameters=algorithm_param)
model.run()
end_ag = time.time()
tempo_ag = end_ag - start_ag

# === PSO ===
evolucao_pso = []

def func_pso(x):
    custos = []
    for row in x:
        custo = custo_total(np.rint(row).astype(int))[0]
        custos.append(custo)
    evolucao_pso.append(min(custos))
    return np.array(custos)

bounds = (np.zeros(num_linhas), np.full(num_linhas, 2))

print("\n=== Rodando PSO ===")
start_pso = time.time()
optimizer = GlobalBestPSO(n_particles=30, dimensions=num_linhas,
                          options={'c1': 1.5, 'c2': 1.5, 'w': 0.6},
                          bounds=bounds)
_, best_position = optimizer.optimize(func_pso, iters=160, verbose=False)
end_pso = time.time()
tempo_pso = end_pso - start_pso

melhor_solucao_pso = np.rint(best_position).astype(int)
melhor_solucao_ag = model.output_dict['variable'].astype(int)

custo_ag, custos_ag_por_linha = custo_total(melhor_solucao_ag)
custo_pso, custos_pso_por_linha = custo_total(melhor_solucao_pso)

# Criar arquivos Excel
criar_planilhas_excel("Resumo_Custos.xlsx", "Detalhamento_Custos.xlsx",
                      melhor_solucao_ag, custos_ag_por_linha,
                      melhor_solucao_pso, custos_pso_por_linha)

print(f"\nTempo de execução AG: {tempo_ag:.2f} segundos")
print(f"Tempo de execução PSO: {tempo_pso:.2f} segundos")
print(f"Melhor solução AG: {melhor_solucao_ag.tolist()}")
print(f"Custo Total AG: R${custo_ag:.2f}")
print(f"Melhor solução PSO: {melhor_solucao_pso.tolist()}")
print(f"Custo Total PSO: R${custo_pso:.2f}")

# === Impressão detalhada ===
def imprimir_configuracao(nome, solucao, custos_por_linha):
    print(f"\nDispositivos ({nome}):")
    total_fixo = 0
    total_falhas = 0
    contagem = {"Nenhum": 0, "FUSÍVEL": 0, "RELIGADOR": 0}

    for i in range(num_linhas):
        tipo = solucao[i]
        fixo, falhas = custos_por_linha[i]
        total_fixo += fixo
        total_falhas += falhas
        tipo_disp = ["Nenhum", "FUSÍVEL", "RELIGADOR"][tipo]
        contagem[tipo_disp] += 1
        print(f"{nomes_linhas[i]}: {tipo_disp}, Custo fixo = R${fixo:.2f}, Falhas = R${falhas:.2f}")

    print(f"\nResumo de dispositivos ({nome}):")
    print(f"  Nenhum: {contagem['Nenhum']}")
    print(f"  Fusível: {contagem['FUSÍVEL']}")
    print(f"  Religador: {contagem['RELIGADOR']}")
    print(f"Custo Total ({nome}) = R${total_fixo + total_falhas:.2f}")

imprimir_configuracao("AG", melhor_solucao_ag, custos_ag_por_linha)
imprimir_configuracao("PSO", melhor_solucao_pso, custos_pso_por_linha)

# === Gráficos ===
# Evolução do custo por iteração - AG
plt.figure(figsize=(10,5))
plt.plot(range(len(evolucao_ag)), evolucao_ag, label='AG', color='blue')
plt.title('Evolução do Custo - Algoritmo Genético')
plt.xlabel('Iterações')
plt.ylabel('Custo Total (R$)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Evolução do custo por iteração - PSO
plt.figure(figsize=(10,5))
plt.plot(range(len(evolucao_pso)), evolucao_pso, label='PSO', color='green')
plt.title('Evolução do Custo - PSO')
plt.xlabel('Iterações')
plt.ylabel('Custo Total (R$)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Comparação da alocação de dispositivos entre AG e PSO
dispositivos = ["Nenhum", "Fusível", "Religador"]
labels = nomes_linhas

x = np.arange(len(labels))
width = 0.35

fig, ax = plt.subplots(figsize=(15,6))
ag_disp = [dispositivos[i] for i in melhor_solucao_ag]
pso_disp = [dispositivos[i] for i in melhor_solucao_pso]

# Converter para números para plotar barras (0, 1, 2)
ag_disp_num = melhor_solucao_ag
pso_disp_num = melhor_solucao_pso

bar1 = ax.bar(x - width/2, ag_disp_num, width, label='AG', color='blue')
bar2 = ax.bar(x + width/2, pso_disp_num, width, label='PSO', color='green')

ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right')
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(dispositivos)
ax.set_ylabel('Dispositivo Alocado')
ax.set_title('Comparação da Alocação de Dispositivos por Linha (AG x PSO)')
ax.legend()
ax.grid(axis='y')
plt.tight_layout()
plt.show()

def plotar_rede(solucao, titulo):
    G = nx.Graph()

    # Criando os nós (nós 0 a 25)
    for i in range(num_linhas + 1):
        G.add_node(i)

    # Criando arestas representando as linhas entre nós consecutivos
    for i in range(num_linhas):
        G.add_edge(i, i + 1)

    # Cores por tipo de dispositivo
    cores = []
    for tipo in solucao:
        if tipo == 0:
            cores.append("gray")         # Nenhum
        elif tipo == 1:
            cores.append("orange")       # Fusível
        elif tipo == 2:
            cores.append("blue")         # Religador

    pos = {i: (i, 0) for i in range(num_linhas + 1)}  # Layout linear
    edge_colors = cores

    plt.figure(figsize=(14, 2))
    nx.draw(G, pos, with_labels=True, node_color='lightgreen',
            edge_color=edge_colors, width=4, node_size=500)
    legend_handles = [
        plt.Line2D([0], [0], color='gray', lw=4, label='Nenhum'),
        plt.Line2D([0], [0], color='orange', lw=4, label='Fusível'),
        plt.Line2D([0], [0], color='blue', lw=4, label='Religador')
    ]
    plt.legend(handles=legend_handles, loc='upper center', ncol=3)
    plt.title(f"Distribuição dos dispositivos na rede ({titulo})")
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# Plotar redes AG e PSO
plotar_rede(melhor_solucao_ag, "AG")
plotar_rede(melhor_solucao_pso, "PSO")
