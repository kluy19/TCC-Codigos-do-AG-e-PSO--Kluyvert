import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import time
from deap import base, creator, tools, algorithms
from pyswarm import pso
import py_dss_interface
import networkx as nx

# === CONFIGURAÇÃO INICIAL ===
dss = py_dss_interface.DSSDLL(r"C:\\Program Files\\OpenDSS")
dss_file = r"C:\\Users\\KLUYVERT\\Desktop\\IC\\34Bus\\ieee34Mod2.dss"
dss.text(f"compile [{dss_file}]")
dss.solution_solve()

# === Dados principais ===
nomes_linhas = [
    'Line.l1', 'Line.l2a', 'Line.l2b', 'Line.l3', 'Line.l4a', 'Line.l4b', 'Line.l5', 'Line.l6', 'Line.l7', 'Line.l24',
    'Line.l8', 'Line.l9a', 'Line.l9b', 'Line.l10a', 'Line.l10b', 'Line.l11a', 'Line.l11b', 'Line.l12a', 'Line.l12b',
    'Line.l13a', 'Line.l13b', 'Line.l14a', 'Line.l14b', 'Line.l15', 'Line.l16a', 'Line.l16b', 'Line.l29a', 'Line.l29b',
    'Line.l18', 'Line.l19a', 'Line.l19b', 'Line.l21a', 'Line.l21b', 'Line.l22a', 'Line.l22b', 'Line.l23a', 'Line.l23b',
]
num_linhas = len(nomes_linhas)

# --- Carregar posições dos barramentos via CSV ---
caminho_csv = r"C:\\Users\\KLUYVERT\\Desktop\\IC\\34Bus\\IEEE34_BusXY.csv"

def carregar_posicoes_bus(caminho_csv):
    df = pd.read_csv(caminho_csv)
    df.columns = ['bus', 'x', 'y']
    posicoes = {}
    for _, row in df.iterrows():
        nome = str(row['bus']).lower()
        posicoes[nome] = (row['x'], row['y'])
    return posicoes

posicoes = carregar_posicoes_bus(caminho_csv)

# === Carregar comprimentos e cargas das linhas ===
comprimento_linhas = []
carga_linhas = []

for nome in nomes_linhas:
    dss.circuit_set_active_element(nome)
    comprimento_linhas.append(dss.lines_read_length())
    potencias = dss.cktelement_powers()
    pot_ativa_total = sum(potencias[::2])
    carga_linhas.append(abs(pot_ativa_total))

comprimento_linhas = np.array(comprimento_linhas)
carga_linhas = np.array(carga_linhas)

# === Custos e parâmetros ===
tarifa_energia = 736
tempo_planejamento = 10
custo_fusivel = 1000
custo_religador = 85000
taxa_falha_permanente = 0.072
taxa_falha_temporaria = 0.98
tempo_reparo_permanente = 4.0
tempo_reparo_temporario = 0.17

custo_fusivel_por_linha = np.array([custo_fusivel] * num_linhas)
custo_religador_por_linha = np.array([custo_religador] * num_linhas)

# === Função custo ===
def calcular_custo_total(solucao):
    custo_perm = custo_temp = custo_disp = 0
    for i, dispositivo in enumerate(solucao):
        lk = comprimento_linhas[i]
        Cf = carga_linhas[i]
        if dispositivo == 0:
            fperm, ftemp = 1.2, 1.2
        elif dispositivo == 1:
            fperm, ftemp = 0.7, 0.9
            custo_disp += custo_fusivel_por_linha[i]
        elif dispositivo == 2:
            fperm, ftemp = 0.5, 0.7
            custo_disp += custo_religador_por_linha[i]
        CPk = Cf * tempo_reparo_permanente * tarifa_energia
        CTk = Cf * tempo_reparo_temporario * tarifa_energia
        custo_perm += taxa_falha_permanente * tempo_planejamento * lk * CPk * fperm
        custo_temp += taxa_falha_temporaria * tempo_planejamento * lk * CTk * ftemp
    return custo_perm + custo_temp + custo_disp

# === AG ===
def executar_ag(pop_size=50, num_generations=130):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()
    toolbox.register("attr_int", random.randint, 0, 2)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=num_linhas)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", lambda ind: (calcular_custo_total(ind),))
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", tools.mutUniformInt, low=0, up=2, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals", "avg", "min", "max"]

    for gen in range(num_generations):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.9, mutpb=0.15)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))
        hof.update(population)
        logbook.record(gen=gen, nevals=len(offspring),
                       avg=np.mean([i.fitness.values[0] for i in population]),
                       min=np.min([i.fitness.values[0] for i in population]),
                       max=np.max([i.fitness.values[0] for i in population]))
    return hof[0], calcular_custo_total(hof[0]), logbook

# === PSO ===
def executar_pso(max_iter=130):
    custos = []
    iter_contador = [0]
    def objetivo(x):
        iter_contador[0] += 1
        x = np.clip(np.round(x), 0, 2).astype(int)
        custo = calcular_custo_total(x)
        custos.append(custo)
        print(f"Iteracao PSO: {iter_contador[0]}")
        return custo
    solucao, custo = pso(objetivo, [0]*num_linhas, [2]*num_linhas, swarmsize=90, maxiter=max_iter)
    return np.clip(np.round(solucao), 0, 2).astype(int).tolist(), custo, custos

# === Função para plotar topologia ===
def plotar_topologia_34bus(solucao, titulo):
    G = nx.Graph()

    for no, (x, y) in posicoes.items():
        G.add_node(no, pos=(x, y))

    # Mapear as conexões baseado nos nomes das linhas via OpenDSS
    linha_map = {}
    for idx, nome in enumerate(nomes_linhas):
        dss.circuit_set_active_element(nome)
        bus1 = dss.lines_read_bus1().lower().split('.')[0]
        bus2 = dss.lines_read_bus2().lower().split('.')[0]
        key = tuple(sorted([bus1, bus2]))
        linha_map[key] = idx
        # Adiciona a aresta no grafo
        if bus1 in posicoes and bus2 in posicoes:
            G.add_edge(bus1, bus2, tipo='linha')

    pos = nx.get_node_attributes(G, 'pos')

    edge_colors = []
    for u, v, d in G.edges(data=True):
        key = tuple(sorted([u, v]))
        idx_linha = linha_map.get(key, None)
        if idx_linha is not None:
            dispositivo = solucao[idx_linha]
            if dispositivo == 0:
                cor = 'gray'    # Nenhum
            elif dispositivo == 1:
                cor = 'blue'    # Fusível
            elif dispositivo == 2:
                cor = 'red'     # Religador
        else:
            cor = 'gray'
        edge_colors.append(cor)

    nx.draw_networkx_nodes(G, pos, node_size=500, node_color='white', edgecolors='black')
    nx.draw_networkx_labels(G, pos, font_size=8, font_color='black')
    nx.draw_networkx_edges(G, pos, edge_color=edge_colors, width=2)

    import matplotlib.lines as mlines
    legenda = [
        mlines.Line2D([], [], color='gray', linewidth=3, label='Nenhum'),
        mlines.Line2D([], [], color='blue', linewidth=3, label='Fusível'),
        mlines.Line2D([], [], color='red', linewidth=3, label='Religador'),
    ]
    plt.legend(handles=legenda, loc='upper right')

    plt.title(titulo)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# === Plotagem de evolução AG e PSO ===
def plotar_evolucao_ag(logbook, max_iter=130):
    plt.figure(figsize=(10,6))
    geracoes = logbook.select("gen")
    custos_min = logbook.select("min")
    if len(custos_min) < max_iter:
        custos_min += [custos_min[-1]] * (max_iter - len(custos_min))
        geracoes += list(range(len(geracoes), max_iter))
    plt.plot(geracoes, custos_min, marker='o', color='blue', label="AG")
    plt.title("Evolução do Custo (AG)")
    plt.xlabel("Iteração")
    plt.ylabel("Custo Mínimo")
    plt.xticks(np.arange(0, max_iter + 1, 10))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plotar_evolucao_pso(custos_por_iteracao, max_iter=130):
    custos_minimos = []
    for i in range(max_iter):
        if i < len(custos_por_iteracao):
            custos_minimos.append(min(custos_por_iteracao[:i+1]))
        else:
            custos_minimos.append(custos_minimos[-1])
    plt.figure(figsize=(10,6))
    plt.plot(range(max_iter), custos_minimos, marker='o', color='green', label="PSO")
    plt.title("Evolução do Custo (PSO)")
    plt.xlabel("Iteração")
    plt.ylabel("Custo Mínimo")
    plt.xticks(np.arange(0, max_iter + 1, 10))
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

# === Plotagem comparativa de alocações ===
def plotar_alocacoes_comparativas(sol_ag, sol_pso):
    x = np.arange(num_linhas)
    width = 0.35
    plt.figure(figsize=(12,6))
    plt.bar(x - width/2, sol_ag, width, label='AG')
    plt.bar(x + width/2, sol_pso, width, label='PSO')
    plt.xticks(x, [nome.split('.')[1] for nome in nomes_linhas], rotation=90)
    plt.yticks([0, 1, 2], ['Nenhum', 'Fusível', 'Religador'])
    plt.ylabel('Dispositivo Alocado')
    plt.title('Comparação de Alocação: AG x PSO')
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()

# === EXECUÇÃO PRINCIPAL ===
if __name__ == "__main__":
    print("Executando Algoritmo Genético...")
    sol_ag, custo_ag, log_ag = executar_ag()

    print("\nExecutando PSO...")
    sol_pso, custo_pso, custos_pso = executar_pso()

    print("\nVetor de alocação AG (0=Nenhum,1=Fusível,2=Religador):")
    print(sol_ag)
    print(f"Custo AG: R$ {custo_ag:,.2f}")

    print("\nVetor de alocação PSO (0=Nenhum,1=Fusível,2=Religador):")
    print(sol_pso)
    print(f"Custo PSO: R$ {custo_pso:,.2f}")

    # Plotagens
    plotar_evolucao_ag(log_ag)
    plotar_evolucao_pso(custos_pso)
    plotar_alocacoes_comparativas(sol_ag, sol_pso)

    # Plotar topologia com as alocações
    plotar_topologia_34bus(sol_ag, "Topologia com Alocação AG")
    plotar_topologia_34bus(sol_pso, "Topologia com Alocação PSO")