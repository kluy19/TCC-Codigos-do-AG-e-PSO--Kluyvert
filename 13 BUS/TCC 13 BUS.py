import py_dss_interface
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
from deap import base, creator, tools, algorithms
from pyswarm import pso

# === CONFIGURAÇÃO OPEN DSS ===
dss = py_dss_interface.DSSDLL(r"C:\Program Files\OpenDSS")
dss_file = r"C:...\IEEE13Nodeckt.dss" # Modifique com o seu diretório
dss.text(f"compile [{dss_file}]")
dss.solution_solve()

# === DADOS DO PROBLEMA ===
nomes_linhas = [
    'Line.650632', 'Line.632670', 'Line.670671', 'Line.671680', 'Line.632633', 'Line.632645', 'Line.645646',
    'Line.692675', 'Line.671684', 'Line.684611', 'Line.684652', 'Line.671692'
]
num_linhas = len(nomes_linhas)

custo_fusivel = 1000
custo_religador = 15000
tarifa_energia = 282.48
tempo_planejamento = 10

taxa_falha_permanente = 0.072
taxa_falha_temporaria = 0.98
tempo_reparo_permanente = 4.0
tempo_reparo_temporario = 0.17

# === COMPRIMENTO DAS LINHAS VIA OpenDSS ===
comprimento_linhas = []
for nome in nomes_linhas:
    dss.circuit_set_active_element(nome)
    comprimento_linhas.append(dss.lines_read_length())
    print(comprimento_linhas)
comprimento_linhas = np.array(comprimento_linhas)

# === CÁLCULO DAS CARGAS POR LINHA (APROXIMADO VIA FLUXO DE POTÊNCIA) ===
carga_linhas = []

for nome in nomes_linhas:
    dss.circuit_set_active_element(nome)
    # Potência complexa nas fases (kW + j kVAr)
    potencias = dss.cktelement_powers()
    pot_ativa_total = sum(potencias[::2])  # Soma apenas os valores reais (kW)
    carga_linhas.append(abs(pot_ativa_total))  # Módulo da potência ativa total

carga_linhas = np.array(carga_linhas)

# === FUNÇÃO DE CUSTO ===
def calcular_custo_total(solucao):
    custo_permanente = 0
    custo_temporario = 0
    custo_dispositivos = 0

    for i, dispositivo in enumerate(solucao):
        lk = comprimento_linhas[i]
        Cf = carga_linhas[i]

        if dispositivo == 0:
            fator_permanente = 1.0
            fator_temporario = 1.0
        elif dispositivo == 1:
            fator_permanente = 0.7
            fator_temporario = 0.9
            custo_dispositivos += custo_fusivel
        elif dispositivo == 2:
            fator_permanente = 0.5
            fator_temporario = 0.7
            custo_dispositivos += custo_religador

        CPk = Cf * tempo_reparo_permanente * tarifa_energia
        CTk = Cf * tempo_reparo_temporario * tarifa_energia

        custo_permanente += taxa_falha_permanente * tempo_planejamento * lk * CPk * fator_permanente
        custo_temporario += taxa_falha_temporaria * tempo_planejamento * lk * CTk * fator_temporario

    return custo_permanente + custo_temporario + custo_dispositivos

# === ALGORITMO GENÉTICO ===
def executar_ag(pop_size=50, num_generations=120):
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
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.7, mutpb=0.2)
        fits = toolbox.map(toolbox.evaluate, offspring)

        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit

        population = toolbox.select(offspring, k=len(population))
        hof.update(population)

        record = {
            "gen": gen,
            "nevals": len(offspring),
            "avg": np.mean([ind.fitness.values[0] for ind in population]),
            "min": np.min([ind.fitness.values[0] for ind in population]),
            "max": np.max([ind.fitness.values[0] for ind in population]),
        }
        logbook.record(**record)

    melhor_solucao = hof[0]
    melhor_custo = calcular_custo_total(melhor_solucao)

    return melhor_solucao, melhor_custo, logbook

# === PARTICLE SWARM OPTIMIZATION ===
def executar_pso():
    custos_por_iteracao = []

    def funcao_objetivo(x):
        x = np.round(x).astype(int)
        x = np.clip(x, 0, 2)
        custo = calcular_custo_total(x)
        custos_por_iteracao.append(custo)
        return custo

    lb = [0] * num_linhas
    ub = [2] * num_linhas

    melhor_solucao, melhor_custo = pso(
        funcao_objetivo,
        lb,
        ub,
        swarmsize=90,
        maxiter=130,
        omega=0.5,
        phip=0.5,
        phig=0.9
    )

    melhor_solucao = np.round(melhor_solucao).astype(int)
    melhor_solucao = np.clip(melhor_solucao, 0, 2)

    return melhor_solucao.tolist(), melhor_custo, custos_por_iteracao

# === EXPORTAÇÃO EXCEL ===
def exportar_excel(solucao, custo, nome_arquivo):
    df = pd.DataFrame({"Linha": nomes_linhas, "Dispositivo": solucao})
    df.loc[len(df)] = ["Custo Total", custo]
    df.to_excel(nome_arquivo, index=False)
    print(f"Arquivo '{nome_arquivo}' salvo com sucesso!")

# === AJUSTE DE TICKS ===
def escolher_locator(min_val, max_val, base=5, max_ticks=20):
    intervalo = base
    num_ticks = (max_val - min_val) / intervalo
    while num_ticks > max_ticks:
        intervalo *= 2
        num_ticks = (max_val - min_val) / intervalo
    return intervalo

# === PLOTAGENS ===
def plotar_evolucao_ag(logbook):
    gerações = [record["gen"] for record in logbook]
    custos_minimos = [record["min"] for record in logbook]

    plt.figure(figsize=(10,6))
    plt.plot(gerações, custos_minimos, marker='o')
    plt.title("Evolução do Custo Mínimo por Geração (AG)")
    plt.xlabel("Geração")
    plt.ylabel("Custo Mínimo")
    plt.grid(True)

    ymin, ymax = min(custos_minimos), max(custos_minimos)
    intervalo_y = escolher_locator(ymin, ymax, base=5, max_ticks=20)
    intervalo_x = escolher_locator(min(gerações), max(gerações), base=10, max_ticks=20)

    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(intervalo_y))
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(intervalo_x))

    plt.show()

def plotar_evolucao_pso(custos_por_iteracao, maxiter):
    blocos = np.array_split(custos_por_iteracao, maxiter)
    custos_minimos = [np.min(bloco) for bloco in blocos]

    plt.figure(figsize=(10,6))
    plt.plot(range(len(custos_minimos)), custos_minimos, marker='o', label="Melhor Custo")
    plt.title("Evolução do Custo por Iteração (PSO)")
    plt.xlabel("Iteração")
    plt.ylabel("Custo")
    plt.grid(True)
    plt.legend()

    ymin, ymax = min(custos_minimos), max(custos_minimos)
    intervalo_y = escolher_locator(ymin, ymax, base=5, max_ticks=20)
    intervalo_x = escolher_locator(0, maxiter, base=10, max_ticks=20)

    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(intervalo_y))
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(intervalo_x))

    plt.show()

def comparar_solucao_ag_pso(sol_ag, sol_pso):
    indices = np.arange(len(nomes_linhas))
    largura = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(indices - largura/2, sol_ag, width=largura, label="AG", color="skyblue")
    plt.bar(indices + largura/2, sol_pso, width=largura, label="PSO", color="salmon")

    plt.xticks(indices, nomes_linhas, rotation=45, ha='right')
    plt.yticks([0, 1, 2], ["Nenhum", "Fusível", "Religador"])
    plt.ylabel("Dispositivo Instalado")
    plt.xlabel("Linhas do Sistema")
    plt.title("Comparação das Soluções de AG e PSO por Linha")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()

# === EXECUÇÃO PRINCIPAL ===
if __name__ == "__main__":
    print("=== Rodando AG ===")
    inicio_ag = time.time()
    solucao_ag, custo_ag, log_ag = executar_ag()
    fim_ag = time.time()
    print("Melhor solução AG:", solucao_ag)
    print("Custo Total AG:", custo_ag)
    print(f"Tempo de execução AG: {fim_ag - inicio_ag:.2f} segundos")
    exportar_excel(solucao_ag, custo_ag, "resultado_ag.xlsx")
    plotar_evolucao_ag(log_ag)

    print("\n=== Rodando PSO ===")
    inicio_pso = time.time()
    solucao_pso, custo_pso, custos_pso = executar_pso()
    fim_pso = time.time()
    print("Melhor solução PSO:", solucao_pso)
    print("Custo Total PSO:", custo_pso)
    print(f"Tempo de execução PSO: {fim_pso - inicio_pso:.2f} segundos")
    exportar_excel(solucao_pso, custo_pso, "resultado_pso.xlsx")
    plotar_evolucao_pso(custos_pso, 130)

    comparar_solucao_ag_pso(solucao_ag, solucao_pso)
