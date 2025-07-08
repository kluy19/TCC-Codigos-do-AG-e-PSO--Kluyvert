import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random, time
from deap import base, creator, tools, algorithms
from pyswarm import pso
from tqdm import tqdm
import py_dss_interface

# === CONFIGURAÇÕES ===
dss_file = r"C:\Users\KLUYVERT\Desktop\IC\123Bus\IEEE123Master.dss"
dss = py_dss_interface.DSSDLL()
dss.text(f"compile [{dss_file}]")
nomes_linhas = [f"Line.{nome}" for nome in dss.lines_all_names() if not nome.lower().startswith("sw")]

tarifa_energia = 736
tempo_planejamento = 10
custo_fusivel_base = 1000
custo_religador_base = 85000
taxa_falha_permanente = 0.072
taxa_falha_temporaria = 0.98
tempo_reparo_permanente = 4.0
tempo_reparo_temporario = 0.17

num_linhas = len(nomes_linhas)
comprimento_linhas = np.ones(num_linhas) * 0.4
carga_linhas = np.ones(num_linhas) * 1.5
custo_fusivel_por_linha = custo_fusivel_base * comprimento_linhas
custo_religador_por_linha = custo_religador_base * comprimento_linhas

# === UTILITÁRIOS ===
def escolher_locator(min_val, max_val, base=5, max_ticks=20):
    intervalo = base
    while (max_val - min_val) / intervalo > max_ticks:
        intervalo *= 2
    return intervalo

# === PERDAS OPEN DSS ===
def calcular_perdas_opendss(solucao):
    dss.text(f"compile [{dss_file}]")
    dss.text("set maxcontroliter=100")
    dss.text("set controlmode=TIME")
    for i, dispositivo in enumerate(solucao):
        if dispositivo == 1:
            dss.text(f"New Fuse.F{i} monitoredobj={nomes_linhas[i]} monitoredterm=1 ratedcurrent=50")
        elif dispositivo == 2:
            dss.text(f"New Recloser.R{i} monitoredobj={nomes_linhas[i]} monitoredterm=1 PhaseTrip=600")
    dss.solution_solve()
    perdas_watts = dss.circuit_losses()
    perdas_kw = perdas_watts[0] / 1000.0
    return perdas_kw * tarifa_energia * 1000 * tempo_planejamento * 365 * 24 / 1e6

# === FUNÇÃO DE CUSTO ===
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
    custo_perdas = calcular_perdas_opendss(solucao)
    return custo_perm + custo_temp + custo_disp + custo_perdas

# === AG ===
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

    for gen in tqdm(range(num_generations), desc="Rodando AG"):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.9, mutpb=0.15)
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring): ind.fitness.values = fit
        population = toolbox.select(offspring, k=len(population))
        hof.update(population)
        logbook.record(gen=gen, nevals=len(offspring),
                       avg=np.mean([i.fitness.values[0] for i in population]),
                       min=np.min([i.fitness.values[0] for i in population]),
                       max=np.max([i.fitness.values[0] for i in population]))
    return hof[0], calcular_custo_total(hof[0]), logbook

# === PSO ===
def executar_pso(maxiter=120):
    custos_por_iteracao = []

    def func_obj(x):
        x = np.clip(np.round(x), 0, 2).astype(int)
        custo = calcular_custo_total(x)
        if custo > 1e4:  # Filtro para evitar valores irreais
            custos_por_iteracao.append(custo)
        return custo

    lb, ub = [0]*num_linhas, [2]*num_linhas
    print("Rodando PSO")
    sol, custo = pso(func_obj, lb, ub, swarmsize=90, maxiter=maxiter)
    sol = np.clip(np.round(sol), 0, 2).astype(int)
    return sol.tolist(), custo, custos_por_iteracao

# === PLOTAGENS ===
def plotar_evolucao_ag(logbook):
    plt.figure(figsize=(10,6))
    plt.plot(logbook.select("gen"), logbook.select("min"), marker='o')
    plt.title("Evolução do Custo Mínimo (AG)")
    plt.xlabel("Geração")
    plt.ylabel("Custo")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("grafico_ag_ieee123.png")
    plt.show()

def plotar_evolucao_pso(custos_por_iteracao, maxiter):
    custos_por_iteracao = [c for c in custos_por_iteracao if c > 1e4]
    blocos = np.array_split(custos_por_iteracao, maxiter)
    custos_minimos = [np.min(bloco) for bloco in blocos if len(bloco) > 0]

    plt.figure(figsize=(10,6))
    plt.plot(range(1, len(custos_minimos)+1), custos_minimos, marker='o', label="Melhor Custo")
    plt.title("Evolução do Custo por Iteração (PSO)")
    plt.xlabel("Iteração")
    plt.ylabel("Custo")
    plt.grid(True)
    plt.legend()

    ymin, ymax = min(custos_minimos), max(custos_minimos)
    intervalo_y = escolher_locator(ymin, ymax, base=10000, max_ticks=20)
    intervalo_x = escolher_locator(0, maxiter, base=10, max_ticks=20)
    plt.gca().yaxis.set_major_locator(ticker.MultipleLocator(intervalo_y))
    plt.gca().xaxis.set_major_locator(ticker.MultipleLocator(intervalo_x))

    plt.tight_layout()
    plt.savefig("grafico_pso_ieee123.png")
    plt.show()

def plotar_alocacoes_comparativas(sol_ag, sol_pso):
    indices = list(range(len(sol_ag)))
    indices_plot = indices[:25] + [i for i in indices if (sol_ag[i] == 2 or sol_pso[i] == 2) and i >= 25]
    nomes_plot = [nomes_linhas[i] for i in indices_plot]
    ag_plot = [sol_ag[i] for i in indices_plot]
    pso_plot = [sol_pso[i] for i in indices_plot]

    x = np.arange(len(indices_plot))
    width = 0.35

    plt.figure(figsize=(14, 6))
    plt.bar(x - width/2, ag_plot, width, label='AG')
    plt.bar(x + width/2, pso_plot, width, label='PSO')
    plt.xticks(x, nomes_plot, rotation=90)
    plt.yticks([0, 1, 2], ['Nenhum', 'Fusível', 'Religador'])
    plt.ylabel('Dispositivo')
    plt.title('Comparativo AG x PSO (Linhas Selecionadas)')
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig("grafico_comparativo_ieee123.png")
    plt.show()

# === EXECUÇÃO PRINCIPAL ===
if __name__ == "__main__":
    inicio_ag = time.time()
    sol_ag, custo_ag, log_ag = executar_ag()
    tempo_ag = time.time() - inicio_ag
    print("Melhor solução AG:", sol_ag)
    print("Custo AG:", custo_ag)
    print(f"Tempo de execução AG: {tempo_ag:.2f} segundos")

    inicio_pso = time.time()
    sol_pso, custo_pso, custos_pso = executar_pso()
    tempo_pso = time.time() - inicio_pso
    print("Melhor solução PSO:", sol_pso)
    print("Custo PSO:", custo_pso)
    print(f"Tempo de execução PSO: {tempo_pso:.2f} segundos")

    df_ag = pd.DataFrame({"Linha": nomes_linhas, "Dispositivo": sol_ag})
    df_pso = pd.DataFrame({"Linha": nomes_linhas, "Dispositivo": sol_pso})
    df_ag.to_excel("resultado_ag_ieee123.xlsx", index=False)
    df_pso.to_excel("resultado_pso_ieee123.xlsx", index=False)

    plotar_evolucao_ag(log_ag)
    plotar_evolucao_pso(custos_pso, maxiter=120)
    plotar_alocacoes_comparativas(sol_ag, sol_pso)
