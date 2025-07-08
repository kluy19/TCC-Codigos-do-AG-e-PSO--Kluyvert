import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import time
from deap import base, creator, tools, algorithms
from pyswarm import pso
import py_dss_interface
import networkx as nx


# === Configurações iniciais e dados ===
dss = py_dss_interface.DSSDLL(r"C:\Program Files\OpenDSS")
dss_file = r"C:...\34Bus\ieee34Mod2.dss" # Modifique com o seu diretório
dss.text(f"compile [{dss_file}]")
dss.solution_solve()
nomes_linhas = [f"Line.{nome}" for nome in dss.lines_all_names() if not nome.lower().startswith("sw")]
num_linhas = len(nomes_linhas)

tarifa_energia = 736
tempo_planejamento = 10
custo_fusivel = 1000      # custo fixo por fusível (R$)
custo_religador = 85000   # custo fixo por religador (R$)

comprimento_linhas_ft = []
for nome in nomes_linhas:
    dss.circuit_set_active_element(nome)
    comprimento_linhas_ft.append(dss.lines_read_length())
comprimento_linhas = np.array(comprimento_linhas_ft, dtype=float) * 0.0003048  # pés para km

carga_linhas = []
for nome in nomes_linhas:
    dss.circuit_set_active_element(nome)
    potencias = dss.cktelement_powers()
    pot_ativa_total = sum(potencias[::2])
    carga_linhas.append(abs(pot_ativa_total))
carga_linhas = np.array(carga_linhas)

# Custo fixo por dispositivo, independente do comprimento da linha
custo_fusivel_por_linha = np.full(num_linhas, custo_fusivel)
custo_religador_por_linha = np.full(num_linhas, custo_religador)

taxa_falha_permanente = 0.072
taxa_falha_temporaria = 0.98
tempo_reparo_permanente = 4.0
tempo_reparo_temporario = 0.17

tronco_principal = [
    'Line.l1', 'Line.l2a', 'Line.l2b', 'Line.l3', 'Line.l6', 'Line.l7', 'Line.l24',
    'Line.l9a', 'Line.l9b', 'Line.l12a', 'Line.l12b', 'Line.l13a', 'Line.l13b',
    'Line.l14a', 'Line.l14b', 'Line.l15', 'Line.l27', 'Line.l25',
    'Line.l16a', 'Line.l16b', 'Line.l29a', 'Line.l29b', 'Line.l18', 'Line.l17a', 'Line.l17b', 'Line.l30a', 'Line.l30b',
    'Line.l20', 'Line.l31a', 'Line.l31b',
]

ramos_secundarios = {
    'R1': ['Line.l4a', 'Line.l4b', 'Line.l5'],
    'R2': ['Line.l8', 'Line.l10a', 'Line.l10b', 'Line.l11a', 'Line.l11b'],
    'R3': ['Line.l12a', 'Line.l12b'],
    'R4': ['Line.l18', 'Line.l21a', 'Line.l21b', 'Line.l22a', 'Line.l22b', 'Line.l23a', 'Line.l23b'],
    'R5': ['Line.l19a', 'Line.l19b'],
    'R6': ['Line.l20', 'Line.l31a', 'Line.l31b'],
    'R7': ['Line.l32']
}

inicio_alimentador = 'Line.l1'

linhas_sem_fusivel = ['Line.l5']  # Linhas proibidas para fusível

PENALIDADE_FUSIVEL_TRONCO = 1e6
PENALIDADE_RELIGADORES_TRONCO_EXCESSO = 1e6
PENALIDADE_RELIGADOR_LATERAL = 5e5
PENALIDADE_FUSIVEL_PROIBIDO = 1e6

# === Função custo com penalidades e cálculo detalhado ===
def calcular_custo_total(solucao):
    custo_total = 0
    penalidade = 0
    custo_disp_total = 0
    custo_falha_total = 0
    religadores_tronco = 0

    # Penalidades
    for i, dispositivo in enumerate(solucao):
        linha = nomes_linhas[i]
        if linha in tronco_principal and dispositivo == 1:
            penalidade += PENALIDADE_FUSIVEL_TRONCO
        if linha in tronco_principal and dispositivo == 2:
            religadores_tronco += 1
        if religadores_tronco > 3:
            penalidade += PENALIDADE_RELIGADORES_TRONCO_EXCESSO
        if linha not in tronco_principal and dispositivo == 2:
            penalidade += PENALIDADE_RELIGADOR_LATERAL
        if linha in linhas_sem_fusivel and dispositivo == 1:
            penalidade += PENALIDADE_FUSIVEL_PROIBIDO

    # Cálculo do custo dispositivos e falhas
    for i, dispositivo in enumerate(solucao):
        Cf = carga_linhas[i]

        custo_dispositivo = 0
        if dispositivo == 1:
            custo_dispositivo = custo_fusivel_por_linha[i]
        elif dispositivo == 2:
            custo_dispositivo = custo_religador_por_linha[i]

        if dispositivo == 0:
            fperm, ftemp = 1.0, 1.0
        elif dispositivo == 1:
            fperm, ftemp = 0.7, 0.9
        else:
            fperm, ftemp = 0.5, 0.7

        CPk = Cf * tempo_reparo_permanente * tarifa_energia
        CTk = Cf * tempo_reparo_temporario * tarifa_energia

        custo_falha = (taxa_falha_permanente * tempo_planejamento * CPk * fperm +
                       taxa_falha_temporaria * tempo_planejamento * CTk * ftemp)

        custo_disp_total += custo_dispositivo
        custo_falha_total += custo_falha

    custo_total = custo_disp_total + custo_falha_total + penalidade
    return custo_total, custo_disp_total, custo_falha_total, penalidade

# === Ajustar solução para respeitar restrições ===
def ajustar_solucao(solucao):
    sol = solucao.copy()

    obrigatorios_religador = ['Line.l1', 'Line.l24']
    for linha in obrigatorios_religador:
        idx = nomes_linhas.index(linha)
        sol[idx] = 2

    for i, linha in enumerate(nomes_linhas):
        if linha in tronco_principal and sol[i] == 1:
            sol[i] = 0

    for ramo in ramos_secundarios.values():
        for linha in ramo:
            idx = nomes_linhas.index(linha)
            if sol[idx] == 2:
                sol[idx] = 0

    for linha_proibida in linhas_sem_fusivel:
        if linha_proibida in nomes_linhas:
            idx = nomes_linhas.index(linha_proibida)
            if sol[idx] == 1:
                sol[idx] = 0

    fusiveis_laterais_idx = [i for i, val in enumerate(sol) if val == 1 and nomes_linhas[i] not in tronco_principal]
    while len(fusiveis_laterais_idx) > 10:
        idx_remover = min(fusiveis_laterais_idx, key=lambda x: carga_linhas[x])
        sol[idx_remover] = 0
        fusiveis_laterais_idx.remove(idx_remover)

    for ramo, linhas in ramos_secundarios.items():
        idxs_ramo = [nomes_linhas.index(l) for l in linhas if l in nomes_linhas]
        if not any(sol[i] == 1 for i in idxs_ramo):
            if idxs_ramo:
                idx_maior_carga = max(idxs_ramo, key=lambda x: carga_linhas[x])
                sol[idx_maior_carga] = 1

    religadores_tronco = [i for i, val in enumerate(sol) if val == 2 and nomes_linhas[i] in tronco_principal]
    if len(religadores_tronco) > 3:
        obrigatorios = [nomes_linhas.index(l) for l in ['Line.l1', 'Line.l24']]
        extras = [i for i in religadores_tronco if i not in obrigatorios]
        for i in extras:
            sol[i] = 0
            religadores_tronco.remove(i)
            if len(religadores_tronco) <= 3:
                break

    return sol

# === Mutação AG respeitando restrições ===
def mutacao_valida(individuo, indpb=0.1):
    for i, linha in enumerate(nomes_linhas):
        if random.random() < indpb:
            if linha in tronco_principal:
                religadores_atuais = sum(1 for j, val in enumerate(individuo) if val == 2 and nomes_linhas[j] in tronco_principal)
                if individuo[i] == 2:
                    individuo[i] = random.choice([0, 2])
                else:
                    if religadores_atuais < 3:
                        individuo[i] = random.choice([0, 2])
                    else:
                        individuo[i] = 0
            elif linha in linhas_sem_fusivel:
                individuo[i] = random.choice([0, 2])
            else:
                individuo[i] = random.choices([0, 1, 2], weights=[0.3, 0.6, 0.1])[0]

    individuo[nomes_linhas.index('Line.l1')] = 2
    individuo[nomes_linhas.index('Line.l24')] = 2

    fusiveis_laterais_idx = [i for i, val in enumerate(individuo) if val == 1 and nomes_linhas[i] not in tronco_principal]
    while len(fusiveis_laterais_idx) > 10:
        idx_remover = min(fusiveis_laterais_idx, key=lambda x: carga_linhas[x])
        individuo[idx_remover] = 0
        fusiveis_laterais_idx.remove(idx_remover)

    for ramo, linhas in ramos_secundarios.items():
        idxs_ramo = [nomes_linhas.index(l) for l in linhas if l in nomes_linhas]
        if not any(individuo[i] == 1 for i in idxs_ramo):
            if idxs_ramo:
                idx_maior_carga = max(idxs_ramo, key=lambda x: carga_linhas[x])
                individuo[idx_maior_carga] = 1

    return individuo,

# === Inicialização AG respeitando restrições ===
def individual_init_limitado():
    ind = [0] * num_linhas
    ind[nomes_linhas.index('Line.l1')] = 2
    ind[nomes_linhas.index('Line.l24')] = 2

    laterais = [i for i in range(num_linhas) if nomes_linhas[i] not in tronco_principal]
    escolhidos = random.sample(laterais, min(10, len(laterais)))
    for i in escolhidos:
        ind[i] = 1

    for ramo, linhas in ramos_secundarios.items():
        idxs_ramo = [nomes_linhas.index(l) for l in linhas if l in nomes_linhas]
        if not any(ind[i] == 1 for i in idxs_ramo) and idxs_ramo:
            idx_maior_carga = max(idxs_ramo, key=lambda x: carga_linhas[x])
            ind[idx_maior_carga] = 1

    for linha_proibida in linhas_sem_fusivel:
        if linha_proibida in nomes_linhas:
            idx = nomes_linhas.index(linha_proibida)
            ind[idx] = 0

    return ind

# === AG ===
def executar_ag(pop_size=50, num_generations=130):
    if 'FitnessMin' not in creator.__dict__:
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    if 'Individual' not in creator.__dict__:
        creator.create("Individual", list, fitness=creator.FitnessMin)
    toolbox = base.Toolbox()

    toolbox.register("individual", tools.initIterate, creator.Individual, individual_init_limitado)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", lambda ind: (calcular_custo_total(ind)[0],))
    toolbox.register("mate", tools.cxUniform, indpb=0.5)
    toolbox.register("mutate", mutacao_valida, indpb=0.15)
    toolbox.register("select", tools.selTournament, tournsize=3)

    population = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals", "avg", "min", "max"]

    for gen in range(num_generations):
        offspring = algorithms.varAnd(population, toolbox, cxpb=0.9, mutpb=0.15)
        offspring = [creator.Individual(ajustar_solucao(ind)) for ind in offspring]

        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit

        population = toolbox.select(offspring, k=len(population))
        hof.update(population)

        logbook.record(gen=gen, nevals=len(offspring),
                       avg=np.mean([i.fitness.values[0] for i in population]),
                       min=np.min([i.fitness.values[0] for i in population]),
                       max=np.max([i.fitness.values[0] for i in population]))

    melhor = ajustar_solucao(hof[0])
    custo, custo_disp, custo_falha, penalidade = calcular_custo_total(melhor)
    return melhor, custo, custo_disp, custo_falha, penalidade, logbook

# === PSO ===
def executar_pso(max_iter=130):
    custos = []
    iter_contador = [0]

    def objetivo(x):
        iter_contador[0] += 1
        x_int = []
        for i, val in enumerate(x):
            val_int = int(round(val))
            if val_int < 0:
                val_int = 0
            elif val_int > 2:
                val_int = 2
            if (nomes_linhas[i] in tronco_principal or nomes_linhas[i] in linhas_sem_fusivel) and val_int == 1:
                val_int = 0
            x_int.append(val_int)
        x_int = ajustar_solucao(x_int)
        custo, _, _, _ = calcular_custo_total(x_int)
        custos.append(custo)
        return custo

    solucao, custo = pso(objetivo, [0]*num_linhas, [2]*num_linhas, swarmsize=90, maxiter=max_iter)

    sol_final = []
    for i, val in enumerate(solucao):
        val_int = int(round(val))
        if val_int < 0:
            val_int = 0
        elif val_int > 2:
            val_int = 2
        if (nomes_linhas[i] in tronco_principal or nomes_linhas[i] in linhas_sem_fusivel) and val_int == 1:
            val_int = 0
        sol_final.append(val_int)

    sol_final = ajustar_solucao(sol_final)
    custo_final, custo_disp, custo_falha, penalidade = calcular_custo_total(sol_final)

    return sol_final, custo_final, custo_disp, custo_falha, penalidade, custos

# === Funções de plotagem ===
def plotar_evolucao_ag(logbook, max_iter=130):
    plt.figure(figsize=(10,6))
    geracoes = logbook.select("gen")
    custos_min = logbook.select("min")
    if len(custos_min) < max_iter:
        custos_min += [custos_min[-1]]*(max_iter - len(custos_min))
        geracoes += list(range(len(geracoes), max_iter))
    plt.plot(geracoes, custos_min, label="Custo mínimo AG", color='blue')
    plt.xlabel("Geração")
    plt.ylabel("Custo")
    plt.title("Evolução do custo no AG")
    plt.grid(True)
    plt.legend()
    plt.xlim(0, max_iter)
    plt.show()

def plotar_evolucao_pso(custos, max_iter=130):
    plt.figure(figsize=(10,6))
    custos_ajustados = custos[:max_iter] if len(custos) >= max_iter else custos + [custos[-1]]*(max_iter - len(custos))
    plt.plot(range(max_iter), custos_ajustados, label="Custo PSO", color='green')
    plt.xlabel("Iteração")
    plt.ylabel("Custo")
    plt.title("Evolução do custo no PSO")
    plt.grid(True)
    plt.legend()
    plt.xlim(0, max_iter)
    plt.show()

# === Execução dos algoritmos e exibição dos resultados ===
if __name__ == "__main__":
    print("OpenDSS Started successfully!")

    start_ag = time.time()
    melhor_ag, custo_ag, custo_disp_ag, custo_falha_ag, penalidade_ag, log_ag = executar_ag()
    tempo_ag = time.time() - start_ag
    print(f"AG - Custo total: {custo_ag:.2f} | Dispositivos: {custo_disp_ag:.2f} | Falhas: {custo_falha_ag:.2f} | Penalidade: {penalidade_ag:.2f}")
    print(f"Tempo AG: {tempo_ag:.2f} s")

    start_pso = time.time()
    melhor_pso, custo_pso, custo_disp_pso, custo_falha_pso, penalidade_pso, log_pso = executar_pso()
    tempo_pso = time.time() - start_pso
    print(f"PSO - Custo total: {custo_pso:.2f} | Dispositivos: {custo_disp_pso:.2f} | Falhas: {custo_falha_pso:.2f} | Penalidade: {penalidade_pso:.2f}")
    print(f"Tempo PSO: {tempo_pso:.2f} s")

    print("\nResultado AG (linhas e dispositivos):")
    for linha, disp in zip(nomes_linhas, melhor_ag):
        print(f"{linha}: {disp}")

    print("\nResultado PSO (linhas e dispositivos):")
    for linha, disp in zip(nomes_linhas, melhor_pso):
        print(f"{linha}: {disp}")

    # Vetores planos
    print("\nResultado em vetor plano:")
    print(f"ag = {melhor_ag}")
    print(f"pso = {melhor_pso}")

    # Resultado detalhado
    print("\n=== RESULTADO AG ===")
    print(f"Custo Dispositivos: R$ {custo_disp_ag:.2f}")
    print(f"Custo Falhas: R$ {custo_falha_ag:.2f}")
    print(f"Penalidade: R$ {penalidade_ag:.2f}")
    print(f"Custo Total: R$ {custo_ag:.2f}")

    print("\n=== RESULTADO PSO ===")
    print(f"Custo Dispositivos: R$ {custo_disp_pso:.2f}")
    print(f"Custo Falhas: R$ {custo_falha_pso:.2f}")
    print(f"Penalidade: R$ {penalidade_pso:.2f}")
    print(f"Custo Total: R$ {custo_pso:.2f}")

    # Gráficos
    plotar_evolucao_ag(log_ag)
    plotar_evolucao_pso(log_pso)
