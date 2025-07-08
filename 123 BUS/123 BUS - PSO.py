import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import random
import time
from pyswarm import pso
import py_dss_interface
import contextlib
import os

# === CONFIGURAÇÕES ===
dss_file = r"C:...\IEEE123Master.dss" #Mude para seu diretório
dss.text(f"compile [{dss_file}]")

nomes_linhas = [f"Line.{nome}" for nome in dss.lines_all_names() if not nome.lower().startswith("sw")]
num_linhas = len(nomes_linhas)

linhas_tronco = [
    'line.l115', 'line.l3', 'line.l7', 'line.l10', 'line.l13', 'line.l19', 'line.l22', 'line.l24', 'line.l26',
    'line.l30', 'line.l31', 'line.l32', 'line.l114', 'line.l36', 'line.l41', 'line.l43', 'line.l45', 'line.l47',
    'line.l48', 'line.l49', 'line.l50', 'line.l51', 'line.l116', 'line.l52', 'line.l53', 'line.l54', 'line.l56',
    'line.l55', 'line.l58', 'line.l60', 'line.l61', 'line.l62', 'line.l63', 'line.l64', 'line.l65', 'line.l117',
    'line.l67', 'line.l68', 'line.l73', 'line.l76', 'line.l77', 'line.l78', 'line.l79', 'line.l80', 'line.l81',
    'line.l82', 'line.l84', 'line.l86', 'line.l88', 'line.l90', 'line.l92', 'line.l94', 'line.l96', 'line.l97',
    'line.l98', 'line.l99', 'line.l118', 'line.l101', 'line.l105', 'line.l108'
]
indices_tronco = [i for i, nome in enumerate(nomes_linhas) if nome.lower() in linhas_tronco]
indices_ramo = [i for i in range(num_linhas) if i not in indices_tronco]

tarifa_energia = 736
tempo_planejamento = 10
custo_fusivel_base = 1000
custo_religador_base = 15000
taxa_falha_permanente = 0.072
taxa_falha_temporaria = 0.98
tempo_reparo_permanente = 4.0
tempo_reparo_temporario = 0.17

comprimento_linhas_ft = []
for nome in nomes_linhas:
    dss.circuit_set_active_element(nome)
    comprimento_linhas_ft.append(dss.lines_read_length())
comprimento_linhas = np.array(comprimento_linhas_ft) * 0.0003048

carga_linhas_valores = []
for nome in nomes_linhas:
    dss.circuit_set_active_element(nome)
    potencias = dss.cktelement_powers()
    pot_ativa = sum(potencias[::2])
    carga_linhas_valores.append(abs(pot_ativa))
carga_linhas = np.array(carga_linhas_valores)

custo_fusivel_por_linha = custo_fusivel_base * comprimento_linhas
custo_religador_por_linha = custo_religador_base * comprimento_linhas

def calcular_perdas_opendss():
    dss.text(f"compile [{dss_file}]")
    dss.text("set maxcontroliter=100")
    dss.text("set controlmode=TIME")
    dss.solution_solve()
    perdas_watts = dss.circuit_losses()[0]
    custo_perdas = perdas_watts / 1000.0 * tarifa_energia * 1000 * tempo_planejamento * 8760 / 1e6
    return custo_perdas

def penalizar_solucao(solucao):
    penalidade = 0
    num_religadores_tronco = 0
    num_religadores_total = 0
    religador_inicio_ok = False

    for i, tipo in enumerate(solucao):
        if tipo == 1 and i in indices_tronco:
            penalidade += 2e6
        if tipo == 2:
            num_religadores_total += 1
            if i in indices_tronco:
                num_religadores_tronco += 1
                if nomes_linhas[i].lower() in ['line.l115', 'line.l3']:
                    religador_inicio_ok = True
            else:
                penalidade += 5e5
    if not religador_inicio_ok:
        penalidade += 5e7
    if num_religadores_tronco > 4:
        penalidade += (num_religadores_tronco - 4) * 2e6
    if num_religadores_total > 4:
        penalidade += (num_religadores_total - 4) * 2e6

    for i in range(num_linhas - 1):
        if i in indices_ramo and solucao[i] == 1 and solucao[i + 1] == 1:
            penalidade += 8e5
        if (solucao[i] == 1 and solucao[i+1] == 2) or (solucao[i] == 2 and solucao[i+1] == 1):
            penalidade += 1e6

    return penalidade

def calcular_custo_total(solucao):
    custo_perm = custo_temp = custo_disp = 0
    for i, tipo in enumerate(solucao):
        lk = comprimento_linhas[i]
        Cf = carga_linhas[i]
        if tipo == 0:
            fperm, ftemp = 1.2, 1.2
        elif tipo == 1:
            fperm, ftemp = 0.7, 0.9
            custo_disp += custo_fusivel_por_linha[i]
        elif tipo == 2:
            fperm, ftemp = 0.5, 0.7
            custo_disp += custo_religador_por_linha[i]
        CPk = Cf * tempo_reparo_permanente * tarifa_energia
        CTk = Cf * tempo_reparo_temporario * tarifa_energia
        custo_perm += taxa_falha_permanente * tempo_planejamento * lk * CPk * fperm
        custo_temp += taxa_falha_temporaria * tempo_planejamento * lk * CTk * ftemp
    perdas = calcular_perdas_opendss()
    penalidade = penalizar_solucao(solucao)
    return custo_perm + custo_temp + custo_disp + perdas + penalidade, custo_disp, custo_perm + custo_temp, penalidade

# === Redução de fusíveis e religadores ===
# === AJUSTE DE FUSÍVEIS E RELIGADORES ===
def ajustar_solucao(solucao, max_fusiveis=15, max_religadores=4):
    nova = solucao.copy()

    # === Religadores ===
    religadores = [i for i, v in enumerate(nova) if v == 2]
    religadores_fixos = [i for i in religadores if nomes_linhas[i].lower() in ['line.l115', 'line.l3']]
    religadores_ajustaveis = [i for i in religadores if i not in religadores_fixos]
    while len(religadores) > max_religadores and religadores_ajustaveis:
        i_remover = religadores_ajustaveis.pop()
        nova[i_remover] = 0
        religadores.remove(i_remover)

    # === Fusíveis ===
    # Zera todos os fusíveis primeiro
    for i in range(num_linhas):
        if nova[i] == 1:
            nova[i] = 0

    # Seleciona as melhores posições nos RAMOS para alocar até 15 fusíveis
    candidatos_fusiveis = [i for i in indices_ramo if nova[i] == 0]
    candidatos_fusiveis_ordenados = sorted(candidatos_fusiveis, key=lambda i: carga_linhas[i], reverse=True)
    for i in candidatos_fusiveis_ordenados[:max_fusiveis]:
        nova[i] = 1  # Aloca fusível

    return nova

def executar_pso(maxiter=120):
    custos_por_iteracao = []
    idx_inicio_tronco_candidates = [i for i, nome in enumerate(nomes_linhas) if nome.lower() in ['line.l115', 'line.l3']]

    def func_obj(x):
        x_int = np.clip(np.round(x), 0, 2).astype(int)
        x_int[idx_inicio_tronco_candidates[0]] = 2
        x_int = ajustar_solucao(x_int)
        custo, _, _, _ = calcular_custo_total(x_int)
        custos_por_iteracao.append(custo)
        return custo

    lb, ub = [0]*num_linhas, [2]*num_linhas
    print("Rodando PSO")
    with open(os.devnull, 'w') as fnull:
        with contextlib.redirect_stdout(fnull):
            sol, custo = pso(func_obj, lb, ub, swarmsize=90, maxiter=maxiter)
    sol = np.clip(np.round(sol), 0, 2).astype(int)
    sol[idx_inicio_tronco_candidates[0]] = 2
    sol = ajustar_solucao(sol)

    # Garantir novamente que não há fusível no tronco
    for i in indices_tronco:
        if sol[i] == 1:
            sol[i] = 0

    custo, custo_disp, custo_falha, penalidade = calcular_custo_total(sol)
    return sol.tolist(), custo, custo_disp, custo_falha, penalidade, custos_por_iteracao

# === MAIN ===
if __name__ == "__main__":
    inicio_pso = time.time()
    sol_pso, custo_pso, custo_disp_pso, custo_falha_pso, penalidade_pso, custos_pso = executar_pso(maxiter=120)
    tempo_pso = time.time() - inicio_pso

    print("=== RESULTADO PSO ===")
    print(f"Custo total: R$ {custo_pso:.2f}")
    print(f"Custo Dispositivos: R$ {custo_disp_pso:.2f}")
    print(f"Custo Falhas: R$ {custo_falha_pso:.2f}")
    print(f"Penalidade: R$ {penalidade_pso:.2f}")
    print(f"Tempo PSO: {tempo_pso:.2f} segundos\n")

    for linha, disp in zip(nomes_linhas, sol_pso):
        tipo = "Nenhum" if disp == 0 else ("Fusível" if disp == 1 else "Religador")
        print(f"{linha}: {tipo}")

    print(f"pso = {sol_pso}")
