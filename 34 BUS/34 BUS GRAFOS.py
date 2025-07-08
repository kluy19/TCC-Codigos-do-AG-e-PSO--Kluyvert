import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import re

# === CAMINHOS ===
caminho_dss = r"C:..\34\ieee34Mod2.dss"  # Modifique com o seu diretório
caminho_posicoes_reais = r"C:...\34Bus\IEEE34_BusXY.csv" # Modifique com o seu diretório
caminho_saida = r"C:...\34Bus\IEEE34_BusXY_com_mid.csv" # Modifique com o seu diretório

# === ALIMENTAÇÃO ===
#Substitua com a resposta do 34 BUS
alocacao_ag = [2, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]
alocacao_pso = [2, 0, 0, 0, 1, 0, 0, 0, 0, 2, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]

# === NOMES DAS LINHAS (na mesma ordem dos vetores acima) ===
nomes_linhas = [
    'Line.l1', 'Line.l2a', 'Line.l2b', 'Line.l3', 'Line.l4a', 'Line.l4b', 'Line.l5', 'Line.l6', 'Line.l7',
    'Line.l24', 'Line.l8', 'Line.l9a', 'Line.l9b', 'Line.l10a', 'Line.l10b', 'Line.l11a', 'Line.l11b',
    'Line.l12a', 'Line.l12b', 'Line.l13a', 'Line.l13b', 'Line.l14a', 'Line.l14b', 'Line.l15', 'Line.l16a',
    'Line.l16b', 'Line.l29a', 'Line.l29b', 'Line.l18', 'Line.l19a', 'Line.l19b', 'Line.l21a', 'Line.l21b',
    'Line.l22a', 'Line.l22b', 'Line.l23a', 'Line.l23b', 'Line.l26a', 'Line.l26b', 'Line.l27', 'Line.l25',
    'Line.l28a', 'Line.l28b', 'Line.l17a', 'Line.l17b', 'Line.l30a', 'Line.l30b', 'Line.l20', 'Line.l31a',
    'Line.l31b', 'Line.l32'
]

# === EXTRAIR CONEXÕES DO DSS ===
with open(caminho_dss, 'r', encoding='utf-8') as f:
    dss_text = f.read()

padrao_linha = r"New Line\.(\S+)\s+.*?Bus1=(\S+)\s+Bus2=(\S+)"
matches = re.findall(padrao_linha, dss_text)

# Criar mapeamento Line.lx → (bus1, bus2)
conexoes_por_linha = {}
for nome, bus1, bus2 in matches:
    nome_linha = f"Line.{nome.lower()}"
    no1 = bus1.split('.')[0].lower()
    no2 = bus2.split('.')[0].lower()
    conexoes_por_linha[nome_linha] = (no1, no2)

# Verifica se todos os nomes estão presentes
assert all(n in conexoes_por_linha for n in nomes_linhas), "Nem todos os nomes de linhas estão no DSS!"

# Construir lista de conexões final
conexoes = [conexoes_por_linha[n] for n in nomes_linhas]

# === POSIÇÕES ===
df = pd.read_csv(caminho_posicoes_reais, header=None, names=["node", "x", "y"])
df["node"] = df["node"].astype(str).str.strip().str.lower()
posicoes = {row["node"]: (row["x"], row["y"]) for _, row in df.iterrows()}

# === MIDPOINTS ===
for origem, destino in conexoes:
    if "mid" in origem or "mid" in destino:
        mid = origem if "mid" in origem else destino
        conexoes_mid = [c for c in conexoes if mid in c]
        if len(conexoes_mid) == 2:
            no1 = conexoes_mid[0][0] if conexoes_mid[0][1] == mid else conexoes_mid[0][1]
            no2 = conexoes_mid[1][0] if conexoes_mid[1][1] == mid else conexoes_mid[1][1]
            if no1 in posicoes and no2 in posicoes:
                x1, y1 = posicoes[no1]
                x2, y2 = posicoes[no2]
                posicoes[mid] = ((x1 + x2) / 2, (y1 + y2) / 2)

# === SALVAR CSV COM POSIÇÕES ATUALIZADAS ===
df_final = pd.DataFrame([(k, v[0], v[1]) for k, v in posicoes.items()], columns=["node", "x", "y"])
df_final.to_csv(caminho_saida, index=False)

# === FUNÇÃO DE PLOTAGEM ===
def plotar_topologia(titulo, nomes_linhas, conexoes, alocacao, posicoes):
    G = nx.Graph()
    for no, (x, y) in posicoes.items():
        G.add_node(no, pos=(x, y))

    pos = nx.get_node_attributes(G, 'pos')
    edges = dict(zip(nomes_linhas, conexoes))
    aloc_dict = dict(zip(nomes_linhas, alocacao))

    plt.figure(figsize=(14, 8), dpi=120)
    nx.draw_networkx_nodes(G, pos, node_size=400, node_color='lightblue', edgecolors='black')
    nx.draw_networkx_labels(G, pos, font_size=5.5)

    for nome, (origem, destino) in edges.items():
        tipo = aloc_dict.get(nome, 0)
        cor = 'gray'
        largura = 1.3
        if tipo == 1:
            cor = 'blue'
            largura = 2.0
        elif tipo == 2:
            cor = 'red'
            largura = 2.0
        G.add_edge(origem, destino)
        nx.draw_networkx_edges(G, pos, edgelist=[(origem, destino)], edge_color=cor, width=largura)

    legenda = [
        mlines.Line2D([], [], color='gray', linewidth=1.5, label='Nenhum'),
        mlines.Line2D([], [], color='blue', linewidth=2, label='Fusível'),
        mlines.Line2D([], [], color='red', linewidth=2, label='Religador'),
    ]
    plt.legend(handles=legenda, loc='upper right', fontsize=9)
    plt.title(titulo, fontsize=11)
    plt.axis('off')
    plt.tight_layout()
    plt.show()

# === PLOTAR ===
plotar_topologia("Topologia IEEE 34 Nós - Solução AG", nomes_linhas, conexoes, alocacao_ag, posicoes)
plotar_topologia("Topologia IEEE 34 Nós - Solução PSO", nomes_linhas, conexoes, alocacao_pso, posicoes)
