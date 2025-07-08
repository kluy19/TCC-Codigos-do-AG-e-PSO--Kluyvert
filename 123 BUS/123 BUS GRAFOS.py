import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import re
import py_dss_interface
dss_file = r"C:...\IC\123Bus\IEEE123Master.dss" #Mude para seu diretório
dss = py_dss_interface.DSSDLL()
dss.text(f"compile [{dss_file}]")
# --- Caminhos dos arquivos ---
caminho_posicoes = r"C:...\BusCoords.dat" #Mude para seu diretório 
caminho_conexoes = r"C:...\conexoes_123bus.csv" #Mude para seu diretório
nomes_linhas = [f"Line.{nome}" for nome in dss.lines_all_names() if not nome.lower().startswith("sw")]
num_linhas = len(nomes_linhas)
print(nomes_linhas)

# --- Lista de nós duplicados a excluir ---
nodos_excluir = {"9r", "25r", "150r", "160r", "61s"}

# --- Solução do AG: 0 = nenhum, 1 = fusível (azul), 2 = religador (vermelho) ---
#Substitua com a resposta do 123 BUS AG
solucao_ag = [2, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 2, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 2, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0]


# --- Solução do PSO (exemplo, substitua pela sua real) ---
#Substitua com a resposta do 123 BUS PSO
solucao_pso = [2, 0, 1, 0, 0, 0, 0, 2, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0]

# --- Função para normalizar nomes de barramentos ---
def normaliza_nome(nome):
    nome = str(nome).strip().lower()
    nome = re.sub(r'[rs]$', '', nome)
    return nome

# --- Ler posições dos barramentos ---
posicoes_df = pd.read_csv(caminho_posicoes, delim_whitespace=True, header=None, names=["node", "x", "y"])
posicoes_df["node"] = posicoes_df["node"].astype(str).str.strip().str.lower()
posicoes_df = posicoes_df.drop_duplicates(subset="node", keep="first")
posicoes = {normaliza_nome(row["node"]): (row["x"], row["y"]) for _, row in posicoes_df.iterrows()}

# --- Ler conexões ---
df_conexoes = pd.read_csv(caminho_conexoes)
df_conexoes["bus1"] = df_conexoes["bus1"].astype(str).str.strip().str.lower()
df_conexoes["bus2"] = df_conexoes["bus2"].astype(str).str.strip().str.lower()

# --- Função para criar grafo com base na solução ---
def cria_grafo_com_solucao(solucao):
    G = nx.Graph()
    for idx, row in df_conexoes.iterrows():
        b1, b2 = row["bus1"], row["bus2"]
        if b1 in nodos_excluir or b2 in nodos_excluir:
            continue
        b1_n = normaliza_nome(b1)
        b2_n = normaliza_nome(b2)
        if b1_n in posicoes and b2_n in posicoes:
            G.add_node(b1, pos=posicoes[b1_n])
            G.add_node(b2, pos=posicoes[b2_n])
            tipo = solucao[idx] if idx < len(solucao) else 0
            G.add_edge(b1, b2, tipo=tipo)
    return G

# --- Função para plotar grafo com legenda ---
def plotar_grafo(G, titulo):
    pos = nx.get_node_attributes(G, "pos")

    edges_nenhum = [(u, v) for u, v, d in G.edges(data=True) if d["tipo"] == 0]
    edges_fusivel = [(u, v) for u, v, d in G.edges(data=True) if d["tipo"] == 1]
    edges_religador = [(u, v) for u, v, d in G.edges(data=True) if d["tipo"] == 2]

    plt.figure(figsize=(14, 10))
    nx.draw_networkx_nodes(G, pos, node_size=250, node_color="lightblue", edgecolors="black", linewidths=0.8)
    nx.draw_networkx_labels(G, pos, font_size=8, font_color="black")

    # Desenhar as arestas conforme tipo
    nx.draw_networkx_edges(G, pos, edgelist=edges_nenhum, edge_color="gray", style="solid", width=1.5)
    nx.draw_networkx_edges(G, pos, edgelist=edges_fusivel, edge_color="blue", style="solid", width=2)
    nx.draw_networkx_edges(G, pos, edgelist=edges_religador, edge_color="red", style="solid", width=2)

    # Legenda personalizada
    legenda = [
        mlines.Line2D([], [], color='gray', linewidth=1.5, label='Nenhum'),
        mlines.Line2D([], [], color='blue', linewidth=2, label='Fusível'),
        mlines.Line2D([], [], color='red', linewidth=2, label='Religador'),
    ]
    plt.legend(handles=legenda, loc='best', fontsize=12)

    plt.title(titulo, fontsize=16)
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# --- Criar e plotar grafos ---
G_ag = cria_grafo_com_solucao(solucao_ag)
plotar_grafo(G_ag, "Topologia IEEE 123 Nós - Solução AG")

G_pso = cria_grafo_com_solucao(solucao_pso)
plotar_grafo(G_pso, "Topologia IEEE 123 Nós - Solução PSO")
