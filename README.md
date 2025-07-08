# TCC-Codigos-do-AG-e-PSO--Kluyvert

# Otimização de Dispositivos de Proteção com AG e PSO

Códigos utilizados para o Trabalho de Conclusão de Curso (TCC) de Kluyvert Ananias Nunes, aplicando Algoritmos Genéticos (AG) e Particle Swarm Optimization (PSO) na alocação de fusíveis e religadores em sistemas de distribuição.

## Objetivo

Analisar o desempenho de algoritmos de otimização na alocação estratégica de dispositivos de proteção, buscando minimizar custos com falhas.

## Ferramentas Utilizadas

- Python
- OpenDSS
- py_dss_interface
- matplotlib, pandas, numpy, tqdm
- Sistemas IEEE 13, 34 e 123 nós

## Sistemas Testados

- IEEE 13 Nós: sistema básico para validação inicial.
- IEEE 34 Nós: rede com ramificações para testar topologia.
- IEEE 123 Nós: rede densa e realista para análise mais robusta.

## Funcionalidades

- Simulação de falhas e impactos nos consumidores
- Alocação otimizada de dispositivos (0: nenhum, 1: fusível, 2: religador)
- Gráficos comparativos entre AG e PSO
- Cálculo dos indicadores 
- Visualização gráfica da topologia com dispositivos

## Estrutura do Projeto

## Estrutura do Projeto

O repositório contém pastas separadas para cada sistema estudado:

- `13_Bus/`       → Código e resultados do sistema IEEE 13 barras  
- `34_Bus/`       → Código e resultados do sistema IEEE 34 barras  
- `123_Bus/`      → Código e resultados do sistema IEEE 123 barras  

Além disso, há arquivos principais que podem ser executados diretamente:

- `13 Bus.py`        → Execução do AG no sistema IEEE 13 barras  
- `34 Bus.py`        → Execução do AG e PSO no sistema IEEE 34 barras
- `34 Bus grafos.py`→ Geração dos grafos com os resultados finais do IEEE 34 barras
- `123 Bus - AG.py`  → Execução do AG no sistema IEEE 123 barras  
- `123 Bus - PSO.py` → Execução do PSO no sistema IEEE 123 barras  
- `123 Bus gGrafos.py`→ Geração dos grafos com os resultados finais do IEEE 123 barras

Após obter os vetores de resultados das alocações (AG e PSO), é possível plotar esses resultados nos grafos substituindo os vetores correspondentes nos arquivos de geração dos grafos, facilitando a visualização comparativa da alocação dos dispositivos.

## Autor

Kluyvert Ananias Nunes  
Bacharelado em Engenharia Elétrica – IFES, Campus Guarapari  
Orientador: Prof. Dr. Murillo Cobe Vargas  

## Referência

Este repositório faz parte do TCC:

"Técnicas de Otimização Aplicadas em Sistemas Elétricos de Potência"  
Instituto Federal do Espírito Santo – 2025

Link para alimentadores do IEEE:
https://cmte.ieee.org/pes-testfeeders/resources/
