# TCC-Codigos-do-AG-e-PSO--Kluyvert

# Otimização de Dispositivos de Proteção com AG e PSO

Códigos utilizados para o Trabalho de Conclusão de Curso (TCC) de Kluyvert Ananias Nunes, aplicando Algoritmos Genéticos (AG) e Particle Swarm Optimization (PSO) na alocação de fusíveis e religadores em sistemas de distribuição.

## Objetivo

Analisar o desempenho de algoritmos de otimização na alocação estratégica de dispositivos de proteção, buscando minimizar custos com falhas e melhorar indicadores de continuidade como DEC e FEC.

## Ferramentas Utilizadas

- Python
- OpenDSS
- py_dss_interface
- matplotlib, pandas, numpy, tqdm
- Sistemas IEEE 13, 34 e 123 barras

## Sistemas Testados

- IEEE 13 Barras: sistema básico para validação inicial.
- IEEE 34 Barras: rede com ramificações para testar topologia.
- IEEE 123 Barras: rede densa e realista para análise mais robusta.

## Funcionalidades

- Simulação de falhas e impactos nos consumidores
- Alocação otimizada de dispositivos (0: nenhum, 1: fusível, 2: religador)
- Gráficos comparativos entre AG e PSO
- Cálculo dos indicadores (custo, DEC, FEC)
- Visualização gráfica da topologia com dispositivos (123 barras)

## Estrutura do Projeto

13_bus/       → Código e resultados do sistema IEEE 13 barras  
34_bus/       → Código e resultados do sistema IEEE 34 barras  
123_bus/      → Código e resultados do sistema IEEE 123 barras  
grafos_123/   → Geração dos grafos com alocação no IEEE 123 barras

## Autor

Kluyvert Ananias Nunes  
Bacharelado em Engenharia Elétrica – IFES, Campus Guarapari  
Orientador: Prof. Dr. Murillo Cobe Vargas  

## Referência

Este repositório faz parte do TCC:

"Técnicas de Otimização Aplicadas em Sistemas Elétricos de Potência"  
Instituto Federal do Espírito Santo – 2025
