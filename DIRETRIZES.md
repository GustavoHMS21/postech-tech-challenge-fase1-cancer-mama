# Diretrizes do projeto — Tech Challenge Fase 1

## Objetivo
Sistema de **apoio ao diagnóstico** (não substitui o médico) que classifica tumores de mama
como maligno/benigno a partir do dataset público Breast Cancer Wisconsin (Diagnostic).

## Estrutura
```
src/data.py           carrega o dataset (sklearn) -> DataFrame
src/eda.py            estatísticas, distribuições, correlação -> reports/
src/preprocessing.py  split treino/val/teste + pipeline de pré-processamento
src/train.py          treina 2 modelos, escolhe pela validação, salva em models/
src/evaluate.py       métricas (accuracy, recall, f1) no teste -> reports/results/
src/explain.py        feature importance + SHAP -> reports/figures/
src/run_pipeline.py   executa tudo em sequência
app/app.py            interface Streamlit (previsão + probabilidade + SHAP + aviso)
tests/                pytest; toda função de src/ tem teste
notebooks/            demonstração; chama funções de src/, não duplica lógica
```

## Princípios de arquitetura
- Protótipo funcional acadêmico, não sistema de produção. Sempre a solução mais simples que atende aos requisitos.
- Sem overengineering: nada de microsserviços, APIs, bancos, filas ou infraestrutura complexa sem necessidade.
- Toda decisão arquitetural deve ser fácil de explicar em uma apresentação, por qualquer integrante do grupo.
- Ferramentas gratuitas, open source e execução local sempre que possível.
- Requisitos obrigatórios primeiro. Toda ideia nova é classificada como **Obrigatório / Recomendável / Extra**
  antes de ser implementada; nada entra só por ser tecnicamente interessante.
- Participação humana nos pontos que exigem validação ou decisão (human-in-the-loop).

## Regras de código
- Python 3.10+, apenas libs de `requirements.txt`.
- Funções pequenas, com docstring curta em português.
- `random_state=42` em tudo que envolve aleatoriedade.
- Todo trecho de código tem um comentário explicando o **porquê** (não só o quê).
- Testes mínimos e diretos (2-3 asserts essenciais por módulo).
- Toda etapa nova: implementar -> `pytest -q` -> só então avançar.

## Cuidados com data leakage
- Split treino/validação/teste **antes** de qualquer ajuste (scaler, seleção de features).
- Scaler e demais transformadores vivem dentro de um `Pipeline` do sklearn e são ajustados só no treino.
- Conjunto de teste é usado **uma única vez**, na avaliação final. Escolha de modelo usa validação.

## Limites do modelo / human-in-the-loop
- A saída do modelo é uma **sugestão com probabilidade**, nunca um diagnóstico.
- Interface e relatório exibem explicitamente que a decisão final é do profissional de saúde.
- Métrica prioritária: **Recall da classe maligna** (falso negativo é o erro mais grave), acompanhada de F1 e accuracy.

## Processo de desenvolvimento
1. Entender: ler contexto e arquivos relevantes.
2. Planejar: menor alteração que resolve.
3. Implementar: só o necessário, sem reescrever o que funciona.
4. Validar: `pytest -q` e execução do script da etapa.
5. Documentar: o que mudou, por quê, como testar.
Uma etapa por vez, validada antes de avançar.
