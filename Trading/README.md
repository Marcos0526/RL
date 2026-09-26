# Stock Trading con Deep RL (Cap. 10 — Stocks Trading Using RL)

## Ejecución

Usa el entorno virtual `gym`:

```bash
/Users/marcosbautista/gym/bin/python agents/train.py
/Users/marcosbautista/gym/bin/python evaluation/evaluate.py
```

El entrenamiento lee `data/train_data.csv` y guarda
`models/ppo_stock_trading.zip`. La evaluación lee `data/test_data.csv`,
compara contra Buy and Hold, imprime retorno, Sharpe, drawdown y operaciones,
y guarda el gráfico en `evaluation/results.png`.

El entorno usa retorno porcentual por paso como recompensa. Las opciones
`volatility_penalty` y `drawdown_penalty` permiten penalizar movimientos
bruscos y caídas desde el máximo histórico del portafolio.
