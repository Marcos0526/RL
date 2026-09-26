# Tarea DQN: Baseline vs Dueling en CartPole-v1

Comparación de DQN clásico (`train_base.py`) contra DQN con red Dueling
(`train_mejorado.py`) en `CartPole-v1`, con 3 semillas y gráficas comparativas (`plots.py`).

## Componente Rainbow elegido: Dueling Network

`train_mejorado.py` implementa **Dueling DQN** (`class DuelingDQN`): separa la
estimación de Q en dos streams, valor del estado `V(s)` y ventaja `A(s,a)`:

```
Q(s, a) = V(s) + (A(s, a) - mean_a A(s, a))
```

**Justificación:** en CartPole muchas observaciones consecutivas comparten el
mismo valor (el poste sigue en pie) y lo único que importa es la ventaja
relativa de mover izquierda vs derecha. Aprender `V(s)` por separado reduce la
varianza de los targets y acelera la convergencia frente al MLP monolítico del
baseline. El resto del algoritmo no cambia (loss MSE a 1 paso, target net,
epsilon-greedy, replay buffer), así que la comparación aísla el efecto de la
arquitectura.

## Hiperparámetros finales

Idénticos en base y mejorado (solo cambia la red):

| Hiperparámetro      | Valor        |
|---------------------|--------------|
| Entorno             | CartPole-v1  |
| Learning rate       | 1e-3 (Adam)  |
| Batch size          | 32           |
| Gamma               | 0.99         |
| Replay size         | 10 000       |
| Replay initial      | 1 000        |
| Target net sync     | cada 100 frames |
| Epsilon             | 1.0 → 0.02 en 5 000 frames |
| Stop reward         | 475.0        |
| Max frames          | 200 000      |
| Red base            | MLP obs → 128 → ReLU → Q |
| Red mejorada        | Dueling: 128 compartidas + V(64→1) + A(64→n_acc) |

## Semillas utilizadas

`42`, `123`, `2024` — fijan `random`, `numpy`, `torch` y el entorno
(`env_seed` + `action_space.seed`). Cada corrida genera
`results/reward_{base,mejorado}_seed{seed}.csv` y su checkpoint
`checkpoints/dqn_{base,dueling}_cartpole_seed{seed}.pt`.

## Reproducir los experimentos

```bash
cd Tarea_DQN

# Baseline (3 semillas)
python3 train_base.py --seed 42
python3 train_base.py --seed 123
python3 train_base.py --seed 2024

# Mejorado Dueling (3 semillas)
python3 train_mejorado.py --seed 42
python3 train_mejorado.py --seed 123
python3 train_mejorado.py --seed 2024

# Graficas comparativas + tabla (metricas, base, mejorado)
python3 plots.py
```

Salidas en `results/`: `comparacion_seed{42,123,2024}.png`,
`comparacion_promedio.png` y `tabla_comparativa.csv` (reward final medio de las
últimas 100 ep, desviación estándar y episodio del primer reward ≥ 475).
