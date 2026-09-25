# Tarea DQN — CartPole-v1 (DQN base vs Dueling DQN)

## Descripcion y objetivo
Proyecto de Deep Q-Network sobre `CartPole-v1` (gymnasium) con `ptan` + PyTorch + ignite.
- `train_base.py` (`01_baseline_cartpole`): DQN fully-connected (obs 4 → 128 → ReLU → 2 acciones).
- `train_mejorado.py` (`06_dueling_cartpole`): Dueling DQN (feature 128 + streams Valor `V(s)` y Ventaja `A(s,a)`, `Q = V + (A - mean(A))`).
- `lib/dqn_model.py`: DQN convolucional Atari (`Conv2d`, normalizacion `x/255.0`).
- `lib/dqn_extra.py`: NoisyDQN, DuelingDQN conv, DistributionalDQN, RainbowDQN, `PrioReplayBuffer`, `distr_projection`.
- `lib/common.py`: `calc_loss_dqn`, `EpsilonTracker`, `batch_generator`, `setup_ignite` (TensorBoard + `EndOfEpisodeHandler` con `stop_reward=475.0`).

## Instalacion
```bash
git clone <repo> && cd APR/Tarea_DQN
python3 -m venv ~/gym
source ~/gym/bin/activate
pip install -r requirements.txt
```

## Ejecucion
```bash
# Entrenar base (DQN)
python train_base.py
# Entrenar mejorado (Dueling, guarda results/rewards_mejorado.csv)
python train_mejorado.py
# Evaluar checkpoints (greedy, sin exploracion)
python eval_cartpole.py --model base --episodes 20
python eval_cartpole.py --model dueling --episodes 20
# TensorBoard
tensorboard --logdir runs
# Regenerar grafica
python plot_results.py  # -> results/reward_curve.png
```

## Resultados
- `results/rewards_mejorado.csv`: 19933 episodios del run Dueling completo (columnas `episode,reward`).
- `results/reward_curve.png`: curva de recompensa + media movil 100 (generada con `plot_results.py`).
- `checkpoints/01_baseline_cartpole.pt` (5.7K) y `checkpoints/06_dueling_cartpole.pt` (72K): pesos smoke-test de 30 iteraciones para validar el pipeline (re-entrenar completo para pesos finales `stop_reward=475`).
- `runs/`: logs TensorBoard por corrida (`cartpole-01_*`, `dueling-06_*`).
- Eval smoke (5 episodios, pesos iniciales): base `mean≈10`, dueling `mean≈9.4` — esperado antes de convergencia.

## Estructura
```
Tarea_DQN/
  train_base.py  train_mejorado.py  eval_cartpole.py  plot_results.py
  requirements.txt  README.md
  lib/common.py  lib/dqn_model.py  lib/dqn_extra.py
  results/rewards_mejorado.csv  results/reward_curve.png
  checkpoints/*.pt  runs/
```
