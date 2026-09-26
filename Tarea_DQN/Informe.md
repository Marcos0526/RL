# Informe: DQN con red Dueling en CartPole-v1

## 1. Introducción

Se implementó el componente **Dueling Network** de la familia Rainbow sobre un
baseline DQN clásico, ambos entrenados en `CartPole-v1`. Dueling descompone la
función Q en valor del estado `V(s)` más ventaja de cada acción `A(s,a)`:

```
Q(s, a) = V(s) + (A(s, a) - mean_a A(s, a))
```

Se eligió este componente porque CartPole es un entorno donde la mayoría de los
estados consecutivos comparten el mismo valor (el poste sigue en pie) y la
decisión relevante es solo la ventaja relativa entre mover el carro a la
izquierda o a la derecha. Aprender `V(s)` de forma explícita y compartida entre
acciones debería reducir la varianza de los objetivos de Bellman y acelerar la
convergencia respecto al MLP monolítico del baseline, sin cambiar ningún otro
elemento del algoritmo (misma loss MSE a 1 paso, misma target net, mismo
epsilon-greedy y mismo replay buffer), de modo que cualquier diferencia observada
sea atribuible a la arquitectura.

## 2. Implementación

**Archivos.** `Tarea_DQN/train_base.py` (reescrito: el original era un baseline
de Pong con CNN y dependencias rotas; ahora es un DQN-MLP autocontenido para
CartPole), `Tarea_DQN/train_mejorado.py` (espejo del base cambiando solo la red
a Dueling, con las secciones modificadas marcadas `[MODIFICADO - Dueling]`),
`Tarea_DQN/plots.py` (comparación por semilla con banda ±1 std y tabla de
métricas) y `Tarea_DQN/README.md` (reproducción).

**Clases de ptan utilizadas** (iguales en ambos): `ptan.agent.DQNAgent` (política
Q + selector de acción), `ptan.actions.EpsilonGreedyActionSelector`
(exploración en entrenamiento) y `ArgmaxActionSelector` (evaluación greedy),
`ptan.experience.ExperienceSourceFirstLast` (generación de transiciones a 1 paso,
`gamma=0.99`, `env_seed` por semilla), `ptan.experience.ExperienceReplayBuffer`
y `ptan.agent.TargetNet` (sincronizada cada 100 frames).

**Pseudocódigo del cambio** (única diferencia real entre ambos programas):

```
# BASE: Q directo
Q(s) = Linear_2(ReLU(Linear_1(s)))            # (B, n_acc)

# MEJORADO (Dueling)
feat   = ReLU(Linear(s))                      # tronco compartido 128
V(s)   = Linear(ReLU(Linear(feat)))           # stream valor -> (B, 1)
A(s,a) = Linear(ReLU(Linear(feat)))           # stream ventaja -> (B, n_acc)
Q(s,a) = V(s) + (A(s,a) - mean_a A(s,a))      # centra ventajas (identificabilidad)
```

El loop de entrenamiento (muestreo del buffer, objetivo `r + γ·max Q_tgt`,
`loss.backward()`, decaimiento lineal de epsilon 1.0→0.02 y log CSV
`(episode, reward)`) es idéntico en ambos archivos.

## 3. Resultados

Semillas `42, 123, 2024` (lr 1e-3, batch 32, γ 0.99, replay 10k/1k).
Gráficas: `results/comparacion_seed{42,123,2024}.png` (curva + banda ±1 std
móvil) y `results/comparacion_promedio.png` (media entre semillas ±1 std).

| seed | Reward final medio (100 ep) base | mejorado | Std base | mejorado | 1.er ep ≥ 475 base | mejorado |
|------|-------------------:|---------:|-------:|---------:|------------------:|---------:|
| 42   | 128.2 | **170.7** | 118.4 | 107.4 | 168 | **161** |
| 123  | 227.2 | **232.4** | 43.7  | 64.7  | **221** | 276 |
| 2024 | 129.8 | **157.7** | 137.6 | 92.8  | 157 | **151** |
| **promedio** | **161.7** | **186.9 (+15.6 %)** | **99.9** | **88.3 (−11.6 %)** | **182** | 196 |

## 4. Discusión

**¿Mejoró el baseline? ¿En qué magnitud?** Parcialmente sí. El Dueling supera al
baseline en reward final medio en las 3 semillas, con una ganancia promedio de
**+25.2 puntos (+15.6 %)** y una desviación estándar promedio **11.6 % menor**,
es decir, converge a políticas mejores y algo más estables. En cambio, en
velocidad para resolver (primer episodio con reward ≥ 475) el resultado es
mixto: gana en 2 de 3 semillas pero pierde claramente en la 123 (276 vs 221),
quedando el promedio 14 episodios por detrás.

**¿A qué lo atribuyo?** La ganancia en reward final encaja con la teoría: al
compartir `V(s)` entre acciones, la red generaliza mejor el valor de los
estados de "poste en pie" y estima ventajas más limpias, lo que se nota al
final del entrenamiento. La inconsistencia en tiempo-para-resolver la atribuyo
a la alta varianza intrínseca de CartPole con DQN (rewards muy ruidosos
episodio a episodio, std ~100) combinada con solo 3 semillas: un par de
episodios afortunados con epsilon alto pueden adelantar mucho la primera
resolución, como parece ocurrir con el base en la semilla 123.

**¿Qué haría distinto con más tiempo?** (1) Más semillas (≥5) y reportar
intervalos de confianza para saber si la ganancia del 15.6 % es significativa;
(2) fijar presupuesto por frames y no por "primer solve", comparando curvas de
aprendizaje completas; (3) ablación del tamaño de los streams V/A y del
`target_net_sync`; (4) añadir un segundo componente Rainbow (p. ej. Double DQN,
que ataca la sobrestimación del `max` y es ortogonal a Dueling) para ver si los
beneficios se suman.
