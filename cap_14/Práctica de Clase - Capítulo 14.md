# Práctica de Laboratorio: Exploración de MiniWoB++ y el Problema de la Navegación Web en RL

---

## 1\. Introducción y Contexto

Hasta el Capítulo 13 hemos explorado entornos discretos estándar (como Atari y CartPole) y mundos textuales (*TextWorld*). En el **Capítulo 14**, el objetivo se traslada a la automatización de interfaces web utilizando **MiniWoB++** (*World of Bits*).

A diferencia de los entornos anteriores, la navegación web presenta desafíos únicos:

* **Espacios de acción y observación multimodales:** La observación combina imágenes en píxeles (RGB), texto con instrucciones (*utterance*) y la estructura de la página (árbol DOM).  
* **Parcial observabilidad (POMDP):** En tareas secuenciales, la captura de pantalla en el tiempo $t$ suele ser insuficiente para decidir de forma óptima si no se cuenta con memoria de las acciones pasadas.  
* **Exploración dispersa:** Las acciones aleatorias casi nunca descubren la combinación correcta de clics y teclas sin guía externa o demostraciones humanas (*human demonstrations*).

---

## 2\. Objetivos de Aprendizaje

Al finalizar esta práctica serás capaz de:

1. Configurar e instanciar entornos de **MiniWoB++** mediante la interfaz moderna de **Gymnasium**.  
2. Inspeccionar y caracterizar las observaciones multimodales y los espacios de acción disponibles en el navegador.  
3. Evaluar cuantitativamente la ineficiencia de una política aleatoria en entornos web.  
4. Identificar por qué ciertas tareas violan la propiedad de Márkov y requieren formulaciones de tipo POMDP.  
5. Capturar o analizar demostraciones humanas estructuradas para tareas secuenciales.

---

## 3\. Requisitos y Configuración Previa

Asegúrate de contar con un entorno de Python 3.8+ con Chrome/Chromium instalado.

Ejecuta en tu terminal o celda de notebook:

```sh
pip install gymnasium miniwob==1.0
```

> **Nota:** MiniWoB++ utiliza Selenium para controlar Chrome en segundo plano. Verifica que tengas Google Chrome instalado en tu sistema.

---

## 4\. Desarrollo de la Actividad

### Parte 1: Inspección de Espacios y Primer Contacto

Crea un script o libreta de Jupyter e inicializa el entorno elemental `click-test-2-v1`:

```py
import time
import gymnasium as gym
import miniwob
from miniwob.action import ActionTypes

# 1. Crear el entorno con renderizado visual para inspección
env = gym.make('miniwob/click-test-2-v1', render_mode='human')
obs, info = env.reset()

print("--- OBSERVACIÓN ---")
print("Llaves en el diccionario de observación:", obs.keys())
print("Instrucción (utterance):", obs.get('utterance'))
print("Dimensiones de la captura (screenshot shape):", obs['screenshot'].shape)

print("\n--- ESPACIO DE ACCIONES ---")
print("Tipo de Action Space:", env.action_space)

env.close()
```

**Preguntas a responder en el reporte:**

1. ¿Qué forma (`shape`) y tipo de datos (`dtype`) tiene la observación visual (`screenshot`)?  
2. ¿Qué información aporta el campo `dom_elements` dentro del diccionario de observación frente a procesar únicamente la imagen RGB?

---

### Parte 2: Agente Aleatorio vs. Entorno Web

Implementa un bucle para evaluar un agente puramente aleatorio (`env.action_space.sample()`) durante **10 episodios** en el entorno `click-test-2-v1` (puedes ejecutarlo con `render_mode=None` para mayor velocidad):

```py
env = gym.make('miniwob/click-test-2-v1', render_mode=None)

num_episodes = 10
rewards = []
durations = []

for ep in range(num_episodes):
    obs, info = env.reset()
    terminated = False
    truncated = False
    total_reward = 0.0
    steps = 0
    
    while not (terminated or truncated):
        # Muestreo de una acción aleatoria
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        steps += 1
        
    rewards.append(total_reward)
    durations.append(steps)

print(f"Recompensa promedio: {sum(rewards)/num_episodes:.4f}")
print(f"Pasos promedio por episodio: {sum(durations)/num_episodes:.2f}")
env.close()
```

**Preguntas a responder en el reporte:** 3\. ¿Cuál fue la tasa de éxito o recompensa media obtenida por el agente aleatorio? 4\. En el Capítulo 14, el autor simplifica el espacio del ratón discretizándolo en una cuadrícula fija (grid de $10\times 10$ píxeles). ¿Por qué esta discretización facilita el entrenamiento en comparación con el espacio continuo original de coordenadas de pantalla?

---

---

### Parte 3: Demostraciones Humanas (*Human Demonstrations*) (15 min)

En el Capítulo 14 se demuestra que el aprendizaje por refuerzo puro (por ejemplo, entrenar A3C desde cero) falla en tareas como `count-sides` o `tic-tac-toe` debido a la dispersión de las recompensas, requiriendo demostraciones humanas:

1. Revisa la herramienta de grabación de demostraciones del repositorio (`record_demo.py` o el módulo nativo `miniwob.scripts.record`).  
2. Responde en tu reporte:  
   * ¿Qué componentes se guardan en el archivo de demostración (JSON) por cada paso ejecutado?  
   * ¿Cómo se incorpora la pérdida por imitación (aprendizaje supervisado) a la función de pérdida del método de Policy Gradients/Actor-Critic?

&nbsp;

&nbsp;

### 📂 Recursos y Enlaces: Grabación de Demostraciones

Para la fase de demostraciones humanas, disponen de dos herramientas:

1. **Grabador oficial de MiniWoB++ (`miniwob.scripts.record`):**  
   1. **Documentación:** [Guía oficial de Demostraciones MiniWoB++](https://miniwob.farama.org/content/demonstrations/)  
   2. **Ejecución:** `python -m miniwob.scripts.record`  
   3. *Nota:* Este script guarda las acciones y el DOM, pero omite los fotogramas de píxeles.  
2. **Grabador con soporte de imágenes del libro (`record_demo.py`):**  
   1. **Repositorio GitHub del libro:** [Deep Reinforcement Learning Hands-On (3E)](https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-3E/)  
   2. **Ubicación del script:** Carpeta `Chapter14/record_demo.py`  
   3. **Demos pregrabadas:** Si no cuentan con interfaz gráfica local, pueden descargar las trayectorias de ejemplo en `Chapter14/demos/`.  
   4. **Ejemplo de ejecución:**

```sh
python Chapter14/record_demo.py -o demos/test -g tic-tac-toe-v1 -d 1
```

&nbsp;

---

## Entrega

1. Enlace a repositorio.  
   2. Un documento con las respuestas breves y concisas a las preguntas 1 a 6\.

### Rúbrica de evaluación (10 pts totales):

* **Configuración y ejecución de código (3 pts):** Instanciación correcta de MiniWoB++ y extracción adecuada de las observaciones/acciones.  
* **Análisis del agente aleatorio (2 pts):** Métricas obtenidas y justificación de la discretización del espacio de acciones.  
* **Discusión POMDP (3 pts):** Claridad en la explicación teórica de la pérdida de la propiedad de Márkov en navegación web y soluciones arquitectónicas.  
* **Comprensión de Demostraciones Humanas (2 pts):** Explicación del papel de las demostraciones en la función objetivo del agente.