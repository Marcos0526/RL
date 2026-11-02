# Notas de Estudio: Navegación Web con Aprendizaje por Refuerzo y MiniWoB++

**Curso:** Aprendizaje por Refuerzo (Reinforcement Learning)  
**Texto de referencia:** Lapan, M., *Deep Reinforcement Learning Hands-On* (3ra Edición), Capítulo 14: *Web Navigation*.  
**Objetivo:** Revisar los conceptos teóricos y técnicos necesarios antes de la sesión práctica de laboratorio.

---

## 1. De Atari a la Web: Formulación del Problema

Hasta el Capítulo 13 se abordaron problemas de control con estados compactos (CartPole), imágenes de videojuegos de arcade clásicos (Atari 2600) y mundos textuales sintéticos (*TextWorld*). En el **Capítulo 14**, el entorno de interacción pasa a ser un **navegador web real**.

La automatización de interfaces web mediante RL modela la interacción humana habitual (rellenar formularios, buscar información, pulsar botones, interactuar con listas desplegables):

* **Entorno:** Una instancia de un navegador web (manejada comúnmente vía Selenium y Chromium/Firefox en modo *headless*).
* **Observación ($S_t$):** La información renderizada en el cliente web, la cual es intrínsecamente **multimodal**:
  * Píxeles de la pantalla renderizada (RGB).
  * Estructura jerárquica de la página (árbol DOM - *Document Object Model*).
  * Consigna o meta textual en lenguaje natural (*utterance*).
* **Acción ($A_t$):** Eventos de interfaz de usuario de bajo o alto nivel:
  * Coordenadas de clic del ratón $(x, y)$.
  * Escritura de secuencias de caracteres mediante teclado.
  * Clics discretos dirigidos por identificadores de nodos HTML (`ref`).
* **Recompensa ($R_t$):** Señal escasa (*sparse reward*) emitida al completar la meta asignada (generalmente $+1.0$ al tener éxito, $-1.0$ o $0$ en fallo o exceso de tiempo), comúnmente penalizada con un factor proporcional al tiempo consumido.

---

## 2. El Benchmark MiniWoB y MiniWoB++

### Origen e Historia
* **MiniWoB (World of Bits):** Desarrollado originalmente por OpenAI (2016) con más de 80 microtareas web estandarizadas. En su versión preliminar, la interacción requería contenedores Docker y el protocolo de escritorio remoto VNC (*Virtual Network Computing*), lo que generaba latencia y sobrecarga de procesamiento.
* **MiniWoB++:** Versión actualizada mantenida por la comunidad científica y la **Farama Foundation** (integrada al ecosistema Gymnasium). Reemplaza VNC por enlaces directos mediante Selenium WebDrivers en entornos locales o *headless*, optimizando drásticamente la tasa de fotogramas por segundo (FPS) y la estabilidad del entrenamiento.

### Ejemplos representativos de tareas
1. `click-button`: Hacer clic en el botón solicitado (ej. "click ONE").
2. `choose-list`: Seleccionar el elemento correcto de una lista desplegable.
3. `login-user`: Introducir credenciales sintéticas en los campos de usuario y contraseña y presionar *Submit*.
4. `click-button-sequence`: Hacer clic en una serie de botones en un orden estricto predeterminado.

---

## 3. Espacios de Observación y de Acción

### 3.1 Espacio de Observaciones (Estructura Multimodal)
En MiniWoB++, `env.reset()` y `env.step()` devuelven un diccionario `gymnasium.spaces.Dict` con varios campos clave:

| Clave del Diccionario | Tipo / Dimensión | Descripción |
| :--- | :--- | :--- |
| `'screenshot'` | `Box(0, 255, (210, 160, 3), uint8)` | Captura de imagen RGB del lienzo activo del navegador. |
| `'utterance'` | `Text` o `str` | Instrucción textual de la tarea (ej. *"Select item X and click Submit"*). |
| `'dom_elements'` | `Sequence[dict]` | Nodos del árbol DOM activos: identificador único (`ref`), etiqueta HTML, texto visible y caja contenedora (`left`, `top`, `width`, `height`). |

### 3.2 Espacio de Acciones
En su formulación matemática más pura, el espacio de acciones de una interfaz gráfica es inmenso (coordenadas continuas de ratón más combinaciones de teclas). Para viabilizar el aprendizaje, comúnmente se adoptan dos enfoques:
1. **Discretización espacial (Grid / Bins):** Se divide la ventana de $160 \times 210$ píxeles en celdas cuadradas (ej. cuadrículas de $10 \times 10$ píxeles), transformando el espacio en un conjunto discreto de regiones cliqueables:
   $$\text{índice\_acción} = y_{\text{bin}} \times \text{num\_columnas} + x_{\text{bin}}$$
2. **Acciones estructuradas por DOM:** El agente selecciona directamente el identificador de nodo (`ref`) sobre el cual ejecutar una primitiva de interacción (`click`, `type`, `press_key`).

---

## 4. Parcial Observabilidad y el Dilema POMDP

En los capítulos iniciales (Cap. 1 y 6) se estudiaron los **Procesos de Decisión de Márkov (MDP)**, los cuales satisfacen la propiedad markoviana:
$$\mathbb{P}(S_{t+1} = s', R_{t+1} = r \mid S_t = s_t, A_t = a_t, \dots, S_0 = s_0, A_0 = a_0) = \mathbb{P}(S_{t+1} = s', R_{t+1} = r \mid S_t = s_t, A_t = a_t)$$

En muchas tareas web, **esta propiedad se viola**, transformando el entorno en un **POMDP** (*Partially Observable Markov Decision Process*):

* **Ambigüedad temporal:** En tareas que exigen secuencias de acciones (por ejemplo, en `click-button-sequence`), la captura de pantalla estática tras haber hecho el primer clic puede lucir visualmente idéntica al estado inicial.
* **Falla de políticas reactivas:** Un agente puramente reactivo (sin memoria temporal) que mapea $O_t \to A_t$ no tiene la capacidad de determinar si debe ejecutar el paso 1 o el paso 2 si la observación $O_t$ no refleja el historial de eventos previos.
* **Soluciones arquitectónicas:**
  1. Concatenación temporal de fotogramas pasados (*Frame Stacking*).
  2. Inclusión de memoria recurrente en la red neuronal (capas **RNN / LSTM** o bloques de atención autorregresivos).
  3. Modificación del vector de estado para concatenar variables de flag o historial de clics ejecutados.

---

## 5. El Desafío de la Exploración y las Demostraciones Humanas

### ¿Por qué falla la exploración aleatoria pura?
Imaginemos una tarea que requiere ingresar un texto de 4 letras o seleccionar 3 opciones específicas en orden secuencial:
* El volumen del espacio de búsqueda conjunto crece de forma exponencial con el horizonte temporal del episodio.
* La probabilidad de que una política aleatoria uniforme ($\pi_{\text{random}}$) encuentre una secuencia exitosa y reciba la recompensa $+1.0$ es cercana a cero.
* Al no recibir recompensas intermedias (*reward sparsity*), el gradiente de optimización o la actualización por diferencias temporales es prácticamente nulo.

### Demostraciones Humanas (*Human Demonstrations*)
Para eludir este cuello de botella, el Capítulo 14 introduce el uso de **trayectorias demostradas por humanos**:
1. **Registro:** El entorno permite registrar partidas exitosas ejecutadas por un operador humano mediante scripts como `miniwob.scripts.record`, generando archivos con trayectorias estructuradas en formato JSON.
2. **Pre-entrenamiento / Imitación (*Imitation Learning*):** El agente entrena inicialmente supervisando sus acciones contra los ejemplos humanos (minimizando la entropía cruzada entre la acción del agente y la acción humana).
3. **RL Fine-tuning:** Una vez que la política se encuentra en una región del espacio de estados cercana al éxito, se continúa el entrenamiento mediante algoritmos de RL estándar (como A2C, PPO o DQN) para optimizar la velocidad y robustez de la ejecución.

---

## 6. Glosario Rápido de Términos

* **DOM (Document Object Model):** Representación jerárquica en árbol de todos los elementos HTML, estilos y textos presentes en una página web.
* **Utterance:** Instrucción textual explícita que define la meta que el agente debe resolver en el episodio en curso.
* **Headless Browser:** Instancia del navegador web que se ejecuta en memoria y en segundo plano sin desplegar una ventana gráfica en pantalla, ideal para servidores de cálculo y entrenamiento masivo.
* **POMDP:** Proceso de decisión en el cual la observación del agente representa únicamente una proyección parcial y ruidosa del estado verdadero del sistema.