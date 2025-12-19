# Desmontamos la IA que Juega a Super Mario: 5 Claves de su 'Cerebro' Digital que No Esperabas

<img width="800" height="447" alt="image" src="https://github.com/user-attachments/assets/cfcca86d-434f-480b-b87c-44a657fe6e1b" />


Que una inteligencia artificial puede aprender a jugar a videojuegos ya no es noticia. Llevamos años viendo cómo algoritmos aplastan récords en toda clase de títulos, desde el ajedrez hasta los juegos de estrategia más complejos. Sin embargo, la mayoría de estos sistemas se basan en la fuerza bruta computacional, probando millones de posibilidades hasta encontrar la estrategia óptima.

<img width="800" height="447" alt="image" src="https://github.com/user-attachments/assets/a9859fa7-ece6-4abb-aeea-43bae321a728" />


Pero, ¿y si una IA pudiera jugar de una forma fundamentalmente distinta? El proyecto "NeuroLogos TRICAMERAL V2.1" se aleja del camino tradicional para explorar una idea radical: construir un agente de IA que no solo aprenda, sino que lo haga a través de una arquitectura explícitamente modelada a partir de un cerebro biológico. En lugar de optimizar algoritmos, los investigadores están cultivando comportamientos, tomando prestado el manual de instrucciones de la propia evolución. Su diseño incorpora conceptos que parecen sacados de la neurociencia, no de la informática, como "hemisferios" especializados, neuronas que experimentan "fatiga", un oído sintético que "escucha" eventos del juego y una memoria que recuerda "vidas" pasadas.

<img width="800" height="447" alt="image" src="https://github.com/user-attachments/assets/a114cde2-d28f-4e94-a266-b8ec33dc5869" />


Este enfoque no busca únicamente superar una puntuación, sino crear una inteligencia más robusta, flexible y, en cierto modo, comprensible. Al imitar los mecanismos que la evolución ha perfeccionado durante millones de años, este agente nos ofrece una fascinante ventana a un nuevo paradigma en la creación de IA. ¿Qué secretos esconde una inteligencia artificial que no solo juega, sino que imita la estructura de nuestro propio pensamiento?

<img width="800" height="447" alt="image" src="https://github.com/user-attachments/assets/45d1fe0c-7918-49cb-88ea-7276889c115b" />

1. Un cerebro en la máquina: Dos hemisferios y un cuerpo calloso
El núcleo de la IA NeuroLogos es su arquitectura "Tricameral", una estructura digital que emula la especialización del cerebro humano. En lugar de una red monolítica, el agente se divide en tres componentes principales, cada uno con un rol definido, tal y como se observa en su código fuente.
• Hemisferio Derecho (RightHemisphere): Es el lado perceptivo e intuitivo. Este componente es el responsable de procesar la información sensorial. Por un lado, analiza los datos visuales del juego a través de un módulo VisualFeatureExtractor. Por otro, interpreta datos "semánticos" (como la posición de Mario, la puntuación o el tiempo) y procesa los "eventos auditivos" generados por el módulo AudioFeatureGenerator. Para procesar esta información, utiliza unas neuronas especializadas de tipo "líquido" (StableLiquidNeuron) que le otorgan una gran plasticidad.
• Hemisferio Izquierdo (LeftHemisphere): Actúa como el centro ejecutivo y lógico. Recibe la información ya integrada por el CorpusCallosum y toma la decisión final sobre qué acción ejecutar. Su tarea es calcular los "Valores Q" para cada movimiento posible, como se visualiza en el gráfico "Valores Q (Hemisferio Izquierdo)" del panel de control. El valor más alto corresponde a la acción que la IA considera óptima en ese instante.
• Cuerpo Calloso (CorpusCallosum): Es el gran integrador. Su función es fusionar las distintas corrientes de información: la visual, la semántica y la auditiva. En lugar de simplemente elegir una, el código revela un proceso mucho más sofisticado: calcula unos pesos de contexto (context_weights) para realizar una fusión ponderada de las tres vías. Los gráficos "Prioridad de Caminos (Callosum)" y "Contribución de Pathways en el tiempo" muestran cómo este componente realiza una síntesis dinámica de la información, mezclando las diferentes percepciones para formar una comprensión unificada y coherente del entorno.
Esta división del trabajo no es un simple capricho de diseño. Es un intento deliberado de crear una inteligencia más flexible, capaz de especializar sus recursos y combinar diferentes tipos de análisis para adaptarse a un entorno tan cambiante como el de Super Mario Bros.
2. Neuronas que "sienten": La lógica de la fatiga y la homeostasis
Una de las características más sorprendentes de esta IA reside en sus neuronas, concretamente en la clase StableLiquidNeuron. A diferencia de las neuronas artificiales convencionales, que son unidades de cálculo estáticas, estas han sido diseñadas con propiedades inspiradas en la biología que regulan su propio comportamiento.
• Homeostasis: Cada neurona líquida intenta mantener un nivel de actividad estable. Tal y como se monitoriza en el gráfico "Neurofisiología Hemisferio Derecho" bajo la métrica Homeostasis, el sistema se autorregula para evitar la sobreexcitación o la inactividad, buscando un equilibrio funcional similar al de las neuronas biológicas.
• Fatiga (fatigue): Quizás el concepto más revolucionario es que tanto las neuronas individuales como las rutas de información pueden "cansarse". El concepto de fatiga se implementa a nivel celular en la propia clase StableLiquidNeuron y, de forma más estratégica, a nivel de sistema en el CorpusCallosum, que mantiene búferes de fatiga (visual_fatigue y semantic_fatigue). Si el sistema depende excesivamente de una fuente de datos, esa vía empieza a fatigarse y su influencia en la decisión final disminuye.

<img width="800" height="447" alt="image" src="https://github.com/user-attachments/assets/21921026-6cb0-4c27-802d-dfe144de2858" />


Este mecanismo evita que la IA se obsesione con una sola estrategia. Si depender únicamente de las señales visuales resulta ineficaz, esa vía neuronal empieza a mostrar "agotamiento", forzando al CorpusCallosum a dar más peso a otras fuentes de información, como los datos semánticos. La implementación de la fatiga fomenta una integración constante de toda la información disponible, resultando en un comportamiento mucho más adaptable y robusto.
4. Un "oído" sintético: Cómo la IA "escucha" eventos sin sonido

<img width="800" height="447" alt="image" src="https://github.com/user-attachments/assets/0e63a3d8-add0-4c93-b155-6836f6d3f27c" />


Aunque el juego de Super Mario Bros tiene sonido, la IA NeuroLogos no lo procesa. Sin embargo, cuenta con su propio "sentido del oído" gracias al módulo AudioFeatureGenerator. Este ingenioso componente no analiza archivos de audio, sino que sintetiza sus propios "eventos sonoros" a partir de cambios en los datos del juego.
La función extract_event_features del código revela exactamente qué "escucha" la IA. En lugar de oír el "cling" de una moneda, detecta que el contador de monedas ha aumentado y genera un coin_event. De forma similar, si la puntuación sube drásticamente, puede inferir un powerup_event (obtención de un champiñón o flor) o un enemy_defeat_event (derrota de un enemigo).

<img width="800" height="447" alt="image" src="https://github.com/user-attachments/assets/d0ec885a-8d45-4e15-aa0f-15ad771e50c7" />


Este concepto es extraordinariamente potente porque representa una forma avanzada de abstracción de datos. La IA no está simplemente añadiendo otro flujo sensorial; está transformando cambios de estado continuos y de bajo nivel (como un número de puntuación que aumenta) en eventos categóricos discretos y de alto nivel. Este proceso es computacionalmente más eficiente y conceptualmente más cercano a cómo un cerebro procesa estímulos complejos, permitiéndole reaccionar a sucesos clave del entorno que podrían pasar desapercibidos en un simple análisis visual.
5. Recuerdos de vidas pasadas: Uso de memoria episódica para decidir mejor
El aprendizaje por refuerzo tradicional se basa en el ensayo y error, pero NeuroLogos va un paso más allá implementando una forma de memoria a largo plazo. El módulo EpisodicMemory permite al agente almacenar y recuperar experiencias clave, de forma similar a como nosotros recordamos eventos específicos.
El sistema no guarda un vídeo de cada éxito. En su lugar, crea "prototipos" abstractos de situaciones importantes. La función store_episode reconoce eventos de alto valor como obtener un 'powerup', derrotar a un 'enemy_defeat' o encontrar un 'secret'. Cuando esto ocurre, no guarda el recuerdo literal, sino que actualiza un prototipo mediante un promedio continuo: (prototipo * contador + estado_actual) / (contador + 1). De esta forma, el agente aprende la "esencia" promediada de cómo se siente una situación de éxito basándose en todas sus experiencias pasadas.

<img width="800" height="447" alt="image" src="https://github.com/user-attachments/assets/28116737-2d7f-4e66-b21c-0ee46459ab81" />


Esta memoria se vuelve una herramienta activa durante la toma de decisiones. La función act consulta constantemente la memoria episódica. Si la situación actual se parece mucho a un prototipo positivo almacenado (por ejemplo, estar cerca de un bloque que en el pasado contenía un champiñón), la IA da un "impulso" a los Valores Q de las acciones que la llevaron a ese éxito anteriormente. Es una forma de aprendizaje mucho más sofisticada, que permite a la IA usar la sabiduría de sus "vidas pasadas" para guiar sus acciones presentes.
6. Leyendo su "mente": Visualizando el foco de atención de la IA

https://cdn-images-1.medium.com/max/800/1*M1_-cm7YsIyIGZS3BMAmvQ.png

Uno de los mayores desafíos de la IA es entender por qué toma ciertas decisiones. A menudo, las redes neuronales son "cajas negras" inescrutables. Sin embargo, NeuroLogos ha sido diseñado para ser transparente, permitiendo a los investigadores visualizar su proceso de "pensamiento" en tiempo real.
La herramienta clave para esto es el "Foco Real", también conocido como mapa de saliencia, generado por el script visualizacion_foco_real.py. Este mapa, visible en el panel de control de Figure_11.png, es una superposición visual sobre la pantalla del juego que muestra en qué píxeles se está "fijando" la IA. Se genera mediante un cálculo (la función compute_real_saliency) que determina qué áreas de la imagen tuvieron la mayor influencia en su decisión final.
De forma crucial, este proceso no es ingenuo. El código implementa una hud_mask, una máscara que reduce deliberadamente la importancia de las zonas de la interfaz de usuario, como la puntuación o el temporizador. Este detalle práctico revela un desafío clave en la interpretabilidad de la IA: los investigadores deben guiar activamente las herramientas de visualización para que se centren en los aspectos relevantes del entorno (el juego) y no en artefactos estáticos (el HUD). Métricas como Cobertura, Pico y Entropía permiten cuantificar la calidad de este foco, proporcionando una ventana directa a su proceso atencional.

<img width="800" height="447" alt="image" src="https://github.com/user-attachments/assets/4077d703-46ee-4bce-8b29-4db9d6a937c7" />


Conclusion
El agente NeuroLogos TRICAMERAL V2.1 es mucho más que otra IA que juega a un videojuego. Es un fascinante experimento que se aleja de la fuerza bruta para adentrarse en un territorio inspirado en la neurociencia. Al tomar prestados principios como la lateralización cerebral, la homeostasis neuronal, la fatiga sináptica y la memoria episódica, el proyecto demuestra que se pueden construir sistemas más robustos e inteligentes.

<img width="800" height="447" alt="image" src="https://github.com/user-attachments/assets/42d960d7-622b-484c-ac1a-bbe36bf23c57" />


Conceptos que hasta ahora pertenecían al dominio de la biología se están convirtiendo en herramientas prácticas para la ingeniería de IA. Este enfoque no solo produce agentes más adaptables, sino también más interpretables, permitiéndonos, literalmente, ver en qué están "pensando". Nos deja con una pregunta inevitable y emocionante: si ya podemos construir IAs con "fatiga" y "memoria", ¿cuál será el próximo rasgo cognitivo que logremos replicar en el silicio?

<img width="800" height="447" alt="image" src="https://github.com/user-attachments/assets/af8d68d1-d4cf-49c3-a857-bb0f5f979978" />

<img width="800" height="447" alt="image" src="https://github.com/user-attachments/assets/fa7ed4d8-99a5-40dc-bb3f-5eee9641d60a" />

<img width="800" height="447" alt="image" src="https://github.com/user-attachments/assets/4c6b7fa7-d707-4553-a94a-9388aacf8401" />

<img width="800" height="447" alt="image" src="https://github.com/user-attachments/assets/63ae6f81-be25-4d31-b4cf-a421483b18b2" />

[https://medium.com/@lazyown.redteam/desmontamos-la-ia-que-juega-a-super-mario-5-claves-de-su-cerebro-digital-que-no-esperabas-3222947f576b?postPublishedType=repub](https://medium.com/@lazyown.redteam/desmontamos-la-ia-que-juega-a-super-mario-5-claves-de-su-cerebro-digital-que-no-esperabas-3222947f576b?postPublishedType=repub)

![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54) ![Shell Script](https://img.shields.io/badge/shell_script-%23121011.svg?style=for-the-badge&logo=gnu-bash&logoColor=white) ![Flask](https://img.shields.io/badge/flask-%23000.svg?style=for-the-badge&logo=flask&logoColor=white) [![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/Y8Y2Z73AV)
