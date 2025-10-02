---
title: 'Explicabilidad de Modelos de Machine Learning: Una Guía sobre LIME, SHAP y Gradcam'
date: 2025-10-02
permalink: /posts/2025/10/explicabilidad-ml-lime-shap-gradcam/
tags:
  - machine learning
  - XAI
  - explicabilidad
  - LIME
  - SHAP
  - Gradcam
excerpt: 'Guía completa sobre técnicas de explicabilidad en modelos de machine learning: LIME, SHAP y Gradcam'
---

**La importancia de la explicabilidad del modelo en machine learning**

En el mundo del machine learning, la explicabilidad del modelo se ha convertido en un aspecto crucial que no puede ser ignorado. A medida que los modelos se vuelven más complejos y poderosos, se vuelve cada vez más importante entender e interpretar sus decisiones.

La importancia de la explicabilidad del modelo es doble. En primer lugar, desde un punto de vista ético, es esencial garantizar que las decisiones tomadas por los modelos de machine learning sean justas, imparciales y no estén influenciadas por variables que no deberían considerarse. Por ejemplo, en decisiones de préstamos o contrataciones, es crucial identificar y prevenir cualquier posible discriminación o sesgo que el modelo pueda exhibir. Al comprender el funcionamiento interno del modelo y su proceso de toma de decisiones, podemos detectar y rectificar cualquier problema que pueda surgir.

En segundo lugar, la explicabilidad del modelo nos permite construir confianza y credibilidad en las predicciones y recomendaciones realizadas por los modelos de machine learning. Cuando un modelo proporciona una salida, no es suficiente aceptarla ciegamente sin entender cómo y por qué llegó a esa conclusión. Al proporcionar información sobre los factores y características que influyeron en la decisión del modelo, podemos evaluar la confiabilidad y precisión de las predicciones. Esta transparencia no solo mejora la confianza, sino que también permite una mejor colaboración y toma de decisiones entre humanos y máquinas.

Han surgido varias técnicas para abordar el desafío de la explicabilidad del modelo, incluyendo LIME, SHAP y Gradcam. Estas técnicas proporcionan diferentes enfoques para comprender e interpretar las decisiones tomadas por los modelos de machine learning.

## 1. LIME (Local Interpretable Model-Agnostic Explanations)

LIME, que significa Local Interpretable Model-Agnostic Explanations (Explicaciones Locales Interpretables Agnósticas al Modelo), es una técnica poderosa que ayuda a arrojar luz sobre la naturaleza de caja negra de los modelos complejos de machine learning.

LIME proporciona una explicación local para predicciones individuales al aproximar los límites de decisión intrincados de un modelo utilizando un modelo más simple e interpretable, como la regresión lineal. Al hacerlo, nos permite entender qué características o entradas contribuyeron más a una predicción particular.

La idea básica detrás de LIME es generar un vecindario local alrededor de una instancia específica de interés y perturbar las características en este vecindario para observar el impacto resultante en la salida del modelo. Estas perturbaciones ayudan a aproximar el comportamiento del modelo en las cercanías de la instancia y nos permiten atribuir importancia a diferentes características.

Una de las ventajas clave de LIME es su naturaleza agnóstica al modelo. Puede aplicarse a cualquier modelo de machine learning, independientemente de si es una red neuronal profunda, un random forest o una máquina de vectores de soporte. Esta flexibilidad hace de LIME una opción popular entre los científicos de datos e investigadores que trabajan con diversos modelos.

Además, LIME proporciona explicaciones que son tanto interpretables como visualmente atractivas. Resalta las características importantes y las presenta en un formato comprensible para humanos, como nubes de palabras para datos de texto o mapas de calor para datos de imagen. Estas explicaciones visuales facilitan que las partes interesadas comprendan el razonamiento detrás de las decisiones de un modelo.

### 1.1 Cómo funciona LIME

En su núcleo, LIME opera sobre el principio de interpretabilidad agnóstica al modelo. Esto significa que puede aplicarse a cualquier modelo de caja negra, independientemente de su arquitectura subyacente o complejidad. LIME logra esto aproximando el comportamiento del modelo original localmente alrededor de un punto de datos específico de interés.

El proceso comienza seleccionando una instancia que queremos explicar. LIME luego genera un vecindario local de instancias perturbadas alrededor de este punto. Estas perturbaciones se crean mediante el muestreo y la modificación de las características de la instancia original mientras se mantiene fija la etiqueta. Las instancias modificadas se utilizan luego para crear un modelo simplificado e interpretable.

Este modelo simplificado, a menudo un modelo lineal, se entrena para aproximar el comportamiento del modelo original dentro del vecindario local. Los pesos asignados a cada característica en este modelo simplificado representan su importancia para determinar la predicción del modelo original. Al analizar estos pesos de características, podemos obtener información valiosa sobre el proceso de toma de decisiones del modelo de caja negra.

Para evaluar la importancia de cada característica, LIME emplea una medida llamada "importancia de perturbación". Calcula las diferencias en las predicciones entre las instancias modificadas y las instancias originales mientras considera los pesos asignados por el modelo simplificado. Cuanto mayor sea la diferencia en las predicciones, más importante se considera la característica correspondiente.

LIME también introduce el concepto de "fidelidad local" para evaluar la confiabilidad de las explicaciones. Cuantifica qué tan bien el modelo simplificado aproxima el comportamiento del modelo original para la instancia específica que se está explicando. Esto ayuda a los usuarios a comprender la confiabilidad y precisión de la interpretabilidad proporcionada por LIME.

### Ejemplo con el Dataset de Diabetes

Para ilustrar cómo funciona LIME en la práctica, consideremos el clasificador Random Forest entrenado en el dataset de diabetes. Supongamos que queremos explicar por qué el modelo predijo que una paciente específica tiene diabetes.

**Instancia a explicar:**

- Glucosa: 104
- Edad: 38 años
- Embarazos: 13
- BMI: 31.2
- Otras características con valores específicos

**Proceso de LIME:**

1. **Generación del vecindario:** LIME crea múltiples versiones "perturbadas" de esta paciente:
   - Paciente ficticia 1: Glucosa=100, Edad=40, Embarazos=12...
   - Paciente ficticia 2: Glucosa=110, Edad=35, Embarazos=14...
   - Y así sucesivamente (cientos de variaciones)

2. **Predicción en el vecindario:** El modelo Random Forest predice para cada una de estas pacientes ficticias.

3. **Modelo local simple:** LIME entrena un modelo de regresión lineal simple usando solo estas variaciones locales. Este modelo lineal "imita" al Random Forest pero solo en el vecindario de nuestra paciente.

4. **Explicación resultante:**

> Predicción: 72% probabilidad de diabetes
>
> Factores que aumentan el riesgo:

- Glucosa alta (99-104): +0.12
- SkinThickness bajo: +0.03
- BloodPressure (70-72): +0.01

> Factores que disminuyen el riesgo:

- Edad relativamente joven (29-40): -0.08
- Muchos embarazos (>6): -0.07
- BMI moderado (27-32): -0.01

<div style="text-align: center;">
  <img src="/images/posts/LIME_explanation_for_the_8th_instance_from_test_dataset.avif" alt="explanation_for_the_8th_instance" />
  <p><em>Explicación de la instancia número 8</em></p>
</div>

**Interpretación intuitiva:**

La glucosa elevada es el factor más determinante para que el modelo prediga diabetes en este caso. Aunque la paciente tiene varios factores protectores (edad relativamente joven, historial de embarazos), la glucosa de 104 mg/dL es suficientemente alta para inclinar la balanza hacia un diagnóstico positivo.

Lo interesante es que LIME nos muestra que **para esta paciente específica**, tener muchos embarazos en realidad reduce su probabilidad de diabetes según el modelo (algo contraintuitivo). Esto podría indicar interacciones complejas que el Random Forest capturó en los datos de entrenamiento.

### 1.2 Casos de uso y beneficios de LIME

Un caso de uso clave de LIME es en el campo de la salud. Cuando se trata de tomar decisiones críticas sobre la salud de un paciente, los profesionales de la salud deben tener una comprensión profunda de los factores que contribuyen a la predicción de un modelo. LIME puede proporcionar información sobre qué características están impulsando la decisión del modelo, permitiendo a los médicos validar y comprender el razonamiento detrás de las predicciones. Esto puede ayudar a mejorar la confianza en el modelo y ayudar a desarrollar herramientas de diagnóstico más precisas y confiables.

En el ámbito de las finanzas, LIME puede emplearse para explicar las predicciones realizadas por modelos de calificación crediticia. Comprender los factores que contribuyen a la solvencia crediticia de una persona es esencial tanto para prestamistas como para prestatarios. LIME puede arrojar luz sobre las variables que tienen la mayor influencia en las puntuaciones de crédito, proporcionando transparencia y equidad en el proceso de toma de decisiones. Esto puede ayudar a prevenir sesgos y garantizar que las personas no sean injustamente privadas de acceso a oportunidades financieras.

Otro beneficio valioso de LIME es su capacidad para ayudar en la depuración de modelos de machine learning. Cuando un modelo produce predicciones inesperadas o erróneas, puede ser desafiante identificar la causa raíz del problema. LIME ayuda en este sentido al resaltar las características específicas que están contribuyendo a la salida incorrecta. Esto permite a los desarrolladores y científicos de datos identificar y rectificar fallas, conduciendo a modelos más confiables y precisos.

### 1.3 Limitaciones de LIME

Si bien LIME es una herramienta poderosa para la explicabilidad del modelo, tiene sus limitaciones. Comprender estas limitaciones es crucial para obtener una comprensión integral del panorama de interpretabilidad.

En primer lugar, LIME se basa en perturbar las características de entrada para crear explicaciones locales. Esto significa que las explicaciones proporcionadas por LIME solo son válidas dentro del vecindario local de la instancia que se está explicando. En consecuencia, las explicaciones pueden no generalizarse bien a diferentes regiones del espacio de características. Es esencial interpretar los resultados de LIME con esta restricción de localidad en mente.

Otra limitación de LIME es su sensibilidad a los hiperparámetros. La elección del ancho del kernel y el número de muestras utilizadas para la perturbación pueden afectar significativamente las explicaciones generadas. Diferentes configuraciones de hiperparámetros pueden conducir a resultados variados, lo que hace esencial ajustar cuidadosamente estos parámetros para un rendimiento óptimo.

Además, LIME puede tener dificultades con datos de alta dimensionalidad. A medida que aumenta el número de características, la interpretabilidad y estabilidad de las explicaciones de LIME pueden disminuir. En tales casos, técnicas alternativas como SHAP (SHapley Additive exPlanations) o Gradcam (Gradient-weighted Class Activation Mapping) pueden ser más adecuadas para capturar la importancia de las características con precisión.

Por último, es importante reconocer que las explicaciones de LIME pueden ser susceptibles a sesgos presentes en los datos de entrenamiento. Si los datos de entrenamiento están sesgados o desequilibrados, LIME puede inadvertidamente resaltar estos sesgos, conduciendo a explicaciones potencialmente engañosas. Por lo tanto, es crucial considerar la calidad de los datos subyacentes y la equidad al interpretar los resultados de LIME.

## 2. SHAP (SHapley Additive exPlanations)

SHAP es una técnica popular agnóstica al modelo que proporciona información sobre cómo los valores de características individuales contribuyen a la predicción realizada por un modelo de machine learning.

En su núcleo, SHAP se basa en los valores de Shapley de la teoría de juegos cooperativos. Asigna un valor a cada característica calculando la contribución que la característica hace a la predicción cuando se combina con otras características. Al considerar todas las combinaciones posibles de características, SHAP evalúa la importancia de cada característica de manera justa y consistente.

Uno de los beneficios clave de usar SHAP es su capacidad para generar explicaciones tanto a nivel local como global. A nivel local, los valores SHAP cuantifican el impacto de cada característica en la predicción de una instancia específica. Esto ayuda a comprender por qué se hizo una predicción particular y permite a los usuarios validar el comportamiento del modelo.

A nivel global, SHAP proporciona una visión general de la importancia de las características en todo el conjunto de datos. Esto permite a los profesionales identificar qué características tienen la influencia más significativa en las predicciones del modelo. Al comprender la importancia global de las características, uno puede obtener información sobre los patrones y relaciones subyacentes dentro de los datos.

Además, SHAP es versátil y puede aplicarse a una amplia gama de modelos de machine learning, incluidos modelos complejos como redes neuronales profundas. También considera las interacciones entre características, proporcionando una comprensión más completa del proceso de toma de decisiones del modelo.

### 2.1 Explicación de los valores SHAP

Los valores SHAP proporcionan una forma de interpretar el impacto de cada característica en una predicción individual realizada por un modelo. Ofrecen una medida cuantitativa de la importancia de las características y ayudan a explicar el razonamiento detrás de la salida de un modelo.

Para decirlo simplemente, los valores SHAP asignan un valor numérico a cada característica, indicando cuánto contribuyó esa característica a una predicción específica. Los valores pueden ser positivos o negativos, indicando si la característica influyó positiva o negativamente en la predicción.

Una de las ventajas clave de los valores SHAP es su capacidad para manejar modelos complejos, incluidos métodos de ensamble y modelos de aprendizaje profundo. Proporcionan un marco unificado para interpretar predicciones en diferentes arquitecturas de modelos.

El cálculo de los valores SHAP se basa en la teoría de juegos, específicamente en los valores de Shapley, que fueron originalmente desarrollados para juegos cooperativos. En el contexto del machine learning, los valores SHAP distribuyen el "crédito" de una predicción entre las diferentes características según sus contribuciones.

Al comprender los valores SHAP para una predicción particular, podemos obtener información sobre qué características tuvieron el impacto más significativo. Este conocimiento puede ser invaluable para diversos propósitos, como identificar factores influyentes que impulsan las predicciones, detectar sesgos en el modelo o explicar el comportamiento del modelo a las partes interesadas.

En términos prácticos, podemos visualizar los valores SHAP utilizando gráficos como summary plots o gráficos de atribución individual. Estas visualizaciones ayudan a resaltar la importancia relativa de las características y proporcionan una comprensión clara de cómo contribuyen a las predicciones.

### Ejemplo con el Dataset de Diabetes

Para entender mejor SHAP, analicemos algunas gráficas que se obtienen del modelo entrenado con el dataset de diabetes:

#### 2.1.1 Variable Importance Plot - Global Interpretation

<div style="text-align: center;">
  <img src="/images/posts/Summary_Plot_showing_Important_Variables.avif" alt="Summary Plot" />
  <p><em>Summary Plot mostrando Atributos Importantes</em></p>
</div>

Esta gráfica nos mostró que:

- **Glucose** es la variable más importante para el modelo en general
- **Age** y **BMI** también tienen un impacto significativo
- **BloodPressure** e **Insulin** tienen una importancia relativamente baja

**Interpretación con valores SHAP:**

Los valores SHAP nos dicen que, en promedio absoluto:

- Glucose puede cambiar las predicciones del modelo en aproximadamente ±0.25 puntos de probabilidad
- Age puede cambiar las predicciones en aproximadamente ±0.15 puntos
- BloodPressure raramente cambia las predicciones en más de ±0.05 puntos

**Separación por clase:**

- La barra azul (Class 1 - Diabetes) para Glucose es muy larga: significa que valores altos de glucosa aumentan fuertemente la probabilidad de diabetes
- La barra roja (Class 0 - No diabetes) también es considerable: valores bajos de glucosa disminuyen fuertemente la probabilidad de diabetes
- Para BloodPressure, ambas barras son pequeñas: el modelo casi no usa esta variable para tomar decisiones

#### 2.1.2 Summary Plot - Análisis detallado

<div style="text-align: center;">
  <img src="/images/posts/Summary_Plot_deep_dive_on_label_1.avif" alt="Summary Plot on Label 1" />
  <p><em>Summary Plot para el Label 1</em></p>
</div>

Esta visualización nos muestra cada observación individual del dataset. Recordemos los patrones clave:

**Para Glucose:**

- Cada punto representa una persona del dataset
- Puntos rojos (glucosa alta) → SHAP muy positivo (~0.3-0.4): "Esta persona tiene glucosa alta, aumentemos mucho su probabilidad de diabetes"
- Puntos azules (glucosa baja) → SHAP negativo (~-0.2 a -0.3): "Esta persona tiene glucosa baja, reduzcamos su probabilidad de diabetes"
- **Patrón claro:** El modelo confía fuertemente en la glucosa y la usa de manera consistente

**Para BloodPressure:**

- Los puntos están concentrados cerca de cero
- Colores mezclados: no hay un patrón claro
- **Significado:** El modelo "aprendió" que la presión arterial no es un buen predictor de diabetes en este dataset
- Incluso personas con presión arterial muy alta o muy baja reciben valores SHAP cercanos a cero

**Para Age:**

- Valores intermedios muestran dispersión
- Hay una transición gradual de valores negativos (jóvenes) a positivos (mediana edad)
- Los colores cambian gradualmente, sugiriendo interacciones con otras variables como Glucose

#### 2.1.3 Dependence Plot de Age

<div style="text-align: center;">
  <img src="/images/posts/Dependence_plot_for_Age_attribute.avif" alt="Dependence Plot for Age" />
  <p><em>Dependence Plot para Atributo Age</em></p>
</div>

Este gráfico nos revela la relación no lineal entre edad y predicción:

**Zona de edades bajas (20-30):**

- SHAP fuertemente negativo (-0.10 a -0.15)
- Puntos azules predominan
- **Interpretación:** Ser joven es "protector" contra diabetes, especialmente si además tienes otras variables favorables (indicadas por el color azul, probablemente glucosa baja)

**Zona de transición (30-40):**

- SHAP cruza de negativo a positivo
- Colores se vuelven morados/intermedios
- **Interpretación:** Esta es la edad donde el riesgo cambia. El modelo detectó que alrededor de los 35-40 años el riesgo de diabetes comienza a aumentar en el dataset

**Zona de máximo riesgo (40-55):**

- SHAP máximo positivo (+0.10 a +0.15)
- Colores más variados (morados y magentas)
- **Interpretación:** Esta es la "edad peligrosa" según el modelo. Sin embargo, el color variable sugiere que el efecto de la edad depende de otras variables. Una persona de 45 con glucosa baja (azul) tendrá un impacto menor que una de 45 con glucosa alta (magenta)

**Zona de edades altas (55-75):**

- SHAP disminuye gradualmente
- Puntos rojos/magentas predominan
- **Interpretación:** Contraintuitivamente, el efecto de la edad disminuye en adultos mayores. ¿Por qué? Probablemente porque en este rango de edad, otras variables como Glucose y BMI dominan la predicción. El modelo "delega" la decisión a esas variables más informativas

**Interacción visible:**

El cambio de colores (azul → morado → magenta → rojo) a través del eje X nos dice:

- Cuando alguien es joven (20-30), típicamente también tiene glucosa baja (azul)
- Cuando alguien es mayor (60-70), típicamente también tiene glucosa alta (rojo)
- Esta correlación en los datos hace que el modelo tenga que "dividir el crédito" entre edad y glucosa

### 2.2 Cómo funciona SHAP en la práctica

Para entender cómo funciona SHAP en la práctica, consideremos nuestro ejemplo del clasificador Random Forest para diabetes. Con SHAP, podemos determinar cómo cada una de las características (número de embarazos, glucosa, edad, BMI, etc.) impacta las predicciones del modelo.

Primero, SHAP genera un valor base, que representa la predicción esperada cuando no se consideran características. Luego, evalúa el efecto de cada característica en las predicciones del modelo incluyéndolas o excluyéndolas sistemáticamente. Al hacerlo, determina la contribución de cada característica a la predicción final.

Los valores SHAP generados pueden ser positivos o negativos, indicando si una característica aumenta o disminuye la predicción del modelo. Además, la magnitud del valor SHAP representa la importancia o influencia de una característica particular. Las características con valores SHAP absolutos más altos tienen un impacto más fuerte en las predicciones del modelo.

Visualizar los valores SHAP puede proporcionar información más profunda sobre el comportamiento de un modelo. Los summary plots de SHAP, por ejemplo, muestran el impacto general de cada característica en las predicciones, permitiéndonos identificar qué características están impulsando más las decisiones del modelo. Los gráficos de valores SHAP individuales proporcionan un desglose detallado de cómo cada característica contribuye a predicciones específicas.

### 2.3 Ventajas y aplicaciones de SHAP

Una de las ventajas clave de SHAP es su capacidad para proporcionar explicaciones locales e individualizadas para cada predicción realizada por un modelo. Esto significa que en lugar de solo entender la importancia general de las características en un modelo, SHAP nos permite entender el impacto de cada característica en una predicción específica.

Además, SHAP es una técnica agnóstica al modelo, lo que significa que puede aplicarse a cualquier modelo de caja negra, ya sea una red neuronal profunda compleja o un simple árbol de decisión. Esta flexibilidad hace de SHAP una herramienta valiosa para una amplia gama de aplicaciones en diversas industrias.

Además de sus ventajas en interpretabilidad de modelos, SHAP también encuentra aplicaciones prácticas en selección de características, depuración de modelos y comparación de modelos. Al comprender el impacto de diferentes características en las predicciones del modelo, los profesionales pueden tomar decisiones informadas sobre ingeniería y selección de características, lo que conduce a un mejor rendimiento del modelo.

**Aplicación en selección de características para diabetes:**

- Si vemos que BloodPressure e Insulin tienen valores SHAP consistentemente cercanos a cero
- Podríamos considerar eliminar estas características para simplificar el modelo
- Esto reduce el riesgo de sobreajuste y hace el modelo más interpretable

Además, SHAP puede usarse para depurar modelos identificando instancias donde el comportamiento del modelo podría ser inconsistente o inesperado. Al analizar los valores SHAP, los profesionales pueden identificar áreas problemáticas y hacer los ajustes necesarios para garantizar la confiabilidad y consistencia del modelo.

**Ejemplo de depuración:** Si encontramos que el modelo predice diabetes para una persona de 25 años con glucosa normal pero muchos embarazos, los valores SHAP nos mostrarían exactamente qué característica está causando esta predicción sospechosa. Podríamos descubrir que el modelo está dando demasiado peso a Pregnancies, lo que indicaría un problema en el entrenamiento.

Por último, SHAP facilita la comparación de modelos al proporcionar un marco unificado para evaluar y contrastar diferentes modelos. Al comparar los valores SHAP entre modelos, los profesionales pueden obtener información sobre las similitudes y diferencias en sus procesos de toma de decisiones, permitiéndoles elegir el modelo más adecuado para sus necesidades específicas.

### 2.4 Desafíos y consideraciones con SHAP

Si bien SHAP es una herramienta poderosa para la explicabilidad del modelo, viene con su propio conjunto de desafíos y consideraciones. Es crucial estar consciente de estos factores al utilizar SHAP para interpretar y comprender nuestros modelos de machine learning.

Un desafío con SHAP es la complejidad computacional que introduce. Calcular valores de Shapley puede ser consumidor de tiempo, especialmente para modelos complejos con un gran número de características. Esto puede convertirse en un cuello de botella al intentar explicar modelos con datos de alta dimensionalidad o al tratar con conjuntos de datos grandes.

**En nuestro ejemplo de diabetes:**

- El dataset tiene 8 características, lo cual es manejable
- Para calcular valores SHAP exactos, el algoritmo debe considerar todas las combinaciones posibles de características: 2^8 = 256 combinaciones
- Si tuviéramos 20 características, serían 2^20 = más de un millón de combinaciones
- Para datasets grandes, se usan aproximaciones (como TreeSHAP para Random Forests) que son más rápidas pero ligeramente menos precisas

Otra consideración es la interpretabilidad de los valores SHAP en sí mismos. Si bien SHAP proporciona una medida cuantitativa de la importancia de las características, entender el significado exacto de estos valores puede ser complicado. Interpretar los valores SHAP requiere un análisis cuidadoso y conocimiento del dominio para traducirlos en información accionable. Es importante evitar la malinterpretación y asegurarse de que las explicaciones proporcionadas por SHAP se alineen con los objetivos previstos del modelo.

**Ejemplo de interpretación cuidadosa:**

- Un valor SHAP de +0.25 para Glucose no significa que la glucosa cause diabetes
- Significa que el modelo aumenta la predicción en 0.25 puntos debido a este valor de glucosa
- Esto refleja correlaciones en los datos de entrenamiento, no necesariamente causalidad

Además, SHAP puede no siempre capturar el contexto completo de las interacciones de características. En ciertos casos, la naturaleza aditiva de los valores SHAP puede simplificar en exceso las relaciones complejas entre características, conduciendo a explicaciones incompletas. Es crucial validar y hacer referencias cruzadas de las explicaciones de SHAP con otras técnicas de interpretabilidad para obtener una comprensión integral del comportamiento del modelo.

**Limitación en interacciones complejas:** En nuestro Dependence Plot de Age, vimos que el efecto de la edad cambia según otras variables (mostrado por los colores). SHAP captura esto parcialmente, pero la visualización aditiva puede no mostrar toda la complejidad. Por ejemplo:

- El efecto de Age podría depender no linealmente de la combinación de Glucose Y BMI simultáneamente
- SHAP considera interacciones de pares, pero no todas las interacciones de orden superior

Por último, la elección del conjunto de datos de fondo utilizado en los cálculos de SHAP es esencial. El conjunto de datos de fondo sirve como punto de referencia para calcular los valores de Shapley y puede impactar significativamente las explicaciones resultantes. Se debe tener cuidado al seleccionar un conjunto de datos de fondo representativo y apropiado que se alinee con la distribución de los datos que se están explicando. No hacerlo puede conducir a explicaciones sesgadas o engañosas.

## 3. Gradcam (Gradient-weighted Class Activation Mapping)

Gradcam, abreviatura de Gradient-weighted Class Activation Mapping, es una técnica popular de explicabilidad de modelos que ha ganado atención en los últimos años. Ofrece varias ventajas que la convierten en una herramienta valiosa en el campo de la interpretabilidad. Gradcam es una herramienta poderosa que nos ayuda a visualizar y entender las regiones o características más importantes dentro de una imagen que conducen a la predicción de un modelo. Proporciona información sobre qué partes de una imagen de entrada contribuyeron más a determinar la salida final.

El concepto detrás de Gradcam es utilizar los gradientes que fluyen hacia la última capa convolucional de una red neuronal para generar un mapa de calor que resalta las regiones de interés. Al analizar los gradientes, Gradcam asigna pesos de importancia a diferentes píxeles, indicando su contribución a la predicción.

Esta técnica es particularmente valiosa cuando se trata de modelos basados en imágenes, como detección de objetos o clasificación de imágenes. No solo nos ayuda a entender por qué un modelo hizo una predicción específica, sino que también nos permite validar el rendimiento del modelo e identificar posibles sesgos o limitaciones.

Al visualizar el mapa de calor generado por Gradcam, podemos obtener información sobre qué áreas de una imagen el modelo se enfoca para tomar sus decisiones. Esta información puede ser crucial en diversas aplicaciones, como imágenes médicas, donde comprender el razonamiento detrás del diagnóstico de un modelo es de suma importancia.

Además, Gradcam puede usarse en conjunto con otras técnicas de explicabilidad de modelos como LIME (Local Interpretable Model-agnostic Explanations) y SHAP (Shapley Additive Explanations) para proporcionar una comprensión integral del comportamiento y las predicciones de un modelo.

### 3.1 Cómo Gradcam mejora la interpretabilidad del modelo

Al visualizar las áreas de enfoque del modelo, Gradcam nos permite comprender el razonamiento detrás de las predicciones del modelo. Esencialmente, resalta las regiones de la entrada que contribuyeron significativamente a la salida final, desmitificando así el proceso de toma de decisiones.

El principio subyacente detrás de Gradcam radica en su capacidad para aprovechar los gradientes que fluyen a través del modelo durante la retropropagación. Calcula la importancia de cada píxel o característica atribuyendo pesos basados en los gradientes de la salida deseada con respecto a los mapas de características de la última capa convolucional.

Esta técnica proporciona una visualización de mapa de calor, superponiendo la imagen de entrada original con regiones codificadas por colores que representan las características salientes que influyeron en la decisión del modelo. Este mapa de calor nos permite identificar las áreas o patrones específicos en los que el modelo se enfocó, facilitando una comprensión más profunda de su proceso de decisión.

### 3.2 Implementando Gradcam en un modelo de aprendizaje profundo

Para implementar Gradcam, primero necesitamos un modelo de aprendizaje profundo entrenado y una capa específica de interés dentro de ese modelo. Esta capa debe capturar características de alto nivel que sean relevantes para la tarea en cuestión. Por ejemplo, en una red neuronal convolucional (CNN) para clasificación de imágenes, la capa convolucional final o la capa de pooling promedio global se elige comú
