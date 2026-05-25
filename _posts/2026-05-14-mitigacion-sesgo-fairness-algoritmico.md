---
title: 'Mitigación del Sesgo en IA: Adversarial Learning, Meta-Fair Classifier y Exponentiated Gradient Reduction'
date: 2025-11-13
permalink: /posts/2025/11/mitigacion-sesgo-fairness-algoritmico/
tags:
  - machine learning
  - fairness
  - sesgo algorítmico
  - ética en IA
  - regulación IA
  - in-processing
excerpt: 'Una guía sobre técnicas de mitigación del sesgo algorítmico: Adversarial Learning, Meta-Fair Classifier y Exponentiated Gradient Reduction, con énfasis en su intuición y fundamentos técnicos.'
---

**El sesgo en los sistemas de inteligencia artificial no es un accidente: es un síntoma**

Cuando un modelo de machine learning aprende a partir de datos históricos, absorbe también las desigualdades que esos datos reflejan. Un sistema de selección de candidatos entrenado con décadas de contrataciones históricamente sesgadas aprenderá, en ausencia de intervención, a reproducir esos sesgos con una apariencia de objetividad. El reto no es solo detectar ese sesgo —lo cual es el dominio de la explicabilidad, abordado en una entrada anterior— sino actuar sobre él durante el proceso de aprendizaje.

A este conjunto de estrategias se le conoce como **mitigación del sesgo algorítmico** o, en la literatura técnica, *algorithmic fairness*. Su objetivo es garantizar que las predicciones de un modelo sean equitativas entre grupos definidos por atributos sensibles como el género, la raza, la edad o la procedencia geográfica. En el contexto de la regulación de la IA, marcos como el **EU AI Act** exigen explícitamente que los sistemas de alto riesgo —aquellos que inciden en empleo, crédito, salud o educación— demuestren medidas activas de mitigación del sesgo antes de su despliegue.

Las técnicas de mitigación se clasifican según el momento del pipeline de machine learning en que intervienen. Esta distinción no es solo técnica: tiene implicaciones directas sobre qué tipo de sesgo se puede corregir y a qué costo.

## Los tres paradigmas de mitigación

### Pre-processing: intervenir en los datos antes de entrenar

Las técnicas de pre-procesamiento actúan sobre el conjunto de datos de entrenamiento antes de que el modelo lo vea. El razonamiento es directo: si el sesgo proviene de los datos, corrígelo en los datos. Las estrategias incluyen el rebalanceo de clases (*resampling*), la transformación de variables para eliminar correlaciones con atributos sensibles, o la reasignación de etiquetas en instancias donde se sospecha discriminación histórica.

La ventaja de este enfoque es su independencia del modelo: cualquier clasificador entrenado sobre los datos ya corregidos se beneficia de la intervención. La limitación es que la corrección es ciega al comportamiento del modelo. Un preprocesamiento que elimina correlaciones lineales puede dejar intactas correlaciones no lineales que el modelo aprenderá igualmente.

### In-processing: incorporar la equidad durante el entrenamiento

Las técnicas de in-procesamiento modifican el proceso de aprendizaje en sí mismo, incorporando la equidad como un objetivo adicional que el modelo debe satisfacer mientras minimiza su error de predicción. Este paradigma reconoce que la equidad y la precisión son objetivos que pueden estar en tensión, y propone mecanismos formales para gestionarla.

Es el paradigma más técnicamente rico y el que ofrece mayores garantías de alineación entre el comportamiento del modelo y los criterios de equidad definidos. Las tres técnicas que aborda esta entrada —**Adversarial Learning**, **Meta-Fair Classifier** y **Exponentiated Gradient Reduction**— pertenecen a esta categoría.

### Post-processing: corregir las predicciones ya producidas

Las técnicas de post-procesamiento ajustan las salidas del modelo después de que este ha sido entrenado, sin modificar sus parámetros internos. Un ejemplo típico es la calibración de umbrales de decisión diferenciada por grupo: si el modelo aprueba créditos con una probabilidad superior al 0.5, se puede ajustar ese umbral por separado para cada grupo demográfico de manera que las tasas de aprobación o de error sean comparables.

Esta estrategia es especialmente útil cuando el modelo ya está en producción y no puede reentrenarse. Sin embargo, actuar solo sobre las salidas sin tocar el proceso de aprendizaje implica que el modelo sigue "pensando" de la misma forma sesgada: la corrección es superficial y puede ser frágil ante distribuciones de datos no vistas durante la calibración.

---

## 1. Adversarial Learning para equidad

### 1.1 La intuición: el duelo entre el predictor y el árbitro

Adversarial Learning para equidad parte de una idea tomada de la teoría de juegos: dos agentes en competencia, entrenados simultáneamente, se fuerzan mutuamente a mejorar. En el contexto de equidad, estos dos agentes son el **predictor** y el **discriminador**.

El predictor tiene la tarea habitual: aprender a hacer buenas predicciones sobre una variable de interés —si una persona pagará un préstamo, si un candidato será seleccionado, si un paciente requerirá hospitalización. El discriminador tiene una tarea diferente: a partir de las representaciones internas del predictor, intentar identificar a qué grupo sensible pertenece cada instancia —si la persona es hombre o mujer, si tiene determinada etnia, si supera cierta edad.

El entrenamiento adversarial crea una tensión deliberada: el predictor quiere hacer buenas predicciones *y al mismo tiempo* quiere que el discriminador falle. Cuanto más difícil le resulte al discriminador adivinar el grupo a partir de las representaciones del predictor, más cerca está ese predictor de haber aprendido a prescindir del atributo sensible.

### 1.2 El mecanismo: optimización minimax

Formalmente, este entrenamiento se enmarca como un problema de **optimización minimax**, el mismo marco matemático que subyace a las Redes Generativas Adversariales (GANs). El objetivo combinado tiene la forma:

```
min_predictor  max_discriminador  [ L_pred − λ · L_disc ]
```

donde `L_pred` es la pérdida del predictor (que se quiere minimizar para predecir bien) y `L_disc` es la pérdida del discriminador (que el predictor quiere maximizar para que el árbitro falle). El parámetro `λ` controla el peso relativo del objetivo de equidad frente al de precisión.

El estado al que tiende este sistema cuando el entrenamiento converge se denomina **equilibrio de Nash**: ninguno de los dos agentes puede mejorar su propio resultado cambiando su estrategia unilateralmente. En ese punto, el predictor ha aprendido a producir predicciones precisas sin revelar información sobre el grupo sensible.

### 1.3 Invarianza de representación

El concepto técnico clave que emerge de este entrenamiento es la **invarianza de representación**: las capas internas del predictor dejan de retener información estadísticamente asociada al atributo sensible. Formalmente, se busca que:

```
P(grupo | representación_interna) ≈ P(grupo)
```

Es decir, conocer la representación que el modelo construye para una instancia no debe mejorar la capacidad de predecir a qué grupo pertenece esa instancia. La representación se vuelve "ciega" al atributo sensible sin que ello exija necesariamente que el modelo lo ignore de forma explícita.

### 1.4 Fortalezas y limitaciones

La principal fortaleza de este enfoque es su **generalidad**: no requiere acceso directo al atributo sensible en tiempo de inferencia, y puede aplicarse a modalidades de datos muy diversas, incluyendo texto e imágenes. Su limitación más importante es la **inestabilidad del entrenamiento**: al igual que ocurre con las GANs, el proceso adversarial puede no converger, o puede hacerlo hacia soluciones subóptimas si el predictor y el discriminador no están equilibrados en capacidad. No existe una garantía matemática de convergencia bajo condiciones generales, lo que exige una supervisión cuidadosa del entrenamiento.

---

## 2. Meta-Fair Classifier

### 2.1 La intuición: la equidad como parámetro ajustable

Los clasificadores tradicionales producen un modelo con un comportamiento fijo respecto a la equidad: si se entrena con una determinada restricción de paridad, esa restricción queda "horneada" en los pesos del modelo. Cambiar el criterio de equidad exige reentrenar desde cero.

El Meta-Fair Classifier propone una arquitectura diferente. En lugar de aprender a predecir, aprende a predecir **dado un nivel de equidad deseado**. Formalmente, el modelo recibe como entrada no solo las variables predictoras `X`, sino también un parámetro `τ ∈ [0,1]` que especifica el grado de equidad que se requiere:

```
f(X, τ) → ŷ
```

La analogía es la de un sastre que no hace un único traje estándar, sino que aprende el proceso general de confección y puede producir cualquier talla bajo demanda. El cliente —en este caso, el sistema o el regulador— especifica sus requerimientos en el momento de la inferencia, y el modelo los satisface sin necesidad de intervención adicional.

### 2.2 El mecanismo: meta-aprendizaje sobre restricciones de equidad

El entrenamiento del Meta-Fair Classifier sigue el paradigma del **meta-aprendizaje** (*learning to learn*): en lugar de optimizar para una tarea específica, el modelo se entrena sobre una distribución de tareas —en este caso, distintos valores de `τ`— y aprende a generalizar a cualquier nivel de equidad dentro de ese rango.

Durante el entrenamiento, el modelo es expuesto a múltiples escenarios en los que el parámetro `τ` varía, y aprende a adaptar su comportamiento para satisfacer el criterio de equidad correspondiente en cada caso. Esta exposición diversa durante el entrenamiento es lo que permite al modelo responder correctamente a valores de `τ` no vistos explícitamente.

### 2.3 Definiciones formales de equidad: SP y EO

El Meta-Fair Classifier puede operar bajo distintas definiciones de equidad. Las dos más relevantes en la literatura son la **paridad estadística** (*Statistical Parity*, SP) y la **igualdad de oportunidades** (*Equal Opportunity*, EO), y su distinción tiene consecuencias importantes tanto técnicas como éticas.

La **paridad estadística** exige que la probabilidad de recibir una predicción positiva sea igual entre grupos:

```
P(ŷ = 1 | grupo = A) = P(ŷ = 1 | grupo = B)
```

En términos concretos, si un modelo aprueba solicitudes de crédito, SP exige que la tasa de aprobación sea la misma para todos los grupos demográficos, independientemente de cualquier otra característica. Este criterio es intuitivo y fácil de auditar, pero puede ser problemático: si los grupos difieren en sus tasas reales de repago, imponer SP implica aprobar créditos a personas con mayor riesgo real en un grupo, o rechazar a personas con menor riesgo en otro.

La **igualdad de oportunidades** aborda esta tensión restringiendo la equidad a las instancias que *merecen* el resultado positivo, es decir, aquellas cuya etiqueta real es positiva:

```
P(ŷ = 1 | y = 1, grupo = A) = P(ŷ = 1 | y = 1, grupo = B)
```

EO exige que entre quienes sí reúnen las condiciones —quienes efectivamente pagarían el crédito, quienes sí tienen la enfermedad, quienes sí son candidatos idóneos—, la tasa de acierto del modelo sea igual para todos los grupos. Este criterio es más preciso que SP porque no penaliza al modelo por diferencias reales entre grupos, sino por diferencias en cómo trata a instancias equivalentes.

La elección entre SP y EO no es solo técnica: es una decisión normativa que debe tomarse en función del dominio de aplicación y del marco regulatorio vigente. El Meta-Fair Classifier tiene la ventaja de poder satisfacer ambas bajo el mismo modelo simplemente variando `τ` y la definición objetivo.

### 2.4 Fortalezas y limitaciones

La fortaleza distintiva de este método es su **flexibilidad operativa**: un único modelo puede servir a distintos requerimientos de equidad sin reentrenamiento, lo que lo hace especialmente valioso en entornos regulatorios donde los criterios pueden cambiar o donde distintos mercados exigen distintos estándares. Su limitación principal es que la definición de equidad a aplicar —SP, EO u otra— debe elegirse antes del entrenamiento, ya que el modelo aprende a generalizar dentro de esa familia de criterios. No puede, en principio, satisfacer simultáneamente definiciones de equidad que sean matemáticamente incompatibles entre sí, lo cual es un resultado conocido en la literatura de *fairness*.

---

## 3. Exponentiated Gradient Reduction

### 3.1 La intuición: convertir un problema difícil en muchos problemas simples

Entrenar un modelo justo es, en esencia, un problema de optimización con restricciones: se quiere maximizar la precisión del modelo sujeto a que las tasas de error no difieran de manera significativa entre grupos. Este tipo de problemas —con objetivos múltiples potencialmente en tensión— es notoriamente difícil de resolver de forma directa.

Exponentiated Gradient Reduction propone una estrategia elegante: **reducir** el problema de equidad a una secuencia de problemas de clasificación estándar, cada uno resoluble con cualquier clasificador convencional. En lugar de diseñar un clasificador especial con equidad incorporada, el método toma cualquier clasificador existente y lo *guía* iterativamente hacia una solución justa.

La analogía es la de equilibrar una dieta. El nutricionista no reformula todos los alimentos; simplemente ajusta semana a semana la proporción de cada grupo alimentario según qué nutrientes están siendo deficitarios. Si falta proteína, se aumenta su proporción en la dieta. Si hay exceso de grasa, se reduce. El proceso converge hacia un balance nutricional sin necesidad de rediseñar cada alimento por separado.

### 3.2 El mecanismo: dualidad de Lagrange y multiplicadores adaptativos

El fundamento matemático del método proviene de la **dualidad de Lagrange**, una técnica de optimización que transforma un problema con restricciones en uno sin restricciones añadiendo una penalización al objetivo.

El problema original —maximizar precisión sujeto a restricciones de equidad— se reescribe como:

```
L = pérdida(ŷ, y) + Σ_g  λ_g · violación_equidad_g
```

donde `λ_g` es el **multiplicador de Lagrange** asociado al grupo `g`, y `violación_equidad_g` mide cuánto está incumpliendo el modelo el criterio de equidad para ese grupo. Cuanto mayor es la violación, mayor es el costo añadido al objetivo, y mayor la presión para corregirla.

Lo que hace al método *exponentiated gradient* es la regla con la que se actualizan los multiplicadores en cada iteración:

```
λ_g ← λ_g · exp(η · violación_g)
```

El crecimiento **exponencial** del multiplicador cuando hay violación garantiza que grupos que están siendo tratados de forma persistentemente injusta reciban una atención proporcionalmente mayor en las iteraciones siguientes. Si un grupo ya es tratado equitativamente, su multiplicador decrece y deja de ejercer presión sobre el entrenamiento.

### 3.3 Garantías de convergencia

La propiedad que distingue a Exponentiated Gradient Reduction de los enfoques adversariales es la existencia de **garantías teóricas de convergencia**. El método tiene demostración formal de que, bajo condiciones generales, el proceso siempre converge a una solución que satisface el criterio de equidad hasta un margen `ε` de tolerancia, en un número de iteraciones acotado por:

```
O(1/ε²)
```

Esto significa que, a diferencia del entrenamiento adversarial, el proceso no puede quedar atrapado indefinidamente en un ciclo inestable. Si se tolera un margen de equidad de 0.05, el número máximo de iteraciones necesarias es predecible desde el inicio del entrenamiento.

Esta garantía es relevante no solo desde el punto de vista técnico, sino también desde el regulatorio: sistemas de IA de alto riesgo desplegados en contextos como el crédito o la salud requieren que los desarrolladores puedan documentar y demostrar las propiedades de equidad del modelo. La existencia de garantías teóricas facilita ese proceso de auditoría.

### 3.4 Fortalezas y limitaciones

La fortaleza central de este método es precisamente la **solidez matemática**: las garantías de convergencia y la interpretabilidad de los multiplicadores `λ` como pesos de grupo lo convierten en el enfoque más auditable de los tres. Un auditor puede inspeccionar el valor de `λ_g` al final del entrenamiento y entender, de forma directa, qué grupos recibieron mayor atención correctiva.

Su limitación principal es que las restricciones de equidad deben poder expresarse en forma lineal sobre las tasas de error del modelo. Definiciones de equidad más sofisticadas, o que involucran intersecciones de atributos sensibles (*fairness interseccional*), pueden no reducirse a la forma requerida sin extensiones adicionales del método. Asimismo, el número de iteraciones puede ser elevado cuando se exigen márgenes de equidad muy estrictos (valores pequeños de `ε`), lo que puede incrementar el costo computacional del entrenamiento.

---

## Comparación y criterios de selección

Las tres técnicas abordan el mismo problema desde ángulos distintos, y su elección debe guiarse por los requerimientos del contexto de aplicación.

**Adversarial Learning** es el enfoque más flexible y expresivo, adecuado cuando se trabaja con datos no estructurados (texto, imágenes) o cuando la relación entre las variables predictoras y el atributo sensible es compleja y difícil de formalizar como restricción explícita. Su coste es la inestabilidad del entrenamiento y la ausencia de garantías formales.

**Meta-Fair Classifier** es la opción más práctica cuando el modelo se despliega en entornos donde el criterio de equidad puede cambiar sin posibilidad de reentrenamiento, o donde distintos segmentos de usuarios o jurisdicciones demandan distintos estándares. El requerimiento es que la familia de criterios de equidad se elija antes del entrenamiento.

**Exponentiated Gradient Reduction** es la elección más defensible desde una perspectiva regulatoria y de auditoría, gracias a sus garantías de convergencia y a la interpretabilidad de sus parámetros. Es particularmente adecuado cuando la equidad debe demostrarse formalmente ante terceros y cuando las restricciones de equidad pueden expresarse sobre tasas de error por grupo.

En la práctica, estas técnicas no son excluyentes: pueden combinarse entre sí, o con estrategias de pre- y post-procesamiento, para construir pipelines de mitigación más robustos. La tendencia en sistemas de IA de alto riesgo es precisamente hacia este tipo de defensas en capas, donde ninguna técnica individual carga con toda la responsabilidad de garantizar la equidad del sistema.

El siguiente widget permite explorar de forma interactiva las características y conceptos técnicos de cada método:

<div class="fw-widget-wrap">

<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">

<style>
  .fw-widget-wrap {
    margin: 2.5rem 0;
    font-family: 'DM Sans', sans-serif;
    --fw-bg:             #faf9f7;
    --fw-surface:        #ffffff;
    --fw-border:         #e8e5e0;
    --fw-border-hover:   #c8c4bc;
    --fw-text-1:         #1a1917;
    --fw-text-2:         #5c5a56;
    --fw-text-3:         #9c9994;
    --fw-radius-sm:      6px;
    --fw-radius-md:      10px;
    --fw-radius-lg:      14px;
  }
  .fw-widget {
    background: var(--fw-surface);
    border: 1px solid var(--fw-border);
    border-radius: var(--fw-radius-lg);
    padding: 2rem;
    max-width: 980px;
    width: 90%;
    margin: 0 auto;
    box-shadow: 0 1px 3px rgba(0,0,0,0.04), 0 4px 16px rgba(0,0,0,0.04);
    box-sizing: border-box;
  }
  .fw-widget *, .fw-widget *::before, .fw-widget *::after {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }
  .fw-header { margin-bottom: 1.75rem; }
  .fw-label {
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--fw-text-3);
    margin-bottom: 6px;
    display: block;
  }
  .fw-title {
    font-size: 18px;
    font-weight: 500;
    color: var(--fw-text-1);
    line-height: 1.3;
    display: block;
  }
  .fw-sub {
    font-size: 13px;
    color: var(--fw-text-2);
    margin-top: 4px;
    line-height: 1.5;
    display: block;
  }
  .fw-card-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 10px;
    margin-bottom: 1.5rem;
  }
  @media (max-width: 560px) {
    .fw-card-grid { grid-template-columns: 1fr; }
    .fw-widget { width: 100%; padding: 1.25rem; }
  }
  .fw-card {
    border: 1px solid var(--fw-border);
    border-radius: var(--fw-radius-md);
    padding: 14px;
    cursor: pointer;
    transition: border-color 0.18s, box-shadow 0.18s;
    background: var(--fw-surface);
    position: relative;
    user-select: none;
  }
  .fw-card:hover {
    border-color: var(--fw-border-hover);
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
  }
  .fw-card.fw-active {
    border-color: transparent;
    box-shadow: 0 0 0 2px var(--fw-accent, #534AB7);
  }
  .fw-badge {
    display: inline-block;
    font-size: 10px;
    font-weight: 500;
    padding: 3px 8px;
    border-radius: 20px;
    margin-bottom: 10px;
    letter-spacing: 0.03em;
  }
  .fw-card-name {
    font-size: 13px;
    font-weight: 500;
    color: var(--fw-text-1);
    margin-bottom: 5px;
    line-height: 1.3;
    display: block;
  }
  .fw-card-meta {
    font-size: 11px;
    color: var(--fw-text-2);
    line-height: 1.5;
    display: block;
  }
  .fw-detail {
    border: 1px solid var(--fw-border);
    border-radius: var(--fw-radius-md);
    overflow: hidden;
    animation: fwFadeIn 0.2s ease;
  }
  @keyframes fwFadeIn {
    from { opacity: 0; transform: translateY(4px); }
    to   { opacity: 1; transform: translateY(0); }
  }
  .fw-detail-header {
    padding: 16px 20px;
    border-bottom: 1px solid var(--fw-border);
    display: flex;
    align-items: center;
    gap: 12px;
  }
  .fw-detail-icon {
    width: 36px;
    height: 36px;
    border-radius: var(--fw-radius-sm);
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 18px;
    flex-shrink: 0;
  }
  .fw-detail-title {
    font-size: 15px;
    font-weight: 500;
    color: var(--fw-text-1);
    display: block;
  }
  .fw-detail-subtitle {
    font-size: 12px;
    color: var(--fw-text-2);
    margin-top: 1px;
    display: block;
  }
  .fw-detail-body {
    display: grid;
    grid-template-columns: 1fr 2fr;
    gap: 0;
  }
  @media (max-width: 560px) {
    .fw-detail-body { grid-template-columns: 1fr; }
  }
  .fw-detail-col { padding: 16px 20px; }
  .fw-detail-col + .fw-detail-col { border-left: 1px solid var(--fw-border); }
  @media (max-width: 560px) {
    .fw-detail-col + .fw-detail-col { border-left: none; border-top: 1px solid var(--fw-border); }
  }
  .fw-section-label {
    font-size: 10px;
    font-weight: 500;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: var(--fw-text-3);
    margin-bottom: 10px;
    display: block;
  }
  .fw-field { margin-bottom: 12px; }
  .fw-field:last-child { margin-bottom: 0; }
  .fw-field-label {
    font-size: 11px;
    color: var(--fw-text-3);
    margin-bottom: 2px;
    font-weight: 500;
    display: block;
  }
  .fw-field-value {
    font-size: 12px;
    color: var(--fw-text-1);
    line-height: 1.55;
    display: block;
  }
  .fw-field-value.fw-positive { color: #2d7a4e; }
  .fw-field-value.fw-negative { color: #9b3a3a; }
  .fw-formula-block {
    background: var(--fw-bg);
    border: 1px solid var(--fw-border);
    border-radius: var(--fw-radius-sm);
    padding: 10px 12px;
    margin-top: 10px;
  }
  .fw-formula-label {
    font-size: 10px;
    color: var(--fw-text-3);
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 5px;
    display: block;
  }
  .fw-formula-text {
    font-family: 'DM Mono', monospace;
    font-size: 11px;
    color: var(--fw-text-2);
    line-height: 1.6;
    display: block;
    white-space: nowrap;
  }
  .fw-dim-section {
    padding: 14px 20px;
    border-top: 1px solid var(--fw-border);
    background: var(--fw-bg);
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 16px;
  }
  @media (max-width: 560px) {
    .fw-dim-section { grid-template-columns: 1fr; gap: 10px; }
  }
  .fw-dim-label {
    font-size: 10px;
    color: var(--fw-text-3);
    font-weight: 500;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 5px;
    display: flex;
    justify-content: space-between;
  }
  .fw-dim-value {
    font-size: 11px;
    font-family: 'DM Mono', monospace;
    color: var(--fw-text-2);
  }
  .fw-dim-track {
    height: 4px;
    background: var(--fw-border);
    border-radius: 2px;
    overflow: hidden;
  }
  .fw-dim-fill {
    height: 100%;
    border-radius: 2px;
    transition: width 0.4s cubic-bezier(0.4, 0, 0.2, 1);
  }
  .fw-tag-row {
    display: flex;
    flex-wrap: wrap;
    gap: 5px;
    padding: 12px 20px;
    border-top: 1px solid var(--fw-border);
  }
  .fw-tag {
    font-size: 10px;
    font-weight: 500;
    padding: 3px 9px;
    border-radius: 20px;
    letter-spacing: 0.03em;
  }
</style>

<div class="fw-widget">
  <div class="fw-header">
    <span class="fw-label">Equidad en IA · In-processing</span>
    <span class="fw-title">Técnicas de Fairness Algorítmico</span>
    <span class="fw-sub">Selecciona una técnica para ver sus características y conceptos técnicos.</span>
  </div>
  <div class="fw-card-grid" id="fw-card-grid"></div>
  <div id="fw-detail"></div>
</div>

<script>
(function() {
  var techniques = [
    {
      id: "adv",
      name: "Adversarial Learning",
      metaphor: "El falsificador vs. el detective — compiten hasta que nadie puede ganar.",
      icon: "⚔️",
      accentColor: "#534AB7",
      badgeBg: "#EEEDFE", badgeText: "#3C3489", badgeLabel: "Adversarial",
      iconBg: "#EEEDFE", tagBg: "#EEEDFE", tagColor: "#3C3489",
      tags: ["GANs de equidad", "In-processing", "Juego de suma cero"],
      dims: { complejidad: 75, flexibilidad: 85, interpretabilidad: 35 },
      stage: "Durante el entrenamiento",
      paradigm: "Minimax + teoría de juegos",
      strength: "Muy flexible; puede mitigar sesgo sin acceder directamente al atributo sensible.",
      weakness: "Entrenamiento inestable; puede no converger si los dos modelos se desequilibran.",
      when: "Cuando los datos tienen variables sensibles bien definidas y se quiere invarianza de representación.",
      concepts: [
        { name: "Optimización minimax",
          plain: "El predictor minimiza su error; el discriminador maximiza su acierto. Ambos entrenan al mismo tiempo, empujándose mutuamente.",
          formula: "min_predictor max_discriminador [ L_pred − λ · L_disc ]" },
        { name: "Equilibrio de Nash",
          plain: "El punto donde ningún jugador puede mejorar cambiando su estrategia sin que el otro lo compense. Es el destino del entrenamiento.",
          formula: "ningún jugador mejora unilateralmente su estrategia" },
        { name: "Invarianza de representación",
          plain: "Las capas internas del modelo dejan de retener información sobre el grupo sensible. El modelo predice sin apoyarse en ese atributo.",
          formula: "P(grupo | representación) ≈ P(grupo)  →  independencia" }
      ]
    },
    {
      id: "meta",
      name: "Meta-Fair Classifier",
      metaphor: "El sastre a medida — ajusta el nivel de equidad como si fuera un parámetro.",
      icon: "🎛️",
      accentColor: "#0F6E56",
      badgeBg: "#E1F5EE", badgeText: "#085041", badgeLabel: "Clasificador",
      iconBg: "#E1F5EE", tagBg: "#E1F5EE", tagColor: "#085041",
      tags: ["Fairness como parámetro", "In-processing", "Sin reentrenar"],
      dims: { complejidad: 60, flexibilidad: 90, interpretabilidad: 55 },
      stage: "Durante el entrenamiento",
      paradigm: "Meta-aprendizaje + restricciones de equidad",
      strength: "Un solo modelo cumple múltiples definiciones de equidad sin reentrenar desde cero.",
      weakness: "Requiere decidir de antemano qué definición de equidad (SP o EO) se quiere cumplir.",
      when: "Cuando necesitas ajustar dinámicamente el nivel de equidad en producción.",
      concepts: [
        { name: "Meta-aprendizaje",
          plain: "En lugar de aprender a predecir, aprende a ajustarse a distintas reglas de equidad. Entrenado sobre muchos escenarios, generaliza a cualquier nivel.",
          formula: "f(X, τ) → ŷ   donde τ ∈ [0,1] es el nivel de equidad" },
        { name: "Paridad estadística (SP)",
          plain: "La probabilidad de una predicción positiva debe ser igual entre grupos. Si aprueba créditos al 60 % para todos, cumple SP.",
          formula: "P(ŷ=1 | A) = P(ŷ=1 | B)" },
        { name: "Igualdad de oportunidades (EO)",
          plain: "Entre quienes sí merecen el resultado positivo, la tasa de acierto debe ser igual. No basta aprobar al mismo porcentaje total.",
          formula: "P(ŷ=1 | y=1, A) = P(ŷ=1 | y=1, B)" }
      ]
    },
    {
      id: "grad",
      name: "Exponentiated Gradient",
      metaphor: "La dieta equilibrada — reasigna atención a quienes están peor atendidos.",
      icon: "📈",
      accentColor: "#854F0B",
      badgeBg: "#FAEEDA", badgeText: "#633806", badgeLabel: "Optimización",
      iconBg: "#FAEEDA", tagBg: "#FAEEDA", tagColor: "#633806",
      tags: ["Optimización convexa", "In-processing", "Garantías teóricas"],
      dims: { complejidad: 65, flexibilidad: 70, interpretabilidad: 60 },
      stage: "Durante el entrenamiento",
      paradigm: "Dualidad de Lagrange + gradiente exponencial",
      strength: "Garantías matemáticas de convergencia; más estable que el enfoque adversarial.",
      weakness: "Asume que la equidad puede expresarse como restricciones lineales sobre las tasas de error.",
      when: "Cuando necesitas garantías teóricas de equidad y estabilidad en el entrenamiento.",
      concepts: [
        { name: "Dualidad de Lagrange",
          plain: "Convierte el problema de 'maximiza precisión sujeto a equidad' en 'maximiza precisión menos una penalización por incumplirla'. Más fácil de resolver.",
          formula: "L = pérdida(ŷ,y) + Σ λ_g · violación_equidad_g" },
        { name: "Multiplicadores λ (pesos de grupo)",
          plain: "Un número por cada grupo. Si ese grupo es tratado injustamente, su λ crece exponencialmente y el modelo lo prioriza. Si ya es justo, λ baja.",
          formula: "λ_g ← λ_g · exp(η · violación_g)" },
        { name: "Garantía de convergencia",
          plain: "A diferencia del adversarial, hay prueba matemática de que el proceso siempre llega a una solución válida. No puede quedar atascado indefinidamente.",
          formula: "ε-equidad garantizada en O(1/ε²) iteraciones" }
      ]
    }
  ];

  var active = 0;

  function renderCards() {
    var grid = document.getElementById('fw-card-grid');
    grid.innerHTML = techniques.map(function(t, i) {
      return '<div class="fw-card' + (i === active ? ' fw-active' : '') + '"'
        + ' style="--fw-accent:' + t.accentColor + '"'
        + ' onclick="fwSelect(' + i + ')">'
        + '<span class="fw-badge" style="background:' + t.badgeBg + ';color:' + t.badgeText + '">' + t.badgeLabel + '</span>'
        + '<span class="fw-card-name">' + t.name + '</span>'
        + '<span class="fw-card-meta">' + t.metaphor + '</span>'
        + '</div>';
    }).join('');
  }

  function renderDetail() {
    var t = techniques[active];
    var conceptsHtml = t.concepts.map(function(c) {
      return '<div class="fw-field">'
        + '<span class="fw-field-label">' + c.name + '</span>'
        + '<span class="fw-field-value">' + c.plain + '</span>'
        + '<div class="fw-formula-block">'
        + '<span class="fw-formula-label">Notación técnica</span>'
        + '<span class="fw-formula-text">' + c.formula + '</span>'
        + '</div></div>';
    }).join('');

    var dims = ['complejidad', 'flexibilidad', 'interpretabilidad'];
    var dimsHtml = dims.map(function(dim) {
      return '<div class="fw-dim-item">'
        + '<div class="fw-dim-label"><span>' + dim.charAt(0).toUpperCase() + dim.slice(1) + '</span>'
        + '<span class="fw-dim-value">' + t.dims[dim] + '%</span></div>'
        + '<div class="fw-dim-track"><div class="fw-dim-fill" style="width:' + t.dims[dim] + '%;background:' + t.accentColor + '"></div></div>'
        + '</div>';
    }).join('');

    var tagsHtml = t.tags.map(function(tag) {
      return '<span class="fw-tag" style="background:' + t.tagBg + ';color:' + t.tagColor + '">' + tag + '</span>';
    }).join('');

    document.getElementById('fw-detail').innerHTML =
      '<div class="fw-detail">'
      + '<div class="fw-detail-header">'
      + '<div class="fw-detail-icon" style="background:' + t.iconBg + '">' + t.icon + '</div>'
      + '<div><span class="fw-detail-title">' + t.name + '</span>'
      + '<span class="fw-detail-subtitle">' + t.paradigm + '</span></div>'
      + '</div>'
      + '<div class="fw-detail-body">'
      + '<div class="fw-detail-col">'
      + '<span class="fw-section-label">Contexto de uso</span>'
      + '<div class="fw-field"><span class="fw-field-label">Etapa</span><span class="fw-field-value">' + t.stage + '</span></div>'
      + '<div class="fw-field"><span class="fw-field-label">Fortaleza</span><span class="fw-field-value fw-positive">' + t.strength + '</span></div>'
      + '<div class="fw-field"><span class="fw-field-label">Limitación</span><span class="fw-field-value fw-negative">' + t.weakness + '</span></div>'
      + '<div class="fw-field"><span class="fw-field-label">Úsalo cuando</span><span class="fw-field-value">' + t.when + '</span></div>'
      + '</div>'
      + '<div class="fw-detail-col">'
      + '<span class="fw-section-label">Conceptos técnicos clave</span>'
      + conceptsHtml
      + '</div>'
      + '</div>'
      + '<div class="fw-dim-section">' + dimsHtml + '</div>'
      + '<div class="fw-tag-row">' + tagsHtml + '</div>'
      + '</div>';
  }

  window.fwSelect = function(i) {
    active = i;
    renderCards();
    renderDetail();
  };

  renderCards();
  renderDetail();
})();
</script>

</div>
