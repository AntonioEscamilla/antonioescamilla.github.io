---
layout: archive
title: "CV"
permalink: /cv/
author_profile: true
redirect_from:
  - /resume
---

{% include base_path %}

Education
======
* Ph.D in Network & Information Technologies, Universitat Oberta de Catalunya, 2023
* M.Sc. in Sound and Music Technologies, Universitat Pompeu Fabra, 2010
* B.Sc. in Audio Engineering, Universidad San Buenaventura, 2008
* B.Sc. in Electronic Engineering, Universidad Pontificia Bolivariana, 2005

Work experience
======
* 2016 - : Full Professor
  * Universidad Pontificia Bolivariana
  * Duties included: Research and Teaching
  * Faculty: Information and Communication Technologies
  * School: Enginieering

* 2017 - 2019: Associate Professor
  * Institución Universitaria ITM
  * Duties included: Teaching in the Digital Arts master programme
  * Faculty: Arts
  
* 2011 - 2015: Full Professor
  * Universidad de San Buenaventura
  * Duties included: Research and Teaching
  * Programme: Audio Enginieering
  * Faculty: Enginieering

* 2011 - 2015: Associate Professor
  * Institución Universitaria Bellas Artes
  * Duties included: Teaching at undergraduate level
  * Programme: Music
  * Faculty: Arts
  
Skills
======
* Human-Computer Interaction
* Artificial Intelligence (AI) and Human Experience
* Computer Vision
* Machine Learning
* Digital Signal Processing
* Music Information Retrieval

Publications
======
  <ul>{% for post in site.publications %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>
    
Teaching
======
  <ul>{% for post in site.teaching %}
    {% include archive-single-cv.html %}
  {% endfor %}</ul>

Teaching
======
## Postgraduate Courses
<ul>{% for post in site.teaching reversed %}
  {% if post.type == "Postgraduate Course" %}
    {% include archive-single-cv.html %}
  {% endif %}
{% endfor %}</ul>

## Undergraduate Courses
<ul>{% for post in site.teaching reversed %}
  {% if post.type == "Undergraduate Course" %}
    {% include archive-single-cv.html %}
  {% endif %}
{% endfor %}</ul>
  
Résumé
======
PhD in Network and Information Technologies from the Universitat Oberta de Catalunya, with a focus on Human-Computer Interaction, Design, and Multimedia. During my doctoral studies (2019-2023), I was a pre-doctoral researcher in the Department of Computer Science, Multimedia, and Telecommunications at the same university. I am a full-time professor in the School of Engineering at Universidad Pontificia Bolivariana and a faculty member in the Information and Communication Technologies Department. I hold a Master's degree in Sound and Music Technologies from Universitat Pompeu Fabra, as well as undergraduate degrees in Audio Engineering and Electronic Engineering.

My academic activities focus on information sciences, the human experience in using artificial intelligence, and the intersection between design and engineering in creating interactive experiences. My research interests primarily include human-computer interaction, creative coding, digital signal processing, music information retrieval, machine learning, artificial intelligence, and computer vision.
