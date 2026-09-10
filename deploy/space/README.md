---
title: VISAGE - Génération de visages avec contrôle d'attributs
emoji: 🎭
colorFrom: gray
colorTo: indigo
sdk: gradio
app_file: app.py
pinned: false
license: cc-by-4.0
---

# VISAGE : génération conditionnelle de visages

Projet MSC AIC, IA générative. **DDPM conditionnel entraîné from scratch**
(15,9 M paramètres, 48 px) avec Classifier-Free Guidance, échantillonnage DDIM.
Contrôle de l'âge (18-70 ans, continu), du genre et de la tonalité de peau
(7 groupes FairFace).

## ⏱️ Avant de cliquer

Ce Space tourne sur **CPU gratuit (2 vCPU)** : comptez **~8 s par visage** à
30 pas DDIM. Après 48 h sans visite, le Space s'endort : le premier chargement
prend alors 30 à 60 s. Ce n'est pas une panne.

## Ce que le modèle fait, et ne fait pas

| Attribut | État |
|---|---|
| Genre | contrôlé (86,8 % d'accord avec un classifieur juge) |
| Tonalité de peau | partiellement contrôlée (44,7 % sur 7 classes ; le juge lui-même plafonne à 62 %) |
| Âge | contrôlé entre ~26 et ~62 ans ; les extrémités 18 et 70 restent tirées vers le centre |

**Résolution 48 px** : contrainte de budget (entraînement sur machine
personnelle). Le FID de 24,0 se lit à cette résolution et n'est pas comparable
aux références haute résolution de la littérature.

## Portée du filtre colorimétrique

L'onglet Génération écarte et régénère les tirages dont la couleur sort de la
plage des visages réels (27 à 29 % des tirages). Les onglets Interpolation et
Atlas partagent un même bruit initial pour préserver l'identité : y régénérer
une image isolée casserait cette propriété. L'interpolation rejoue donc la
séquence entière si nécessaire ; l'atlas, qui coûte 14 visages par âge, est
affiché tel quel.

## Éthique

Les visages produits sont **entièrement synthétiques** : le modèle génère
depuis du bruit et ne peut ni reconstituer ni éditer une personne réelle.
Usage strictement pédagogique. Les biais démographiques du jeu d'entraînement
(FairFace) se retrouvent dans les sorties : la qualité mesurée varie selon les
groupes (FID de 41 à 57 selon la tonalité de peau), et ces écarts sont
rapportés par groupe plutôt que masqués dans une moyenne.

## Données

Entraîné sur [FairFace](https://github.com/joojs/fairface) (Kärkkäinen &
Joo, 2021), choisi pour son équilibre démographique. Le jeu de données n'est
pas redistribué ici ; seuls les poids appris le sont.
