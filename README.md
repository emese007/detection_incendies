# Détection de zones brûlées sur images satellites en vision par ordinateur

## Installation et utilisation

### Prérequis

- [uv](https://github.com/astral-sh/uv)

### 1. Cloner le projet

```bash
git clone git@github.com:emese007/detection_incendies.git
cd detection_incendies
```

### 2. Lancer le script

```bash
# Préparation des données
uv run main.py

# Entraînement du modèle
uv run train.py
```

## Documentation

### Nettoyage des données

Tous les fichiers ne correspondant pas à un format d'image ou au format `.json` sont supprimés de notre dossier de données.

Nous supprimons du fichier COCO JSON :

- Les images n'ayant aucune annotation
- Les images introuvables dans le dossier de données
- Les annotations orphelines
- Les annotations invalides (largeur/hauteur inattendue, hors du cadre)

### Choix du modèle

**Quel(s) type(s) de modèle(s) permet(tent) de répondre à la problématique du projet ?**

- Convolutional Neural Networks (CNN)

Modèle de réseau de neurones profonds spécifiquement conçu pour le traitement des images, en utilisant des couches de convolution pour extraire des caractéristiques hiérarchiques.

- Region-based Convolutional Neural Networks (R-CNN)

Extension des CNN pour la détection d'objets. Utilise des propositions de régions (regions of interest) comme entrée pour un CNN afin de classer et localiser les objets.

- YOLO (You Only Look Once)

Un modèle de détection d'objets en temps réel qui divise l'image en une grille et prédit les annotations et leurs classes simultanément.

- CLIP (Contrastive Language–Image Pretraining)

Un modèle pré-entraîné qui lie des images et du texte en associant des représentations visuelles et linguistiques, permettant par exemple, de savoir si un texte donné décrit une image particulière.

**Quelles sont les spécificités d’un modèle pré-entraîné ?**

Un modèle peut être pré-entraîné sur un dataset très large pour une tâche spécifique puis être réutilisé ou 'fine-tuned' pour une tâche similaire. Cela permet de gagner du temps et ne nécessite donc pas toute la ressource informatique autrement utilisée pour l'entraînement.

**Quelles sont les principales différences architecturales entre un CPU et un GPU ? Comment est équipé votre ordinateur ?**

Le CPU gère les fonctions au niveau du système, la logique globale de l'application et le traitement des données, il se doit d'être réactif et apte gérer des tâches variées. Le GPU va, quant à lui, plutôt être spécialisé dans certains types de tâches comme par exemple le rendu vidéo ou l'apprentissage automatique.

L'une des machines utilisées sur ce projet est équipé d'une puce avec CPU 14 cœurs et GPU 20 cœurs mais son plus grand atout est sans doute sa mémoire unifiée.

En effet, dans une architecture plus classique, le CPU a sa propre mémoire (RAM) et le GPU également (VRAM). Dans cette situation, les données doivent être constamment copiées de l'une à l'autre, ce qui crée une latence.

Dans le cas de la mémoire unifiée, c'est différent. Cette dernière est directement accessible par le CPU et le GPU sans copie.

**Comment la vitesse d'entraînement de YOLO varie-t-elle entre CPU et GPU ?**

L'entrainement de YOLO repose sur des millions d'opérations mathématiques, la différence est donc considérable. L'architecture d'un GPU est conçue pour permettre le traitement de plusieurs tâches en parallèle, là où le CPU le traitement serait beaucoup plus séquentiel.

**Quels modèles privilégier pour notre problématique en prenant en compte la taille du dataset ? Les contraintes de ressources d'entraînement ?**

On peut considérer les différentes versions et tailles de YOLO. Pour faire ce choix, il faut tenir compte du fait que nous avons un petit dataset. Il faut également trouver un équilibre entre vitesse et précision. Il serait intéressant de commencer par un petit modèle avant de monter en complexité si nécessaire.

**Quelles solutions existent pour adapter la taille de notre dataset si c’est nécessaire au modèle ?**

Le problème principal ici est que notre dataset est trop petit. Nous pourrions l'enrichir artificiellement pour rendre notre modèle plus robuste. La solution pour cela serait la 'data augmentation' qui nous permettrait de créer de nouvelles images d'entrainement en appliquant des transformations aléatoires à nos images existantes.
