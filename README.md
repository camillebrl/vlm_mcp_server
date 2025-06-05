# VLM MCP Server

Un serveur MCP (Model Context Protocol) pour l'analyse d'images utilisant le modèle Vision-Language InternVL3. Ce serveur permet d'évaluer la pertinence d'images par rapport à des requêtes et de générer des réponses textuelles basées sur le contenu visuel.

## Fonctionnalités

- **Analyse d'images avec requête** : Analyse multiple d'images avec vérification automatique de pertinence
- **Vérification de pertinence** : Évalue si une image est pertinente pour une requête donnée
- **Analyse d'image unique** : Analyse détaillée d'une seule image
- **Gestion intelligente de la mémoire** : Chargement/déchargement automatique du modèle
- **Support GPU optimisé** : Utilisation efficace de la mémoire GPU avec nettoyage automatique

## Architecture

Le serveur utilise une architecture modulaire avec :

- **VLMServer** : Serveur MCP principal gérant les outils et les requêtes
- **VLMSModelManager** : Gestionnaire singleton pour le chargement/déchargement du modèle
- **VLMSModel** : Interface avec le modèle InternVL3 pour l'analyse d'images
- **Utilitaires** : Scripts de nettoyage GPU et gestion des processus

## Configuration client MCP

```json
{
  "mcpServers": {
    "vlm-server": {
      "command": "/path/to/vlm-server/start_vlm_server.sh"
      }
    }
  }
}
```

## Debugging
Pour afficher les logs du server:
```bash
tail -n 1000 /tmp/vlm_mcp_startup.log
```

## Installation

### Prérequis

- Python 3.10+
- CUDA (pour l'accélération GPU)
- Poetry pour la gestion des dépendances

### Installation des dépendances

```bash
# Cloner le repository
git clone <repository-url>
cd vlm-server

# Installer avec Poetry
make install
# ou directement avec Poetry
poetry install
```

### Démarrage du serveur

```bash
# Via le script de démarrage
./start_vlm_server.sh

# Ou directement via Poetry
poetry run python -m vlm_server.cli --log-level INFO
```

### Outils disponibles

#### 1. `analyze_images_with_query`

Analyse multiple d'images avec vérification de pertinence automatique.

**Paramètres :**
- `image_paths` (array) : Liste des chemins vers les images
- `query` (string) : Question ou requête sur les images
- `check_relevance` (boolean, optionnel) : Vérifier la pertinence (défaut: true)
- `metadata` (array, optionnel) : Métadonnées pour chaque image

**Exemple :**
```json
{
  "image_paths": ["/path/to/image1.jpg", "/path/to/image2.png"],
  "query": "Quels sont les éléments techniques visibles dans ces diagrammes ?",
  "check_relevance": true
}
```

#### 2. `check_image_relevance`

Vérifie si une image est pertinente pour une requête.

**Paramètres :**
- `image_path` (string) : Chemin vers l'image
- `query` (string) : Requête pour vérifier la pertinence

**Exemple :**
```json
{
  "image_path": "/path/to/document.jpg",
  "query": "Cette image contient-elle des informations sur les ventes ?"
}
```

#### 3. `analyze_single_image`

Analyse détaillée d'une seule image.

**Paramètres :**
- `image_path` (string) : Chemin vers l'image
- `query` (string) : Question sur l'image

#### 4. `get_model_status`

Obtient l'état actuel du modèle (chargé/déchargé, mémoire utilisée).

## Gestion de la mémoire

Le serveur implémente une gestion intelligente de la mémoire GPU :

### Chargement dynamique
- Le modèle n'est chargé qu'au premier usage
- Déchargement automatique quand aucune référence active
- Support de différents modèles InternVL3

### Nettoyage automatique
- Garbage collection Python après chaque opération
- Vidage agressif du cache CUDA
- Réinitialisation des statistiques mémoire

## Formatage du code

```bash
# Formater et vérifier le code
make format

# Nettoyer les fichiers temporaires
make clean
```

## Modèles utilisés

Le serveur utilise par défaut `OpenGVLab/InternVL3-1B` mais supporte d'autres modèles InternVL3 :

- `OpenGVLab/InternVL3-1B` (recommandé, plus léger)
- `OpenGVLab/InternVL3-2B`
- `OpenGVLab/InternVL3-8B`

## Configuration GPU
- Support multi-GPU avec `device_map="auto"`
- Quantification 8-bit pour réduire l'usage mémoire
- Flash Attention quand disponible
- Preprocessing dynamique des images: Taille d'image par défaut : 448x448 pixels, maximum 12 patches par image pour le preprocessing

## Limites
- Maximum 2 images par requête pour éviter l'épuisement mémoire GPU (à modifier si plus de mémoire gpu disponible)