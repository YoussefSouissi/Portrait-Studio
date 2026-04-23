# Portrait Studio

Génération de portraits humains par intelligence artificielle, propulsé par **Stable Diffusion XL** fine-tuné avec **LoRA** sur le dataset CelebA.

---

## Description

Portrait Studio génère des portraits photo-réalistes à partir de descriptions textuelles. Le modèle LoRA a été entraîné sur CelebA dataset.

L'application est disponible en deux modes : **local via Docker** ou **en ligne via Google Colab**.

---

## Architecture du projet

```
portrait-studio/
├── Dockerfile                # Définition du build multi-étapes du conteneur de production
├── README.md                 # Documentation du projet
├── app.py                    # Point d'entrée du conteneur Docker
├── docker-compose.yml        # Configuration des services Docker Compose
├── download_model.py         # Script de téléchargement des poids SDXL depuis Hugging Face
├── requirements.txt          # Dépendances Python du backend
├── server.py                 # Serveur Flask principal — API REST et inférence ML
├── src/                      # Environnement d'entraînement et d'évaluation ML
│   ├── config.py             # Hyperparamètres et chemins pour l'entraînement
│   ├── dataset.py            # Chargement et transformations du dataset CelebA
│   ├── evaluate.py           # Scripts d'évaluation et de métriques du modèle
│   ├── generate_image.py     # Logique d'inférence et génération d'images en batch
│   ├── improved_prompts.py   # Scripts d'ingénierie et d'amélioration des prompts
│   ├── inference_config.py   # Paramètres d'inférence optimisés
│   ├── mlflow_utils.py       # Intégration du tracking d'expériences MLFlow
│   ├── train.py              # Boucle principale de fine-tuning de l'adaptateur LoRA
│   └── visualize.py          # Utilitaires de visualisation des données et générations
├── notebooks/                # Notebooks Jupyter pour l'expérimentation initiale
├── data/                     # Datasets bruts (CelebA)
├── output/                   # Checkpoints d'entraînement et poids LoRA sauvegardés
├── results/                  # Résultats d'inférence et d'évaluation sauvegardés
└── front-end/                # Environnement frontend React SPA
```

---

## Solution 1 — Locale (Docker)

L'application tourne sur votre machine et est accessible via `http://localhost:8000`.

**Prérequis :** Docker Desktop, GPU NVIDIA avec drivers CUDA, NVIDIA Container Toolkit.

**1. Télécharger l'image :**
```bash
docker pull youssefsouissi/portrait-studio:latest
```

**2. Lancer le conteneur :**
```bash
docker run --gpus all -p 8000:8000 -v portrait-models:/app/models -e HF_TOKEN=votre_token youssefsouissi/portrait-studio:latest
```

> `HF_TOKEN` est optionnel mais recommandé pour accélérer le téléchargement des modèles. Obtenez-en un gratuitement sur https://huggingface.co/settings/tokens

Au premier lancement, les modèles (~7 Go) sont téléchargés et mis en cache dans le volume `portrait-models`. Les lancements suivants démarrent directement.

**3. Ouvrir l'interface :** http://localhost:8000

La résolution de sortie est ajustée automatiquement selon la VRAM disponible (512×512 jusqu'à 1024×1024).

---

## Solution 2 — En ligne (Google Colab)

Aucune installation requise. Le modèle tourne sur les GPU de Google Colab et est accessible via un lien public généré par ngrok.

**Prérequis :** un token ngrok gratuit — https://ngrok.com

**1.** Ouvrir https://colab.research.google.com et importer le fichier `portrait_studio_online.ipynb`

**2.** Activer le GPU : **Exécution → Modifier le type d'exécution → T4 GPU**

**3.** Dans la Cellule 4, coller votre token ngrok :
```python
NGROK_TOKEN = 'votre_token_ngrok_ici'
```

**4.** Lancer toutes les cellules : **Exécution → Tout exécuter**

À la fin de l'exécution, l'URL publique s'affiche dans le terminal. Partagez-la ou ouvrez-la directement dans votre navigateur.

---

## Modèle

| | |
|---|---|
| Base | Stable Diffusion XL 1.0 |
| Technique | LoRA |
| Dataset | CelebA — 182 339 images |
