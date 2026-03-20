"""
Prompts améliorés pour la génération d'images avec Stable Diffusion v1.5.

Objectif :
- Proposer un ensemble de prompts réalistes et variés
- Tous en anglais pour maximiser la compatibilité avec les embeddings CLIP
- Fournir des helpers pour échantillonner facilement un prompt pendant l'entraînement
  ou en phase d'évaluation / démo.
"""

from __future__ import annotations

import random
from typing import List


# ---------------------------------------------------------------------------
# Liste de prompts améliorés
# ---------------------------------------------------------------------------

# IMPORTANT :
# - Tous les prompts sont en anglais
# - Longueur cible >= 15 mots
# - Focalisation sur des portraits réalistes, avec variation d'âge, de genre,
#   d'expression, de style de lumière et de contexte.
# - Compatibles SD1.5 et SDXL ; certains prompts incluent des
#   mots-clés classiques pour SDXL ("masterpiece", "best quality", etc.).

IMPROVED_PROMPTS: List[str] = [
    "a hyperrealistic portrait photo of a young woman with brown hair, soft studio lighting, detailed skin texture, 8k resolution, sharp focus, neutral background, subtle makeup",
    "a cinematic close-up of a middle-aged man with a short beard, dramatic side lighting, high contrast, film grain, ultra detailed face, shallow depth of field, 4k resolution",
    "a portrait of an elderly woman with deep wrinkles and kind eyes, warm indoor lighting, high detail, natural skin tones, sharp focus, soft blurred background, 8k",
    "a close-up portrait of a young man with curly hair, natural daylight from a window, realistic skin imperfections, highly detailed eyes, photo realistic, 4k ultra sharp",
    "a studio portrait of a woman wearing glasses, soft key light and subtle rim light, detailed hair strands, crisp focus, 8k resolution, neutral grey background",
    "a realistic portrait of an older man with gray hair and mustache, soft window light, muted colors, sharp details on eyes and beard, 4k, shallow depth of field",
    "a close-up portrait of a smiling young woman with freckles, golden hour sunlight, warm color grading, ultra detailed skin texture, 8k, cinematic composition",
    "a fashion-style portrait of a young man in a black turtleneck, high contrast studio lighting, clean background, sharp focus, 4k resolution, editorial photography style",
    "a realistic headshot of a businesswoman in her thirties, neutral expression, soft corporate lighting, natural makeup, sharp focus on eyes, 4k resolution, clean background",
    "a portrait of a teenage boy with messy hair, soft daylight, slight smile, visible skin texture, realistic shadows, high detail, 4k ultra sharp",
    "a dramatic black and white portrait of an older man with deep wrinkles, hard side lighting, strong contrast, film noir style, extremely detailed face, 4k",
    "a close-up portrait of a young woman with long blonde hair, studio beauty lighting, smooth but realistic skin, sharp eyes, high detail, 8k resolution, clean background",
    "a realistic portrait of a man with dark curly hair and stubble, indoor ambient light, subtle color grading, sharp facial features, 4k, natural look",
    "a candid style portrait of a young woman laughing, outdoor natural light, background softly blurred, detailed hair movement, realistic skin, 4k ultra detailed",
    "a close-up portrait of an elderly man wearing a flat cap, overcast daylight, muted colors, highly detailed beard and skin, sharp focus on eyes, 4k resolution",
    "a studio portrait of a young woman with short dyed hair, colored rim lights, modern fashion photography style, sharp focus, 8k, detailed skin and lips",
    "a realistic portrait of a middle-aged woman with curly hair, warm indoor lighting, visible pores and skin details, subtle smile, 4k, sharp focus on eyes",
    "a close-up shot of a young man with buzz cut hair, cool toned lighting, high detail, sharp focus, 4k resolution, simple gradient background",
    "a portrait of an elderly woman wearing glasses, soft daylight from a window, detailed wrinkles and hair texture, gentle expression, 4k ultra sharp realism",
    "a realistic studio portrait of a young woman with dark hair in a bun, beauty dish lighting, high frequency skin details, 8k resolution, sharp eyes",
    "a cinematic portrait of a man in his forties with a trimmed beard, warm key light and cool rim light, shallow depth of field, high detail, 4k",
    "a realistic portrait of a young woman with braided hair, outdoor soft light, detailed braids and skin texture, natural colors, sharp focus, 4k",
    "a close-up portrait of a teenage girl with long straight hair, subtle smile, diffused lighting, visible fine hair strands, high detail, 4k ultra sharp",
    "a dramatic portrait of a middle-aged man with shaved head, hard top lighting, strong shadows, high contrast, ultra detailed facial features, 4k",
    "a realistic portrait of an older woman with short gray hair, neutral expression, soft indoor lighting, detailed skin and eyes, 4k resolution, shallow depth of field",
    "a close-up portrait of a young man with freckles and short curly hair, overcast outdoor light, realistic skin imperfections, 4k, sharp focus on eyes",
    "a studio portrait of a young woman wearing subtle jewelry, soft beauty lighting, high detail on eyes and lips, smooth but realistic skin, 8k resolution",
    "a realistic portrait of a middle-aged man with slight smile, office style lighting, natural skin tones, sharp focus, 4k, clean background",
    "a close-up portrait of a young woman with wavy hair, golden backlight, lens flare, high detail, cinematic warm color grading, 4k resolution",
    "a realistic portrait of an elderly man with long white beard, soft side lighting, detailed hair strands, deep wrinkles, 4k ultra sharp resolution",
    "a portrait of a young woman with short curly afro hair, studio softbox lighting, high detail skin texture, sharp eyes, 4k, neutral background",
    "a realistic portrait of a middle-aged woman with light makeup, soft daylight, natural colors, detailed eyes and eyebrows, 4k, shallow depth of field",
    "a close-up portrait of a young man wearing round glasses, cool indoor lighting, reflections in the lenses, sharp focus on eyes, 4k ultra detailed",
    "a realistic portrait of an older woman with gentle smile, warm indoor light, high detail on wrinkles and hair, 4k, soft background blur",
    "a cinematic portrait of a young man looking sideways, single strong key light, dark background, dramatic contrast, detailed skin texture, 4k",
    "a realistic portrait of a young woman with dark straight hair and bangs, studio beauty lighting, highly detailed eyes and eyelashes, 8k resolution",
    "a close-up portrait of an elderly woman with expressive eyes, cool daylight, realistic skin texture, high detail, 4k, subtle background blur",
    "a realistic portrait of a young man with medium length hair, soft indoor light, natural color grading, visible skin pores, 4k, sharp focus",
    "a studio portrait of a young woman with long curly hair, strong key light and soft fill, detailed hair strands, 8k resolution, sharp focus",
    "a realistic close-up of a middle-aged man with faint smile lines, neutral lighting, high detail, 4k, minimal background distractions",
    "a portrait of a young woman in three quarter view, soft daylight, realistic skin shading, detailed eyes and lips, 4k ultra sharp",
    "a close-up portrait of an elderly man with sun damaged skin, outdoor soft light, detailed wrinkles and texture, 4k resolution, shallow depth of field",
    "a realistic portrait of a young woman with subtle freckles and brown eyes, studio soft light, detailed eyelashes, 8k resolution, clean background",
    "a cinematic portrait of a man with stubble and messy hair, warm backlight and cool fill light, high detail, filmic color grading, 4k",
    "a realistic portrait of a young woman with ponytail hairstyle, neutral indoor light, natural makeup, detailed hairline and skin, 4k resolution",
    "a close-up portrait of a young man with dark skin tone, soft specular highlights, detailed pores and facial structure, 4k, sharp eyes",
    "a realistic portrait of an older woman with short curly gray hair, warm indoor lighting, high detail on hair and eyes, 4k resolution",
    "a close-up portrait of a young woman with intense gaze, dramatic split lighting, dark background, ultra detailed face, 4k resolution",
    "a realistic portrait of a middle-aged man with glasses and short hair, office style lighting, natural colors, detailed eyes and frames, 4k",
    "a studio portrait of a young woman with long straight black hair, softbox lighting, subtle catchlights in eyes, highly detailed hair, 8k resolution",
    "a realistic portrait of a young man smiling gently, outdoor overcast light, visible teeth details, natural skin tones, 4k, shallow depth of field",
    "masterpiece, best quality, ultra detailed professional studio portrait of a beautiful young woman, soft rim lighting, 8k, sharp focus, realistic skin texture, subtle makeup, clean background",
    "masterpiece, ultra high res, cinematic close-up of an older man with gray beard, dramatic lighting, detailed pores and wrinkles, 8k, sharp focus, moody background, film still",
    "best quality, ultra detailed studio headshot of a young man with curly hair, soft diffused light, highly detailed eyes and eyelashes, 8k, smooth background bokeh, realistic photo",
    "masterpiece, photorealistic portrait of a smiling middle-aged woman, beauty lighting, high detail skin, 8k, sharp focus on eyes, professional photography, clean white background",
    "masterpiece, ultra realistic close-up portrait of an elderly woman with expressive eyes, soft window light, detailed wrinkles, 8k, shallow depth of field, documentary style photo",
]


# ---------------------------------------------------------------------------
# Fonctions utilitaires
# ---------------------------------------------------------------------------

def get_random_prompt() -> str:
    """
    Retourne un prompt aléatoire parmi la liste IMPROVED_PROMPTS.
    
    Utilisable dans un notebook, pendant l'évaluation ou dans un script
    d'inférence pour obtenir rapidement un exemple de description réaliste.
    """
    if not IMPROVED_PROMPTS:
        raise RuntimeError("IMPROVED_PROMPTS is empty.")
    return random.choice(IMPROVED_PROMPTS)


def get_prompt_by_index(idx: int) -> str:
    """
    Retourne le prompt à l'index donné (avec gestion des erreurs simple).
    
    Args:
        idx: index dans la liste IMPROVED_PROMPTS (supporte les indices négatifs
             comme une liste Python standard).
    """
    if not IMPROVED_PROMPTS:
        raise RuntimeError("IMPROVED_PROMPTS is empty.")
    try:
        return IMPROVED_PROMPTS[idx]
    except IndexError as exc:
        raise IndexError(
            f"Index {idx} is out of range for IMPROVED_PROMPTS (len={len(IMPROVED_PROMPTS)})."
        ) from exc


