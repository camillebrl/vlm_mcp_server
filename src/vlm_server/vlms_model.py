"""Modèle VLMS (InternVL3) pour l'évaluation de pertinence d'images et génération de réponses."""

import gc
import logging
import traceback

import torch
import torchvision.transforms as T  # type: ignore[import-untyped]
from PIL import Image
from torchvision.transforms.functional import InterpolationMode  # type: ignore[import-untyped]
from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger(__name__)


class VLMSModel:
    """Modèle de vision-langage pour l'analyse d'images avec InternVL3.

    Cette classe gère le chargement, l'initialisation et l'utilisation du modèle
    InternVL3 pour évaluer la pertinence des images et générer des réponses
    textuelles basées sur des images et des requêtes.
    """

    def __init__(self, model_path="OpenGVLab/InternVL3-1B"):
        """Initialise le modèle VLMS (InternVL3) pour l'évaluation d'images.

        Args:
            model_path: Chemin vers le modèle VLMS préentraîné
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.IMAGENET_MEAN = (0.485, 0.456, 0.406)
        self.IMAGENET_STD = (0.229, 0.224, 0.225)
        self.init_model()

    def build_transform(self, input_size):
        """Crée un pipeline de transformation pour les images."""
        transform = T.Compose(
            [
                T.Lambda(lambda img: (img.convert("RGB") if img.mode != "RGB" else img)),
                T.Resize(
                    (input_size, input_size),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                T.ToTensor(),
                T.Normalize(mean=self.IMAGENET_MEAN, std=self.IMAGENET_STD),
            ]
        )
        return transform

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        """Trouve le ratio d'aspect le plus proche."""
        best_ratio_diff = float("inf")
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        """Prétraitement dynamique des images pour InternVL3."""
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = {
            (i, j)
            for n in range(min_num, max_num + 1)
            for i in range(1, n + 1)
            for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        }
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size,
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def load_image(self, image_file, input_size=448, max_num=12):
        """Charge et prétraite une image pour InternVL3."""
        image = Image.open(image_file).convert("RGB")
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    def init_model(self):
        """Initialise le modèle VLMS."""
        if self.model is not None and self.tokenizer is not None:
            return

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Distribution sur GPU
        torch_dtype = torch.bfloat16

        try:
            # Chargement du modèle
            self.model = AutoModel.from_pretrained(
                self.model_path,
                load_in_8bit=True,
                torch_dtype=torch_dtype,
                use_flash_attn=bool(torch.cuda.is_available()),
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).eval()

            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True, use_fast=False)

            device_info = next(self.model.parameters()).device
            dtype_info = next(self.model.parameters()).dtype
            logger.info(f"Modèle VLMS chargé sur {device_info} avec type {dtype_info}")
        except Exception as e:
            logger.error(f"Erreur lors du chargement du modèle VLMS: {e}")
            logger.error(traceback.format_exc())
            raise

    def evaluate_image_relevance(self, image_path, query):
        """Évalue si une image est pertinente pour une requête.

        Args:
            image_path: Chemin vers le fichier image à évaluer
            query: Requête utilisateur pour évaluer la pertinence

        Returns:
            Booléen indiquant si l'image est pertinente
        """
        logger.info(f"image path received by evaluate_image_relevance : {image_path}")
        if self.model is None or self.tokenizer is None:
            self.init_model()

        try:
            # Utiliser les paramètres par défaut du modèle
            pixel_values = self.load_image(image_path)

            # Déterminer le périphérique et le type du modèle
            model_device = next(self.model.parameters()).device
            model_dtype = next(self.model.parameters()).dtype

            # Préparation de la requête
            prompt = f"<image>\nIs this image useful for the question: '{query}'? Answer with 'Yes' or 'No'."

            # Configuration pour la génération
            generation_config = {"max_new_tokens": 10, "do_sample": False}

            # Nettoyage mémoire avant exécution
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            with torch.no_grad():
                # Déplacer les données vers le même périphérique que le modèle
                pixel_values = pixel_values.to(model_dtype).to(model_device)

                # Utiliser la méthode chat du modèle
                response = self.model.chat(self.tokenizer, pixel_values, prompt, generation_config)
            logger.info(f"evaluate image relevance function returned: {response}")
            # Nettoyage mémoire
            del pixel_values
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Déterminer si l'image est pertinente
            is_relevant = "yes" in response.lower()
            return is_relevant
        except Exception as e:
            logger.error(f"Erreur évaluation image: {str(e)}")
            logger.error(traceback.format_exc())
            # Par défaut, considérer l'image comme pertinente en cas d'erreur
            return True

    def generate_response(self, images_paths, query, filepaths=None, page_numbers=None):
        """Génère une réponse textuelle basée sur les images et la requête.

        Args:
            images_paths: Liste des chemins vers les fichiers images à analyser
            query: Requête utilisateur
            filepaths: Liste de chemins de fichiers sources (optionnel)
            page_numbers: Liste de numéros de pages (optionnel)

        Returns:
            Réponse textuelle du modèle
        """
        if self.model is None or self.tokenizer is None:
            self.init_model()

        try:
            # Limite le nombre d'images à traiter
            max_images = 5
            if len(images_paths) > max_images:
                logger.info(f"Limitant l'analyse à {max_images} images sur {len(images_paths)}")
                images_paths = images_paths[:max_images]
                if filepaths:
                    filepaths = filepaths[:max_images]
                if page_numbers:
                    page_numbers = page_numbers[:max_images]

            # Aucune image pertinente
            if not images_paths:
                return "Aucune image pertinente trouvée pour cette requête."

            # Initialiser filepaths et page_numbers si non fournis
            if not filepaths:
                filepaths = ["document inconnu"] * len(images_paths)
            if not page_numbers:
                page_numbers = ["inconnue"] * len(images_paths)

            # Configuration du modèle
            model_device = next(self.model.parameters()).device
            model_dtype = next(self.model.parameters()).dtype

            # Configuration pour la génération
            generation_config = {"max_new_tokens": 200, "do_sample": False}

            # Charger les images
            # num_patches_list = []
            pix_values = []
            for image_path in images_paths:
                img_tensor = self.load_image(image_path)
                # num_patches_list.append(img_tensor.size(0))
                pix_values.append(img_tensor)
            pixel_values = torch.cat(pix_values)

            # Nettoyage mémoire
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            with torch.no_grad():
                # Déplacer les données vers le périphérique du modèle
                pixel_values = pixel_values.to(model_dtype).to(model_device)

                # formatter la question
                # question = "\n".join([f'Image-{i}: <image>' for i in range(len(num_patches_list))]) + f'\n{query}'
                question = f"<image>\n{query}"
                # Générer la réponse
                response = self.model.chat(
                    self.tokenizer,
                    pixel_values,
                    question,
                    generation_config,
                    # num_patches_list=num_patches_list
                )

                # Libérer la mémoire
                del pixel_values

            # Nettoyage final
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            # Joindre les réponses
            return response

        except Exception as e:
            logger.error(f"Erreur lors de la génération de la réponse VLM: {e}")
            logger.error(traceback.format_exc())
            return f"Erreur lors de l'analyse des images: {str(e)}"

    def cleanup(self):
        """Libère les ressources du modèle."""
        self.model = None
        self.tokenizer = None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("Ressources du modèle VLMS libérées")
