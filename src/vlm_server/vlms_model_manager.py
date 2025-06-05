"""Gestionnaire singleton pour le modèle InternVL3 avec chargement/déchargement dynamique."""

import gc
import logging
import time
from threading import Lock
from typing import Optional

import torch

from .vlms_model import VLMSModel

logger = logging.getLogger(__name__)


class VLMSModelManager:
    """Gestionnaire singleton pour le modèle InternVL3 avec chargement/déchargement dynamique."""

    _instance: Optional["VLMSModelManager"] = None
    _lock = Lock()

    def __new__(cls, *args, **kwargs):
        """Crée ou retourne l'instance unique du gestionnaire.

        Implémente le pattern Singleton pour s'assurer qu'une seule instance
        du gestionnaire de modèle existe dans l'application.

        Returns:
            L'instance unique de VLMSModelManager
        """
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        # Ne réinitialiser que si c'est la première fois
        # Ne réinitialiser que si c'est la première fois
        if not hasattr(self, "_initialized"):
            self._initialized = True
            self._model: VLMSModel | None = None
            self._model_path: str | None = None
            self._last_used: float = 0
            self._load_lock = Lock()
            self._reference_count = 0
            self._unload_delay = 0  # Délai en secondes avant déchargement (0 = immédiat)

    def acquire(self, model_path: str = "OpenGVLab/InternVL3-1B") -> VLMSModel:
        """Acquiert une référence au modèle et le charge si nécessaire."""
        with self._load_lock:
            self._reference_count += 1
            self._last_used = time.time()

            # Charger le modèle si pas déjà chargé ou si le chemin a changé
            if self._model is None or self._model_path != model_path:
                self._load_model(model_path)

            # À ce stade, _model ne devrait jamais être None
            if self._model is None:
                raise RuntimeError("Échec du chargement du modèle")

            return self._model

    def release(self):
        """Libère une référence au modèle."""
        with self._load_lock:
            self._reference_count = max(0, self._reference_count - 1)
            self._last_used = time.time()

            # Décharger immédiatement si aucune référence et pas de délai
            if self._reference_count == 0 and self._unload_delay == 0:
                self._unload_model()

    def check_and_unload(self):
        """Vérifie si le modèle doit être déchargé (appelé périodiquement)."""
        with self._load_lock:
            if (
                self._model is not None
                and self._reference_count == 0
                and time.time() - self._last_used > self._unload_delay
            ):
                self._unload_model()

    def _load_model(self, model_path: str):
        """Charge le modèle en mémoire."""
        # Décharger l'ancien modèle si nécessaire
        if self._model is not None:
            self._unload_model()

        try:
            logger.info(f"📦 Chargement du modèle {model_path}...")
            start_time = time.time()

            # Nettoyage mémoire avant chargement
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

            # Créer et charger le modèle
            self._model = VLMSModel(model_path=model_path)
            self._model_path = model_path

            load_time = time.time() - start_time
            logger.info(f"✅ Modèle chargé en {load_time:.2f}s")

            # Afficher l'utilisation mémoire
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated(0) / 1024**3
                reserved = torch.cuda.memory_reserved(0) / 1024**3
                logger.info(f"📊 Mémoire GPU: Allouée={allocated:.2f}GB, Réservée={reserved:.2f}GB")

        except Exception as e:
            logger.error(f"❌ Erreur lors du chargement du modèle: {e}")
            self._model = None
            self._model_path = None
            raise

    def _unload_model(self):
        """Décharge le modèle de la mémoire GPU."""
        if self._model is None:
            return

        try:
            logger.info("🧹 Déchargement du modèle InternVL3...")

            # Appeler la méthode cleanup du modèle
            if hasattr(self._model, "cleanup"):
                self._model.cleanup()

            # Supprimer les références
            del self._model
            self._model = None
            self._model_path = None

            # Forcer le garbage collection Python
            logger.info("   -> Garbage collection...")
            gc.collect()

            # Vider le cache CUDA de manière agressive
            if torch.cuda.is_available():
                logger.info("   -> Vidage agressif du cache CUDA...")

                # Plusieurs passes de nettoyage
                for _ in range(3):
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()

                # Essayer de réinitialiser le contexte CUDA si possible
                try:
                    # Réinitialiser l'allocateur de mémoire (PyTorch >= 1.10)
                    if hasattr(torch.cuda, "memory._set_allocator_settings"):
                        torch.cuda.memory._set_allocator_settings("")

                    # Forcer la libération de toute la mémoire réservée
                    torch.cuda.reset_peak_memory_stats()
                    torch.cuda.reset_accumulated_memory_stats()

                except Exception as e:
                    logger.debug(f"Méthodes de reset CUDA non disponibles: {e}")

                # Afficher la mémoire après nettoyage
                after_allocated = torch.cuda.memory_allocated(0) / 1024**3
                after_reserved = torch.cuda.memory_reserved(0) / 1024**3

                logger.info("✅ Modèle déchargé:")
                logger.info(f"📊 Mémoire GPU finale: Allouée={after_allocated:.2f}GB, Réservée={after_reserved:.2f}GB")

                if after_reserved > 0.5:  # Plus de 500MB encore réservés
                    logger.warning(
                        f"⚠️ {after_reserved:.2f}GB de mémoire encore réservée. "
                        "PyTorch garde la mémoire en cache. Définissez PYTORCH_NO_CUDA_MEMORY_CACHING=1 "
                        "pour forcer la libération complète."
                    )
            else:
                logger.info("✅ Modèle déchargé")

        except Exception as e:
            logger.error(f"❌ Erreur lors du déchargement: {e}")
            # Même en cas d'erreur, s'assurer que les références sont nulles
            self._model = None
            self._model_path = None

            # Tenter quand même un nettoyage basique
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def force_unload(self):
        """Force le déchargement immédiat du modèle, ignore les références."""
        with self._load_lock:
            logger.info("⚠️ Déchargement forcé du modèle InternVL3...")
            self._reference_count = 0
            self._unload_model()

    @property
    def is_loaded(self) -> bool:
        """Vérifie si le modèle est chargé."""
        return self._model is not None

    def get_model_info(self) -> dict:
        """Retourne des informations sur le modèle."""
        info = {
            "loaded": self.is_loaded,
            "model_path": self._model_path,
            "reference_count": self._reference_count,
            "last_used": (time.time() - self._last_used if self._last_used > 0 else None),
            "unload_delay": self._unload_delay,
        }

        if self.is_loaded and torch.cuda.is_available():
            info["gpu_memory_allocated"] = f"{torch.cuda.memory_allocated(0) / 1024**3:.2f} GB"
            info["gpu_memory_reserved"] = f"{torch.cuda.memory_reserved(0) / 1024**3:.2f} GB"

        return info


# Fonction utilitaire pour obtenir le gestionnaire singleton
def get_vlms_manager() -> VLMSModelManager:
    """Retourne l'instance singleton du gestionnaire InternVL3."""
    return VLMSModelManager()
