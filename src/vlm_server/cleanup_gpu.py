#!/usr/bin/env python3
"""Script de nettoyage de la mémoire GPU pour les serveurs MCP."""

import argparse
import gc
import logging
import subprocess

import torch

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def kill_python_gpu_processes(force=False):
    """Tue les processus Python utilisant la GPU."""
    try:
        # Lister les processus GPU
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,name",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            logger.warning("⚠️ nvidia-smi non disponible")
            return

        processes = []
        for line in result.stdout.strip().split("\n"):
            if line:
                parts = line.split(", ")
                if len(parts) == 2:
                    pid, name = parts
                    if "python" in name.lower():
                        processes.append((int(pid), name))

        if not processes:
            logger.info("✅ Aucun processus Python utilisant la GPU")
            return

        logger.info(f"🔍 {len(processes)} processus Python trouvés sur GPU:")
        for pid, name in processes:
            logger.info(f"   PID {pid}: {name}")

        if force:
            for pid, _name in processes:  # Renamed unused variable
                try:
                    subprocess.run(["kill", "-9", str(pid)], check=True)
                    logger.info(f"   ✅ Processus {pid} tué")
                except subprocess.CalledProcessError:
                    logger.warning(f"   ⚠️ Impossible de tuer le processus {pid}")
        else:
            logger.info("💡 Utilisez --force pour tuer ces processus")

    except Exception as e:
        logger.error(f"❌ Erreur: {e}")


def cleanup_gpu_memory():
    """Nettoie la mémoire GPU."""
    if not torch.cuda.is_available():
        logger.info("ℹ️ CUDA non disponible")
        return

    try:
        logger.info("🧹 Nettoyage de la mémoire GPU...")

        # Afficher l'état avant
        before_allocated = torch.cuda.memory_allocated(0) / 1024**3
        before_reserved = torch.cuda.memory_reserved(0) / 1024**3
        logger.info(f"📊 Avant: Allouée={before_allocated:.2f}GB, Réservée={before_reserved:.2f}GB")

        # Garbage collection Python
        logger.info("   -> Garbage collection Python...")
        gc.collect()

        # Vider le cache CUDA
        logger.info("   -> Vidage du cache CUDA...")
        for _ in range(3):
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Reset des statistiques
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()

        # Afficher l'état après
        after_allocated = torch.cuda.memory_allocated(0) / 1024**3
        after_reserved = torch.cuda.memory_reserved(0) / 1024**3
        logger.info(f"📊 Après: Allouée={after_allocated:.2f}GB, Réservée={after_reserved:.2f}GB")

        if after_reserved > 0.1:
            logger.warning(
                f"⚠️ {after_reserved:.2f}GB encore réservée. "
                "Définissez PYTORCH_NO_CUDA_MEMORY_CACHING=1 pour forcer la libération."
            )
        else:
            logger.info("✅ Mémoire GPU nettoyée")

    except Exception as e:
        logger.error(f"❌ Erreur lors du nettoyage: {e}")


def reset_gpu(force=False):
    """Reset complet de la GPU (nécessite sudo)."""
    try:
        logger.info("🔄 Reset de la GPU...")

        if force:
            # Reset forcé
            result = subprocess.run(
                ["sudo", "nvidia-smi", "--gpu-reset"],
                capture_output=True,
                text=True,
            )

            if result.returncode == 0:
                logger.info("✅ GPU reset avec succès")
            else:
                logger.error(f"❌ Échec du reset: {result.stderr}")
        else:
            logger.info("💡 Utilisez --force pour forcer le reset GPU (nécessite sudo)")

    except Exception as e:
        logger.error(f"❌ Erreur: {e}")


def main():
    """Point d'entrée principal."""
    parser = argparse.ArgumentParser(description="Nettoie la mémoire GPU pour les serveurs MCP")

    parser.add_argument(
        "--kill-all",
        action="store_true",
        help="Tuer tous les processus Python utilisant la GPU",
    )

    parser.add_argument(
        "--reset",
        action="store_true",
        help="Reset complet de la GPU (nécessite sudo)",
    )

    parser.add_argument(
        "--force",
        action="store_true",
        help="Force les actions destructives",
    )

    args = parser.parse_args()

    # Actions
    if args.kill_all:
        kill_python_gpu_processes(force=args.force)

    if args.reset:
        reset_gpu(force=args.force)

    # Toujours nettoyer la mémoire
    cleanup_gpu_memory()


if __name__ == "__main__":
    main()
