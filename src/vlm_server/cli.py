"""Interface CLI corrigée pour le serveur VLM MCP."""

import argparse
import asyncio
import contextlib
import logging
import signal
import sys

from mcp.server.stdio import stdio_server

from .vlm_server import VLMServer

# Configuration du logging - UNIQUEMENT vers fichier pour MCP
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("vlm_server.log", mode="a")],
)

logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Créer le parser d'arguments."""
    parser = argparse.ArgumentParser(
        description="Serveur MCP pour l'analyse d'images avec VLM (InternVL3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model",
        default="OpenGVLab/InternVL3-1B",
        help="Modèle VLM à utiliser (défaut: OpenGVLab/InternVL3-1B)",
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Niveau de log (défaut: INFO)",
    )

    return parser


async def async_main():
    """Point d'entrée asynchrone principal."""
    parser = create_parser()
    args = parser.parse_args()

    # Configurer le niveau de log
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    logger.info("🚀 Démarrage du serveur VLM MCP...")
    logger.info(f"🤖 Modèle: {args.model}")

    # Variable pour gérer l'arrêt propre
    shutdown_event = asyncio.Event()

    # Gestionnaire de signal pour arrêt propre
    def signal_handler(sig, frame):
        logger.info(f"📛 Signal {sig} reçu, arrêt en cours...")
        shutdown_event.set()

    # Installer les gestionnaires de signaux
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Créer le serveur
    try:
        server = VLMServer(model_path=args.model)
    except Exception as e:
        logger.error(f"❌ Erreur lors de l'initialisation du serveur: {e}")
        sys.exit(1)

    # Lancer le serveur avec stdio
    try:
        async with stdio_server() as (read_stream, write_stream):
            logger.info("✅ Serveur VLM prêt")
            logger.info(
                "🔧 Outils disponibles: analyze_images_with_query, check_image_relevance, analyze_single_image"
            )

            # Créer une tâche pour le serveur
            server_task = asyncio.create_task(server.run(read_stream, write_stream))

            # Créer une tâche pour surveiller l'arrêt
            shutdown_task = asyncio.create_task(shutdown_event.wait())

            # Attendre que l'une des deux tâches se termine
            done, pending = await asyncio.wait(
                [server_task, shutdown_task],
                return_when=asyncio.FIRST_COMPLETED,
            )

            # Annuler les tâches en attente
            for task in pending:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task

            # Vérifier si le serveur s'est terminé avec une erreur
            if server_task in done:
                try:
                    await server_task  # Re-raise any exception
                except Exception as e:
                    logger.error(f"❌ Erreur du serveur: {e}")
                    raise

    except KeyboardInterrupt:
        logger.info("👋 Arrêt demandé par l'utilisateur")
    except Exception as e:
        logger.error(f"❌ Erreur fatale: {e}")
        import traceback

        logger.error(traceback.format_exc())
        sys.exit(1)
    finally:
        logger.info("🛑 Serveur arrêté")


def main():
    """Point d'entrée principal."""
    try:
        # Utiliser asyncio.run avec debug désactivé
        asyncio.run(async_main(), debug=False)
    except KeyboardInterrupt:
        # Ignorer KeyboardInterrupt ici car déjà géré dans async_main
        pass
    except Exception as e:
        logger.error(f"❌ Erreur fatale: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
