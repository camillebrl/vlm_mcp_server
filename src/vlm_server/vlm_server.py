"""Serveur MCP pour l'analyse d'images avec VLM (InternVL3)."""

import base64
import logging
import traceback
from pathlib import Path
from typing import Any, Sequence  # noqa: UP035

from mcp.server import NotificationOptions, Server
from mcp.server.models import InitializationOptions
from mcp.types import EmbeddedResource, ImageContent, TextContent, Tool

from .vlms_model_manager import get_vlms_manager

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class VLMServer:
    """Serveur MCP pour l'analyse d'images avec VLM."""

    def __init__(self, model_path: str = "OpenGVLab/InternVL3-1B"):
        """Initialise le serveur VLM.

        Args:
            model_path: Chemin vers le modèle VLM
        """
        self.server: Server = Server("vlm-server")
        self.model_path: str = model_path

        logger.info(f"🚀 Initialisation du serveur VLM avec le modèle {model_path}")

        self._setup_handlers()

    def _load_image_as_base64(self, image_path: str) -> str:
        """Charge une image depuis un chemin et la convertit en base64.

        Args:
            image_path: Chemin vers l'image

        Returns:
            String base64 de l'image
        """
        try:
            path = Path(image_path)
            if not path.exists():
                raise FileNotFoundError(f"Image non trouvée: {image_path}")

            with open(path, "rb") as f:
                image_data = f.read()
                return base64.b64encode(image_data).decode("utf-8")
        except Exception as e:
            logger.error(f"Erreur lors du chargement de l'image {image_path}: {e}")
            raise

    def _setup_handlers(self) -> None:
        """Configure les gestionnaires MCP."""

        @self.server.list_tools()
        async def handle_list_tools() -> list[Tool]:
            """Liste les outils disponibles."""
            tools = [
                Tool(
                    name="analyze_images_with_query",
                    description="Analyse des images avec une requête en utilisant le VLM. Vérifie d'abord la pertinence puis génère une réponse.",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "image_paths": {
                                "type": "array",
                                "items": {"type": "string"},
                                "description": "Liste des chemins vers les images",
                            },
                            "query": {
                                "type": "string",
                                "description": "Question ou requête à propos des images",
                            },
                            "check_relevance": {
                                "type": "boolean",
                                "description": "Vérifier la pertinence avant l'analyse",
                                "default": True,
                            },
                            "metadata": {
                                "type": "array",
                                "items": {
                                    "type": "object",
                                    "properties": {
                                        "filepath": {"type": "string"},
                                        "page_number": {"type": "integer"},
                                    },
                                },
                                "description": "Métadonnées optionnelles pour chaque image",
                            },
                        },
                        "required": ["image_paths", "query"],
                    },
                ),
                Tool(
                    name="check_image_relevance",
                    description="Vérifie si une image est pertinente pour une requête donnée",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "image_path": {
                                "type": "string",
                                "description": "Chemin vers l'image",
                            },
                            "query": {
                                "type": "string",
                                "description": "Requête pour vérifier la pertinence",
                            },
                        },
                        "required": ["image_path", "query"],
                    },
                ),
                Tool(
                    name="analyze_single_image",
                    description="Analyse une seule image et répond à une question",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "image_path": {
                                "type": "string",
                                "description": "Chemin vers l'image",
                            },
                            "query": {
                                "type": "string",
                                "description": "Question à propos de l'image",
                            },
                        },
                        "required": ["image_path", "query"],
                    },
                ),
                Tool(
                    name="get_model_status",
                    description="Obtenir l'état actuel du modèle InternVL3 (chargé/déchargé, mémoire utilisée, etc.)",
                    inputSchema={"type": "object", "properties": {}},
                ),
            ]
            return tools

        @self.server.call_tool()
        async def handle_call_tool(
            name: str, arguments: dict
        ) -> Sequence[TextContent | ImageContent | EmbeddedResource]:
            """Exécute un outil."""
            try:
                if name == "analyze_images_with_query":
                    return await self._analyze_images_with_query(arguments)
                elif name == "check_image_relevance":
                    return await self._check_image_relevance(arguments)
                elif name == "analyze_single_image":
                    return await self._analyze_single_image(arguments)
                elif name == "get_model_status":
                    return await self._get_model_status(arguments)
                else:
                    return [TextContent(type="text", text=f"❌ Outil inconnu: {name}")]

            except Exception as e:
                logger.exception(f"Erreur lors de l'exécution de l'outil {name}")
                return [TextContent(type="text", text=f"❌ Erreur: {str(e)}")]

    async def _analyze_images_with_query(self, args: dict[str, Any]) -> list[TextContent]:
        """Analyse des images avec une requête."""
        image_paths = args.get("image_paths", [])
        query = args["query"]
        args.get("check_relevance", True)
        args.get("metadata", [])

        if not image_paths:
            return [
                TextContent(
                    type="text",
                    text="❌ Aucun chemin d'image fourni pour l'analyse",
                )
            ]

        logger.info(f"🔍 Analyse de {len(image_paths)} images avec la requête: '{query}'")

        # Acquérir le modèle pour cette opération
        manager = get_vlms_manager()
        vlm_model = manager.acquire(self.model_path)

        try:
            # 1) Vérifier que chaque fichier existe et collecter les chemins valides
            valid_paths = []
            for path in image_paths:
                try:
                    # On vérifie juste l’existence, pas besoin de base64
                    Path(path).resolve(strict=True)
                    valid_paths.append(path)
                except FileNotFoundError:
                    logger.warning(f"⚠️ Impossible de charger l'image {path}: fichier introuvable")

            if not valid_paths:
                return [TextContent(type="text", text="❌ Aucune image valide fournie")]

            # 2) Vérifier la pertinence image par image
            relevant_paths = []
            for path in valid_paths:
                try:
                    is_relevant = vlm_model.evaluate_image_relevance(path, query)
                    if is_relevant:
                        relevant_paths.append(path)
                        logger.info(f"✅ Image {path} pertinente")
                    else:
                        logger.info(f"❌ Image {path} non pertinente")
                except Exception as e:
                    logger.warning(f"⚠️ Erreur lors de la vérification de l'image {path}: {e}")
                    # En cas d'erreur, on considère l'image comme pertinente
                    relevant_paths.append(path)

            if not relevant_paths:
                return [
                    TextContent(
                        type="text",
                        text=f"❌ Aucune image pertinente trouvée pour la requête '{query}'",
                    )
                ]

            # 3) Extraire filepaths/page_numbers
            filepaths = list(relevant_paths)
            page_numbers = [1 for _ in relevant_paths]  # ou récupérer les vraies métadonnées si disponible

            # 4) Générer la réponse
            response = vlm_model.generate_response(
                relevant_paths,  # on passe directement la liste de chemins
                query,
                filepaths=filepaths,
                page_numbers=page_numbers,
            )

            return [TextContent(type="text", text=response)]

        except Exception as e:
            logger.error(f"Erreur lors de l'analyse: {e}")
            logger.error(traceback.format_exc())
            return [TextContent(type="text", text=f"❌ Erreur lors de l'analyse: {str(e)}")]

        finally:
            # Libérer le modèle
            manager.release()

    async def _check_image_relevance(self, args: dict[str, Any]) -> list[TextContent]:
        """Vérifie la pertinence d'une image."""
        image_path = args["image_path"]
        query = args["query"]

        logger.info(f"🎯 Vérification de pertinence pour l'image: {image_path}")
        logger.info(f"   Requête: '{query}'")

        # Acquérir le modèle pour cette opération
        manager = get_vlms_manager()
        vlm_model = manager.acquire(self.model_path)

        try:
            is_relevant = vlm_model.evaluate_image_relevance(image_path, query)

            result = "✅ L'image est pertinente" if is_relevant else "❌ L'image n'est pas pertinente"

            return [
                TextContent(
                    type="text",
                    text=f"{result} pour la requête: '{query}'\nImage: {image_path}",
                )
            ]

        except Exception as e:
            logger.error(f"Erreur lors de la vérification: {e}")
            return [
                TextContent(
                    type="text",
                    text=f"❌ Erreur lors de la vérification: {str(e)}",
                )
            ]
        finally:
            # Libérer le modèle
            manager.release()

    async def _analyze_single_image(self, args: dict[str, Any]) -> list[TextContent]:
        """Analyse une seule image."""
        image_path = args["image_path"]
        query = args["query"]

        logger.info(f"🔍 Analyse de l'image: {image_path}")
        logger.info(f"   Requête: '{query}'")

        # Acquérir le modèle pour cette opération
        manager = get_vlms_manager()
        vlm_model = manager.acquire(self.model_path)

        try:
            response = vlm_model.generate_response([image_path], query)

            summary = f"📊 Analyse de l'image: {image_path}\n"
            summary += f"🔍 Requête: '{query}'\n\n"
            summary += "🤖 Réponse:\n"
            summary += "─" * 50 + "\n"
            summary += response

            return [TextContent(type="text", text=summary)]

        except Exception as e:
            logger.error(f"Erreur lors de l'analyse: {e}")
            return [TextContent(type="text", text=f"❌ Erreur lors de l'analyse: {str(e)}")]
        finally:
            # Libérer le modèle
            manager.release()

    async def _get_model_status(self, args: dict[str, Any]) -> list[TextContent]:
        """Obtient l'état actuel du modèle InternVL3."""
        try:
            manager = get_vlms_manager()
            info = manager.get_model_info()

            status = "🤖 État du modèle InternVL3\n\n"

            if info["loaded"]:
                status += "✅ État: CHARGÉ\n"
                status += f"📦 Modèle: {info['model_path']}\n"
                status += f"🔗 Références actives: {info['reference_count']}\n"

                if info.get("gpu_memory_allocated"):
                    status += f"💾 Mémoire GPU allouée: {info['gpu_memory_allocated']}\n"
                if info.get("gpu_memory_reserved"):
                    status += f"💾 Mémoire GPU réservée: {info['gpu_memory_reserved']}\n"

                if info["last_used"] is not None:
                    status += f"⏱️ Dernière utilisation: il y a {info['last_used']:.1f} secondes\n"

                status += f"\n💡 Le modèle sera automatiquement déchargé après {info['unload_delay']}s d'inactivité"
            else:
                status += "💤 État: DÉCHARGÉ\n"
                status += "💡 Le modèle sera chargé automatiquement lors de la prochaine utilisation"

            return [TextContent(type="text", text=status)]

        except Exception as e:
            logger.error(f"Erreur lors de la récupération du statut: {e}")
            return [TextContent(type="text", text=f"❌ Erreur: {str(e)}")]

    async def run(self, read_stream, write_stream) -> None:
        """Lance le serveur MCP."""
        logger.info("🌐 Démarrage du serveur VLM...")

        try:
            init_options = InitializationOptions(
                server_name="vlm-server",
                server_version="1.0.0",
                capabilities=self.server.get_capabilities(
                    notification_options=NotificationOptions(),
                    experimental_capabilities={},
                ),
            )

            await self.server.run(read_stream, write_stream, init_options)

        except Exception as e:
            logger.error(f"❌ Erreur serveur: {e}")
            logger.error(traceback.format_exc())
            raise
        finally:
            # Nettoyage
            logger.info("🧹 Arrêt du serveur VLM...")

            try:
                # Forcer le déchargement du modèle s'il est encore chargé
                manager = get_vlms_manager()
                if manager.is_loaded:
                    logger.info("🧹 Déchargement du modèle InternVL3...")
                    manager.force_unload()
            except Exception as e:
                logger.warning(f"Avertissement lors du nettoyage: {e}")
