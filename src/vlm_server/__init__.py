"""Package vlm_server.

Serveur MCP pour l'analyse d'images avec VLM (InternVL3).
Intégration avec ColPali pour le workflow RAG multimodal.
"""

from .vlm_server import VLMServer
from .vlms_model import VLMSModel
from .vlms_model_manager import VLMSModelManager, get_vlms_manager

__all__ = ["VLMServer", "VLMSModel", "VLMSModelManager", "get_vlms_manager", "cli"]
__version__ = "1.0.0"