"""FastAPI server entry point for the CRISPR Guide RNA Design Environment."""

from openenv.core.env_server import create_fastapi_app

from .environment import CRISPREnvironment

app = create_fastapi_app(CRISPREnvironment)
