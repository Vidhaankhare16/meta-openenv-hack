"""FastAPI server entry point for the CRISPR Guide RNA Design Environment."""

import uvicorn
from openenv.core.env_server import create_fastapi_app

from .environment import CRISPREnvironment

app = create_fastapi_app(CRISPREnvironment)


def main() -> None:
    """Entry point for `uv run server`."""
    uvicorn.run(
        "server.app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )


if __name__ == "__main__":
    main()
