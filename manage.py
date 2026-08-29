from __future__ import annotations

import os
import sys
import asyncio
import secrets
import subprocess
from functools import partial
from itertools import chain
from pathlib import Path

import httpx
import typer
import uvicorn
from tortoise import Tortoise, connections
from honcho.manager import Manager as HonchoManager

from app.db.config import TORTOISE_ORM

cli = typer.Typer()


@cli.command("work")
def work():
    """Run all the dev services in a single command."""
    manager = HonchoManager()
    project_env = {
        **os.environ,
        "PYTHONPATH": str(Path().resolve(strict=True)),
        "PYTHONUNBUFFERED": "true",
    }
    manager.add_process(
        "server", 
        "python manage.py run-server", 
        env=project_env
    )
    manager.loop()
    sys.exit(manager.returncode)

@cli.command("run-server")
def run_server(
    port: int = 8000,
    host: str = "localhost",
    log_level: str = "debug",
    reload: bool = True,
):
    """Run the API development server (uvicorn)."""
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        log_level=log_level,
        reload=reload,
    )


# ---------------------------------------------------------------------------
# Not needed for PoC

# @cli.command("run-prod-server")
# def run_prod_server():
#     """Run the API production server (gunicorn)."""

#     config_file = str(
#         settings.ROOT_DIR
#                 .joinpath("gunicorn.conf.py")
#                 .resolve(strict=True)
#     )

#     class APPServer(Application):
#         def init(self, parser, opts, args):
#             pass

#         def load_config(self):
#             self.load_config_from_file(config_file)

#         def load(self):
#             return util.import_app("app.main:app")

#     APPServer().run()


# @cli.command("start-app")
# def start_app(app_name: str):
#     """Create a new fastapi component, similar to django startapp"""
#     package_name = app_name.lower().strip().replace(" ", "_").replace("-", "_")
#     app_dir = settings.BASE_DIR / package_name
#     files = {
#         "__init__.py": "",
#         "models.py": "",
#         "schemas.py": "from pydantic import BaseModel",
#         "routes.py": f"from fastapi import APIRouter\n\nrouter = APIRouter(prefix='/{package_name}')",
#         "tests/__init__.py": "",
#         "tests/factories.py": "from factory import Factory, Faker",
#     }
#     app_dir.mkdir()
#     (app_dir / "tests").mkdir()
#     for file, content in files.items():
#         with open(app_dir / file, "w") as f:
#             f.write(content)
#     typer.secho(f"App {package_name} created", fg=typer.colors.GREEN)


# @cli.command("run-worker")
# def run_worker(reload: bool = typer.Option(True)):
#     """Run the saq worker process"""
#     if reload:
#         subprocess.run(["hupper", "-m", "saq", "app.worker.settings", "--web"])
#     else:
#         subprocess.run(["python", "-m", "saq", "app.worker.settings", "--web"])


if __name__ == "__main__":
    cli()
