from fastapi import FastAPI
from tortoise.contrib.fastapi import register_tortoise

from app.core.config import config


TORTOISE_ORM = {
    "connections": {"default": config.DB_DSN},
    "apps": {
        "models": {
            "models": [
                # "app.subscribers.models",  # Commented out for PoC
                # "aerich.models", # Commented out for PoC
            ],
            "default_connection": "default",
        },
    },
}


def register_db(app: FastAPI) -> None:
    register_tortoise(
        app,
        config=TORTOISE_ORM,
        generate_schemas=False,
        add_exception_handlers=True,
    )
