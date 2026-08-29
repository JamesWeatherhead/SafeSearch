from tortoise import Tortoise, connections
from .core.fastapi_config import settings
from .db.config import TORTOISE_ORM


class DummyQueue:
    """A dummy queue that doesn't require Redis connection."""
    async def connect(self):
        pass

    async def disconnect(self):
        pass

    async def enqueue(self, *args, **kwargs):
        pass


async def startup(_: dict):
    """Binds a connection set to the db object."""
    await Tortoise.init(config=TORTOISE_ORM)


async def shutdown(_: dict):
    """Closes all connections."""
    await connections.close_all()


# Not using background tasks for PoC
# queue = Queue.from_url(settings.REDIS_URL)
queue = DummyQueue()

settings = {
    "queue": queue,  # DummyQueue for PoC
    "functions": [],  # Empty list for PoC
    "concurrency": 10,
    "startup": startup,
    "shutdown": shutdown,
}

