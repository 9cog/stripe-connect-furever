"""Push a sample job onto the thumbs queue so you can watch the worker pick it up."""
import os

from redis import Redis
from rq import Queue

q = Queue("thumbs", connection=Redis.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0")))
q.enqueue("thumbgen.render", "samples/cat.jpg", 256)
print("enqueued")
