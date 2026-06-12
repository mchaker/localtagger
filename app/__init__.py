"""localtagger: multi-model anime tagging backend.

A FastAPI microservice that interrogates images with Danbooru taggers
(WD14 v3, animetimm dbv4, Camie v2) via dghs-imgutils / timm, plus the
Kaloscope artist-style classifier. See ``app.main`` for the entrypoint.
"""

__version__ = "2.0.0"
