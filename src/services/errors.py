"""Service-layer domain errors.

These are raised by services when an operation cannot be completed as
requested. They are distinct from ``ValueError`` (which the routers map to
4xx client errors) because they describe a *server-side* step that did not
happen — the caller's request was well-formed.
"""


class DeployError(RuntimeError):
    """A deployment step that the reported outcome depends on did not happen.

    Raised instead of silently degrading. A deploy that never reached the
    cluster must not be reported as ``deployed``: the routing table would be
    written with an endpoint pointing at a Service that does not exist, and
    every consumer of that endpoint would fail with no upstream signal.

    Mapped to HTTP 502 by ``routers/deploy.py``.
    """
