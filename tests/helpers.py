"""Shared test helpers.

``gitops_publish_succeeds`` exists because a GGUF deploy now *fails closed*
when the values file does not reach gitops (see
``services.deploy_service.DeployService.deploy_model``). Any test that wants a
successful GGUF deploy must therefore make the gitops publish actually
succeed, rather than relying on the old behaviour where a skipped or failed
push still reported ``deployed``.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch


def deploy_settings(**overrides: Any) -> Any:
    """Build a settings double from the real test settings.

    Deliberately a real ``Settings`` instance, not a ``MagicMock``: a mock
    auto-creates any attribute as a truthy object, so a *future* settings flag
    read by the deploy path would silently take its "enabled" branch here and
    the test would pass for the wrong reason.

    Args:
        **overrides: Field values to override on the copy.

    Returns:
        A ``Settings`` instance suitable for patching
        ``deploy_service.get_settings``.
    """
    from settings import get_settings

    base: dict[str, Any] = {"gitops_enabled": True, "deployment_tracking_enabled": False}
    base.update(overrides)
    return get_settings().model_copy(update=base)


@contextmanager
def gitops_publish_succeeds(**overrides: Any) -> Iterator[MagicMock]:
    """Patch settings + git subprocesses so a GGUF deploy can succeed.

    Args:
        **overrides: Settings field overrides (e.g.
            ``deployment_tracking_enabled=True``).

    Yields:
        The mock returned by the patched ``create_subprocess_exec``'s factory,
        i.e. the ``mock_exec`` handle, so callers can assert git invocations.
    """
    fake_proc = AsyncMock()
    fake_proc.communicate.return_value = (b"ok", b"")
    fake_proc.returncode = 0

    with (
        patch("services.deploy_service.get_settings", return_value=deploy_settings(**overrides)),
        patch(
            "services.deploy_service.asyncio.create_subprocess_exec",
            return_value=fake_proc,
        ) as mock_exec,
    ):
        yield mock_exec


@contextmanager
def gitops_push_fails(**overrides: Any) -> Iterator[MagicMock]:
    """Patch settings + git subprocesses so ``git push`` fails.

    Args:
        **overrides: Settings field overrides.

    Yields:
        The patched ``create_subprocess_exec`` mock.
    """
    ok_proc = AsyncMock()
    ok_proc.communicate.return_value = (b"ok", b"")
    ok_proc.returncode = 0

    fail_proc = AsyncMock()
    fail_proc.communicate.return_value = (b"", b"push rejected")
    fail_proc.returncode = 1

    async def _exec(*cmd: str, **_kwargs: Any) -> AsyncMock:
        # git add / git commit succeed; only the network step fails, so the
        # test exercises the retry-then-give-up path rather than an early bail.
        return fail_proc if cmd[1] == "push" else ok_proc

    with (
        patch("services.deploy_service.get_settings", return_value=deploy_settings(**overrides)),
        patch(
            "services.deploy_service.asyncio.create_subprocess_exec",
            side_effect=_exec,
        ) as mock_exec,
        patch("services.deploy_service.asyncio.sleep", new=AsyncMock()),
    ):
        yield mock_exec
