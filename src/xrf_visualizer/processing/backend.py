"""Device policy (M3): native JAX selection, x64 forced for float64 reductions.

This venv ships a CPU-only jax build, so importing never touches the GPU.
A GPU build (jax[cuda12], NOT TESTED) follows JAX_PLATFORM_NAME: set it to
"cpu" to keep forcing CPU, "cuda" to run on the GPU. See deploy/uv/README.md.
"""
import jax

# The float64 reduction rule (scientific_notes.md) needs x64 before any jax op.
jax.config.update("jax_enable_x64", True)


def current_device():
    """The default jax device all reductions run on."""
    return jax.devices()[0]


def backend_info():
    """Backend report for the UI: {'platform': 'cpu'|'cuda'|..., 'device': str}."""
    d = current_device()
    return {"platform": d.platform, "device": str(d)}


def require_backend(kind):
    """A jax device of the requested kind, or RuntimeError if not available."""
    try:
        devs = jax.devices(kind)
    except (ValueError, RuntimeError) as e:  # unknown backend on this jax build
        raise RuntimeError(f"jax backend {kind!r} not available: {e}") from e
    if not devs:
        raise RuntimeError(f"jax backend {kind!r} not available")
    return devs[0]