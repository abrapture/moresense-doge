"""Initialize schemas already shipped in the kitless Docker image.

The standalone importer can otherwise claim its incomplete fallback before
the installed OvPhysX schema package has registered. Call before importing
Isaac Lab's scene/task modules. This changes only this process's USD registry.
"""
import importlib.util
from pathlib import Path


def register_kitless_importer_schemas():
    from pxr import Plug

    registry = Plug.Registry()
    if not registry.GetPluginWithName("physxSchema"):
        if registry.GetPluginWithName("physxSchemaFallback"):
            raise RuntimeError("Register installed PhysX schemas before importing the standalone URDF importer.")
        spec = importlib.util.find_spec("ovphysx")
        if spec is not None and spec.origin is not None:
            info = Path(spec.origin).parent / "schemas/physx/PhysxSchema/resources/plugInfo.json"
            if info.is_file():
                registry.RegisterPlugins(str(info))
    mjc_spec = importlib.util.find_spec("mujoco_usd_converter")
    if mjc_spec is not None and mjc_spec.origin is not None:
        mjc_info = Path(mjc_spec.origin).parent / "plugins/mjcPhysics/resources/plugInfo.json"
        if mjc_info.is_file():
            registry.RegisterPlugins(str(mjc_info))
