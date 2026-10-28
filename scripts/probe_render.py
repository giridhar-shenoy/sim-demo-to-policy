"""Report which MuJoCo offscreen GL backends work here (egl, osmesa).

Each backend is tried in a fresh interpreter because MUJOCO_GL is read at import time.
Exit code 0 if at least one backend renders a non-blank frame.
"""

import subprocess
import sys

SNIPPET = """
import mujoco, numpy as np
m = mujoco.MjModel.from_xml_string(
    '<mujoco><worldbody><light pos="0 0 1"/>'
    '<geom type="box" size=".1 .1 .1" rgba="1 0 0 1"/></worldbody></mujoco>')
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
with mujoco.Renderer(m, 64, 64) as r:
    r.update_scene(d)
    img = r.render()
print(int(img.mean() > 0))
"""


def probe(backend: str) -> bool:
    env = {"MUJOCO_GL": backend, "PYOPENGL_PLATFORM": backend, "PATH": "/usr/bin:/bin"}
    proc = subprocess.run(
        [sys.executable, "-c", SNIPPET], env=env, capture_output=True, text=True, timeout=120
    )
    ok = proc.returncode == 0 and proc.stdout.strip().endswith("1")
    detail = "ok" if ok else (proc.stderr.strip().splitlines() or ["no output"])[-1]
    print(f"MUJOCO_GL={backend}: {detail}")
    return ok


if __name__ == "__main__":
    results = [probe(b) for b in ("egl", "osmesa")]
    sys.exit(0 if any(results) else 1)
