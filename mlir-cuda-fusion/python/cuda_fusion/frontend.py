"""Generate the narrow MLIR input contract and invoke the native compiler."""
import json
from pathlib import Path
import subprocess


def make_mlir(rows, width, epsilon=1e-5):
    if type(rows) is not int or type(width) is not int or rows < 1 or width < 1:
        raise ValueError("dimensions must be positive integers")
    mat, vec = f"tensor<{rows}x{width}xf32>", f"tensor<{width}xf32>"
    return f'''module {{
  func.func @main(%x: {mat}, %residual: {mat}, %gamma: {vec}) -> {mat} {{
    %sum = "fusion.add"(%x, %residual) : ({mat}, {mat}) -> {mat}
    %out = "fusion.rms_norm"(%sum, %gamma) {{epsilon = {epsilon:.8e} : f32}} : ({mat}, {vec}) -> {mat}
    return %out : {mat}
  }}
}}
'''


def compile_graph(compiler, rows, width, directory, *, options="", epsilon=1e-5):
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    source, output, lowered = directory / "input.mlir", directory / "plan.json", directory / "planned.mlir"
    source.write_text(make_mlir(rows, width, epsilon), encoding="utf-8")
    # Relative output avoids embedding a filesystem path in MLIR pass options.
    proc = subprocess.run([str(Path(compiler).resolve()), source.name,
                           f"--fusion-plan=output=plan.json {options}".rstrip(),
                           "-o", lowered.name], cwd=directory, capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(f"fusion-opt failed:\n{proc.stderr}")
    return json.loads(output.read_text(encoding="utf-8"))
