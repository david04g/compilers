"""Emit CUDA for the four explicitly supported residual/RMSNorm schedules.

The C++ MLIR pass owns graph verification and schedule enumeration. This small
template backend is intentionally not a general linalg-to-NVVM implementation.
"""
import math

MODES = {"cached", "stream", "partial", "split"}


def validate_plan(plan):
    if plan.get("schema_version") != 1 or plan.get("workload") != "residual_rms_norm":
        raise ValueError("unsupported plan schema or workload")
    m, n, eps = plan.get("rows"), plan.get("width"), plan.get("epsilon")
    if type(m) is not int or type(n) is not int or not (0 < m <= 65535 and 0 < n <= 131072):
        raise ValueError("invalid shape")
    if m * n > 16777216 or not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0:
        raise ValueError("invalid element count or epsilon")
    ids = set()
    for c in plan.get("candidates", []):
        t = c.get("threads")
        if c.get("mode") not in MODES or type(t) is not int or t < 32 or t > 1024 or t & (t - 1):
            raise ValueError("invalid schedule")
        if c.get("id") != f"{c['mode']}-t{t}" or c["id"] in ids:
            raise ValueError("invalid or duplicate candidate ID")
        if type(c.get("legal")) is not bool:
            raise ValueError("missing legality flag")
        ids.add(c["id"])
    legal_ids = {c["id"] for c in plan["candidates"] if c["legal"]}
    if not legal_ids or plan.get("selected") not in legal_ids:
        raise ValueError("selected candidate is not legal")


def emit_cuda(plan, candidate):
    validate_plan(plan)
    if candidate not in plan["candidates"] or not candidate["legal"]:
        raise ValueError("candidate is not a legal member of this plan")
    m, n = plan["rows"], plan["width"]
    t, mode = candidate["threads"], candidate["mode"]
    args = "const float* __restrict__ x, const float* __restrict__ r, const float* __restrict__ g, float* tmp, float* inv, float* y"
    prefix = f"""// Generated from verified fusion MLIR. FP32, no fast math, no atomics.
#define M {m}
#define N {n}
#define B {t}
#define Q {(n + t - 1) // t}
#define EPS {format(plan['epsilon'], '.17e')}f
__device__ __forceinline__ float reduce_row(float s, float* scratch) {{
  int tid = threadIdx.x;
  scratch[tid] = s;
  __syncthreads();
  for (int d = B / 2; d; d /= 2) {{
    if (tid < d) scratch[tid] += scratch[tid + d];
    __syncthreads();
  }}
  return rsqrtf(scratch[0] / (float)N + EPS);
}}
"""
    row_start = """
  int row = blockIdx.x, tid = threadIdx.x;
  extern __shared__ float scratch[];
  float sum = 0.0f;
"""
    if mode == "cached":
        body = row_start + """
  float values[Q];
  #pragma unroll
  for (int j = 0; j < Q; ++j) {
    int col = tid + j * B;
    float v = col < N ? x[row*N+col] + r[row*N+col] : 0.0f;
    values[j] = v;
    sum += v * v;
  }
  float scale = reduce_row(sum, scratch);
  #pragma unroll
  for (int j = 0; j < Q; ++j) {
    int col = tid + j * B;
    if (col < N) y[row*N+col] = values[j] * scale * g[col];
  }
"""
    elif mode == "stream":
        body = row_start + """
  #pragma unroll 1
  for (int col = tid; col < N; col += B) {
    float v = x[row*N+col] + r[row*N+col];
    sum += v * v;
  }
  float scale = reduce_row(sum, scratch);
  #pragma unroll 1
  for (int col = tid; col < N; col += B)
    y[row*N+col] = (x[row*N+col] + r[row*N+col]) * scale * g[col];
"""
    else:
        load = "x[row*N+col] + r[row*N+col]" if mode == "partial" else "tmp[row*N+col]"
        store = "tmp[row*N+col] = v;" if mode == "partial" else ""
        body = row_start + f"""
  #pragma unroll 1
  for (int col = tid; col < N; col += B) {{
    float v = {load};
    {store}
    sum += v * v;
  }}
  float scale = reduce_row(sum, scratch);
  if (tid == 0) inv[row] = scale;
"""
    source = prefix + f'extern "C" __global__ void row_kernel({args}) {{\n{body}}}\n'
    flat_grid = (m * n + t - 1) // t
    launches = [{"name": "row_kernel", "grid": m, "threads": t, "shared": 4 * t}]
    if mode in {"partial", "split"}:
        source += f'''extern "C" __global__ void normalize({args}) {{
  int i = blockIdx.x * B + threadIdx.x;
  if (i < M*N) y[i] = tmp[i] * inv[i/N] * g[i%N];
}}
'''
        launches.append({"name": "normalize", "grid": flat_grid, "threads": t, "shared": 0})
    if mode == "split":
        source += f'''extern "C" __global__ void add({args}) {{
  int i = blockIdx.x * B + threadIdx.x;
  if (i < M*N) tmp[i] = x[i] + r[i];
}}
'''
        launches.insert(0, {"name": "add", "grid": flat_grid, "threads": t, "shared": 0})
    return source, launches
