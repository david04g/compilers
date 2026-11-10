# Supported MLIR input contract

`dataflow-opt` registers real `arith`, `func`, `linalg`, `math`, and `tensor` dialects.
It uses MLIR parsing/verifying and C++ operation inspection, not text matching.
Unsupported programs fail with a diagnostic before any plan is emitted.

- Exactly one nonexternal, single-block `func.func` in the module.
- Arguments are `A: tensor<MxKxf32>`, `B: tensor<KxNxf32>`,
  `bias: tensor<Nxf32>`, and optionally `W2: tensor<NxPxf32>`.
- Positive static dimensions up to 4096; no batches, transposes or layouts.
- One standard `linalg.matmul` initialized by `linalg.fill` of scalar zero into
  `tensor.empty`. Its result must feed the bias operation directly.
- Bias is a two-parallel-loop `linalg.generic` with maps identity, column, identity;
  scalar body `arith.addf(input, bias)` and `linalg.yield`.
- GELU is a two-parallel-loop generic with identity maps and the exact f32 scalar
  expression emitted by `python -m dataflow.generate`:
  `(.5*x) * (1 + tanh(sqrt(2/pi) * (x + .044715*((x*x)*x))))`.
- Each intermediate has one use. The optional second standard zero-initialized
  matmul consumes GELU and W2. The function returns the final result directly.
- Generic outputs are initialized with `tensor.empty` and their payloads do not
  read the output block argument. Only the allowed scalar operations are accepted.

This intentionally strict recognizer is the starter's semantic boundary. Equivalent
rewritten GELU expressions or additional operations may be rejected. Extend its
matcher with positive and negative numerical tests when broadening the contract;
do not accept a `dataflow.kind` tag without validating the operation's semantics.

The shape generator is a convenience for experiments, not a new frontend. Handwritten
MLIR following the same contract compiles through the same C++ optimization pass.

The compiler bounds tile dimensions to 1–256, accepts up to 16 distinct tile triples,
and rejects explicit event expansion above 200,000 events per tile choice. This
keeps a prototype JSON compiler usable; a loop-based plan representation is a later
scaling improvement.
