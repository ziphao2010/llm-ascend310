"""
ATC operator compiler — generates .om files for any LLaMA model dimensions.
Reads config.json to determine required operator shapes.

Usage:
    python -m engine.compile --model /path/to/model --output /path/to/om_models
"""
import os, sys, json, subprocess, argparse
from typing import List, Tuple


def create_onnx_matmul(m: int, n: int, output_dir: str):
    """Create ONNX model for [1,M] @ [M,N] → [1,N] matmul."""
    import onnx
    from onnx import helper, TensorProto

    A = helper.make_tensor_value_info("A", TensorProto.FLOAT16, [1, m])
    B = helper.make_tensor_value_info("B", TensorProto.FLOAT16, [m, n])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [1, n])

    # For Ascend 310: Cube Unit can't do FP16×FP16→FP16 directly
    # We need to Cast inputs to FP32, matmul in FP32, Cast back to FP16
    A_f32 = helper.make_node("Cast", ["A"], ["A_f32"], name="cast_a", to=TensorProto.FLOAT)
    B_f32 = helper.make_node("Cast", ["B"], ["B_f32"], name="cast_b", to=TensorProto.FLOAT)
    matmul = helper.make_node("MatMul", ["A_f32", "B_f32"], ["Y_f32"], name="matmul")
    cast_back = helper.make_node("Cast", ["Y_f32"], ["Y"], name="cast_y", to=TensorProto.FLOAT16)

    graph = helper.make_graph(
        [A_f32, B_f32, matmul, cast_back],
        f"mm_1_{m}_{n}", [A, B], [Y])
    model = helper.make_model(graph, producer_name="llm-ascend310",
                               opset_imports=[helper.make_opsetid("", 13)])

    path = os.path.join(output_dir, f"mm_1_{m}_{n}.onnx")
    onnx.save(model, path)
    return path


def create_onnx_rmsnorm(dim: int, output_dir: str, eps: float = 1e-6):
    """Create ONNX model for RMSNorm: (hidden, weight) → normalized."""
    import onnx
    from onnx import helper, TensorProto

    H = helper.make_tensor_value_info("H", TensorProto.FLOAT16, [1, dim])
    W = helper.make_tensor_value_info("W", TensorProto.FLOAT16, [dim])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [1, dim])

    # RMSNorm: H / sqrt(mean(H^2) + eps) * W
    # Using Cast→FP32 for precision, Cast→FP16 for output
    h_f32 = helper.make_node("Cast", ["H"], ["h_f32"], to=TensorProto.FLOAT)
    h_sq = helper.make_node("Pow", ["h_f32"], ["h_sq"], name="pow")
    mean_node = helper.make_node("ReduceMean", ["h_sq"], ["mean"], axes=[1], keepdims=1)
    eps_const = helper.make_tensor("eps", TensorProto.FLOAT, [], [eps])
    add_eps = helper.make_node("Add", ["mean", "eps"], ["mean_eps"], name="add_eps")
    sqrt_node = helper.make_node("Sqrt", ["mean_eps"], ["rms"], name="sqrt")
    div_node = helper.make_node("Div", ["h_f32", "rms"], ["normed"], name="div")
    w_f32 = helper.make_node("Cast", ["W"], ["w_f32"], to=TensorProto.FLOAT)
    mul_node = helper.make_node("Mul", ["normed", "w_f32"], ["scaled"], name="mul")
    cast_back = helper.make_node("Cast", ["scaled"], ["Y"], to=TensorProto.FLOAT16)

    graph = helper.make_graph(
        [h_f32, h_sq, mean_node, add_eps, sqrt_node, div_node, w_f32, mul_node, cast_back],
        f"rmsnorm_{dim}", [H, W], [Y],
        initializer=[eps_const])
    model = helper.make_model(graph, producer_name="llm-ascend310",
                               opset_imports=[helper.make_opsetid("", 13)])

    path = os.path.join(output_dir, f"ops_rmsnorm_{dim}.onnx")
    onnx.save(model, path)
    return path


def create_onnx_silu(dim: int, output_dir: str):
    """SiLU activation: x * sigmoid(x)."""
    import onnx
    from onnx import helper, TensorProto

    X = helper.make_tensor_value_info("X", TensorProto.FLOAT16, [dim])
    Y = helper.make_tensor_value_info("Y", TensorProto.FLOAT16, [dim])

    x_f32 = helper.make_node("Cast", ["X"], ["x_f32"], to=TensorProto.FLOAT)
    sig = helper.make_node("Sigmoid", ["x_f32"], ["sig"], name="sigmoid")
    mul = helper.make_node("Mul", ["x_f32", "sig"], ["y_f32"], name="silu")
    cast_back = helper.make_node("Cast", ["y_f32"], ["Y"], to=TensorProto.FLOAT16)

    graph = helper.make_graph([x_f32, sig, mul, cast_back],
                               f"silu_{dim}", [X], [Y])
    model = helper.make_model(graph, producer_name="llm-ascend310",
                               opset_imports=[helper.make_opsetid("", 13)])

    path = os.path.join(output_dir, f"ops_silu_{dim}.onnx")
    onnx.save(model, path)
    return path


def compile_with_atc(onnx_path: str, output_dir: str, soc: str = "Ascend310"):
    """Run ATC to compile ONNX → .om."""
    base = os.path.splitext(os.path.basename(onnx_path))[0]
    om_path = os.path.join(output_dir, f"{base}.om")

    cmd = [
        "atc", "--model=" + onnx_path,
        "--framework=5",
        "--output=" + os.path.join(output_dir, base),
        f"--soc_version={soc}",
        "--log=error"
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  ⚠ ATC failed for {base}: {result.stderr[:200]}")
        return False
    print(f"  ✅ {base}.om")
    return True


def compile_model_ops(config_path: str, output_dir: str, soc: str = "Ascend310",
                      skip_existing: bool = True):
    """Compile all operators needed for a LLaMA model."""
    with open(config_path) as f:
        cfg = json.load(f)

    hs = cfg.get("hidden_size", 1536)
    nl = cfg.get("num_hidden_layers", 24)
    nh = cfg.get("num_attention_heads", 16)
    nkv = cfg.get("num_key_value_heads", cfg.get("num_kv_heads", 2))
    hd = cfg.get("head_dim", 128)
    im = cfg.get("intermediate_size", 4608)

    q_dim = nh * hd
    k_dim = nkv * hd
    half_im = im // 2

    print(f"Compiling operators for {cfg.get('_name_or_path', 'model')}:")
    print(f"  hidden_size={hs}, q_dim={q_dim}, k_dim={k_dim}, im={im}")
    print(f"  Output: {output_dir}")

    os.makedirs(output_dir, exist_ok=True)

    operators = [
        # MatMuls
        ("mm", hs, hs, hs),        # Q projection, O projection, z gate
        ("mm", hs, k_dim, 0),      # K projection
        ("mm", hs, q_dim, 0),      # Q projection (if different)
        ("mm", hs, im, 0),         # gate/up projections
        ("mm", half_im, hs, 0),    # down projection (first half)
        # RMSNorm
        ("rmsnorm", hs, 0, 0),
        # SiLU
        ("silu", im, 0, 0),
        ("silu", hs, 0, 0),
    ]

    # Add unique operators (deduplicate)
    seen = set()
    unique_ops = []
    for op_type, m, k, n in operators:
        if op_type == "mm":
            name = f"mm_1_{m}_{n}"
        elif op_type == "rmsnorm":
            name = f"ops_rmsnorm_{m}"
        elif op_type == "silu":
            name = f"ops_silu_{m}"
        else:
            name = f"ops_{op_type}_{m}"

        if name not in seen:
            seen.add(name)
            unique_ops.append((op_type, m, k, n))

    print(f"\n  {len(unique_ops)} operators to compile:")
    for op_type, m, k, n in unique_ops:
        if op_type == "mm":
            if skip_existing and os.path.exists(os.path.join(output_dir, f"mm_1_{m}_{n}.om")):
                print(f"    ⏭ mm_1_{m}_{n}.om (exists)")
                continue
            onnx = create_onnx_matmul(m, n, output_dir)
        elif op_type == "rmsnorm":
            if skip_existing and os.path.exists(os.path.join(output_dir, f"ops_rmsnorm_{m}.om")):
                print(f"    ⏭ ops_rmsnorm_{m}.om (exists)")
                continue
            onnx = create_onnx_rmsnorm(m, output_dir)
        elif op_type == "silu":
            if skip_existing and os.path.exists(os.path.join(output_dir, f"ops_silu_{m}.om")):
                print(f"    ⏭ ops_silu_{m}.om (exists)")
                continue
            onnx = create_onnx_silu(m, output_dir)

        compile_with_atc(onnx, output_dir, soc)

    print(f"\nDone. Operators in {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Model config.json path", required=True)
    parser.add_argument("--output", help="Output dir for .om files",
                        default="./om_models")
    parser.add_argument("--soc", default="Ascend310")
    parser.add_argument("--force", action="store_true",
                        help="Recompile even if .om exists")
    args = parser.parse_args()

    config_path = args.model if args.model.endswith("config.json") \
                  else os.path.join(args.model, "config.json")
    compile_model_ops(config_path, args.output, args.soc,
                      skip_existing=not args.force)
