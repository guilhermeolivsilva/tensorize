import os 
from pathlib import Path


dir_path = os.path.dirname(os.path.realpath(__file__))



def _parse_type(_type: str, split: bool = True) -> list[str]:
    _type = _type.removeprefix("tensor<").removesuffix(">")

    # Returns a list of `n` strings, where the first `n-1` items are
    # dimension sizes, and the last item is the element type
    if split:
        return _type.split("x")

    return _type


def _instantiate_rms_norm_kernel(func_name: str, operand_types: list[str]) -> str:
    input_type, weight_type, _ = operand_types
    parsed_input_type = _parse_type(input_type)

    # Detect the rank. The final component is the element type, so rank is 
    # len - 1.
    rank = len(parsed_input_type) - 1

    *_, element_type = parsed_input_type
    features, features_element_type = _parse_type(weight_type)

    # Emit `f32_type`. It is identical to `input_type`, except for the element
    # type
    input_dims = parsed_input_type[:-1]
    f32_type = "tensor<" + "x".join(input_dims) + "xf32>"

    # Emit `reduced_type`. It is the first 1 or 2 dimensions of `input_type`
    # (for ranks 2 or 3, respectively), with `f32` element type
    reduced_type = "tensor<" + "x".join(parsed_input_type[:-2]) + "xf32>"

    # Load the base template
    _path = Path(f"{dir_path}/templates/rms_norm/kernel.hlo")
    kernel = _path.read_text()

    # Mapping between placeholders and values to use
    values = {
        "{{FUNC_NAME}}": func_name,
        "{{INPUT_TYPE}}": input_type,
        "{{F32_TYPE}}": f32_type,
        "{{OUTPUT_TYPE}}": input_type,
        "{{FEATURES}}": str(features),
        "{{FEATURES_ELEM_TYPE}}": features_element_type,
        "{{ELEM}}": element_type,
    }

    # Add rank-specific values to the map
    if rank == 3:
        sum_sq_dimensions = "dimensions = dense<[2]> : tensor<1xi64>"
        inv_rms_bcast_dimensions = "broadcast_dimensions = dense<[0, 1]> : tensor<2xi64>"
        weight_bcast_dimensions = "broadcast_dimensions = dense<[2]> : tensor<1xi64>"

    elif rank == 2:
        sum_sq_dimensions = "dimensions = dense<[1]> : tensor<1xi64>"
        inv_rms_bcast_dimensions = "broadcast_dimensions = dense<[0]> : tensor<1xi64>"
        weight_bcast_dimensions = "broadcast_dimensions = dense<[1]> : tensor<1xi64>"

    else:
        raise ValueError(
            "Only rank-2 and rank-3 RMSNorm inputs are supported: "
            + input_type
        )

    rank_specific_values = {
        "{{REDUCED_TYPE}}": reduced_type,
        "{{SUM_SQ_DIMENSIONS}}": sum_sq_dimensions,
        "{{INV_RMS_BCAST_DIMENSIONS}}": inv_rms_bcast_dimensions,
        "{{WEIGHT_BCAST_DIMENSIONS}}": weight_bcast_dimensions
    }
    values = {**values, **rank_specific_values}

    for placeholder, value in values.items():
        kernel = kernel.replace(placeholder, value)

    return kernel


def _make_sin_cos_full(mode: str, trig_function: str):
    _path = Path(f"{dir_path}/templates/rotary_embedding/sin_cos_{mode}.hlo")
    sin_cos_full = _path.read_text()

    # Add the trigonometric function
    return sin_cos_full.replace("{{SIN_COS}}", trig_function)


def _instantiate_rotary_embedding_kernel(func_name: str, operand_types: list[str]) -> str:
    query_type, key_type, cos_type, _ = operand_types

    flat_rows, q_heads, head_dim, element_type = _parse_type(query_type)
    _, k_heads, _, _ = _parse_type(key_type)
    positions, half_dim, cos_element_type = _parse_type(cos_type)

    if int(head_dim) % 2:
        raise ValueError("RoPE requires an even head dimension")

    # Load the base template
    _path = Path(f"{dir_path}/templates/rotary_embedding/kernel.hlo")
    kernel = _path.read_text()

    # Emit `%cos_full` and `%sin_full` following `sin_cos_mode`
    if int(head_dim) == 2 * int(half_dim):
        sin_cos_mode = "concatenate"
    else:
        sin_cos_mode = "reshape"

    cos_full = _make_sin_cos_full(mode=sin_cos_mode, trig_function="cos")
    sin_full = _make_sin_cos_full(mode=sin_cos_mode, trig_function="sin")

    # Mapping between placeholders and values to use
    values = {
        "{{FUNC_NAME}}": func_name,
        "{{COS_FULL_DEF}}": cos_full,
        "{{SIN_FULL_DEF}}": sin_full,
        "{{FLAT_ROWS}}": flat_rows,
        "{{POSITIONS}}": positions,
        "{{ROWS_PER_POSITION}}": str(int(flat_rows) // int(positions)),
        "{{Q_HEADS}}": q_heads,
        "{{K_HEADS}}": k_heads,
        "{{HEAD_DIM}}": head_dim,
        "{{HALF_DIM}}": half_dim,
        "{{COS_ELEM_TYPE}}": cos_element_type,
        "{{ELEM}}": element_type,
    }

    for placeholder, value in values.items():
        kernel = kernel.replace(placeholder, value)


    return kernel


def _instantiate_unified_attention_kernel(func_name: str, operand_types: list[str]) -> str:
    query_type, key_type, value_type = operand_types

    bq, hq, m, d, element_type = _parse_type(query_type)
    bk, hk, n, dk, _ = _parse_type(key_type)
    bv, hv, nv, dv, _ = _parse_type(value_type)

    # Mount the placeholder values depending on the types
    if element_type == "f16":
        neg_inf = "0xFC00"
    elif element_type == "bf16":
        neg_inf = "0xFF80"
    elif element_type == "f32":
        neg_inf = "0xFF800000"

    if (
        bq == bk == bv
        and hq == hk == hv
        and d == dk == dv
        and n == nv
    ):
        # {{B}}       Batch size
        # {{H}}       Number of heads
        # {{M}}       Query length
        # {{N}}       Key/value length
        # {{D}}       Head dimension
        # {{ELEM}}    f16, bf16, or f32
        # {{SCALE}}   1 / sqrt(D), expressed as a literal
        scale = str(1 / (float(d)**(1/2)))

        values = {
            "{{FUNC_NAME}}": func_name,
            "{{B}}": bq,
            "{{H}}": hq,
            "{{M}}": m,
            "{{N}}": n,
            "{{D}}": d,
            "{{ELEM}}": element_type,
            "{{SCALE}}": scale,
            "{{NEG_INF}}": neg_inf,
        }
        kernel_file = "standard_kernel.hlo"

    elif (
        bq == bk == bv
        and d == dk == dv
        and int(hk) % int(hq) == 0
        and hv == hq
        and int(nv) == int(n) * (int(hk) // int(hq))
    ):
        # Q: [B, HQ, M, D]
        # K: [B, HQ * G, N, D]    → reshape → [B, HQ, G, N, D]
        # V: [B, HQ, N * G, D]    → reshape → [B, HQ, G, N, D]
        # O: [B, HQ, M, D]
        g = str(int(hk) // int(hq))
        ng = str(int(n) * int(g))
        scale = str(1 / (float(d)**(1/2)))

        values = {
            "{{FUNC_NAME}}": func_name,
            "{{B}}": bq,
            "{{HQ}}": hq,
            "{{HK}}": hk,
            "{{G}}": g,
            "{{M}}": m,
            "{{N}}": n,
            "{{NG}}": ng,
            "{{D}}": d,
            "{{ELEM}}": element_type,
            "{{SCALE}}": scale,
            "{{NEG_INF}}": neg_inf,
        }
        kernel_file = "expanded_kernel.hlo"

    else:
        raise ValueError("Unsupported unified-attention signature")

    # Load the base template
    _path = Path(f"{dir_path}/templates/unified_attention/{kernel_file}")
    kernel = _path.read_text()

    for placeholder, value in values.items():
        kernel = kernel.replace(placeholder, value)

    return kernel


def _instantiate_silu_and_mul_kernel(func_name: str, operand_types: list[str]) -> str:
    input_type, _ = operand_types

    rows, features, element_type = _parse_type(input_type)

    values = {
        "{{FUNC_NAME}}": func_name,
        "{{ROWS}}": rows,
        "{{FEATURES}}": features,
        "{{ELEM}}": element_type,
    }

    _path = Path(f"{dir_path}/templates/silu_and_mul/kernel.hlo")
    kernel = _path.read_text()
    
    for placeholder, value in values.items():
        kernel = kernel.replace(placeholder, value)

    return kernel


def instantiate_kernel(kernel_name: str, specialized_kernel_name: str, operand_types: list[str]) -> str:
    AVAILABLE_KERNELS = {
        "vendor_rms_norm": _instantiate_rms_norm_kernel,
        "vendor_unified_attention": _instantiate_unified_attention_kernel,
        "vendor_silu_and_mul": _instantiate_silu_and_mul_kernel,
        "vendor_rotary_embedding": _instantiate_rotary_embedding_kernel
    }

    assert kernel_name in AVAILABLE_KERNELS, f"No kernel instantiation function for {kernel_name} available."

    kernel_inst_func = AVAILABLE_KERNELS[kernel_name]

    return kernel_inst_func(specialized_kernel_name, operand_types)
