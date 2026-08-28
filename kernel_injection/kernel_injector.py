import re as pyre

from pathlib import Path
from warnings import warn

from kernel_instantiator import instantiate_kernel

import re as pyre


CUSTOM_CALL_PATTERN = pyre.compile(
    r"""
    (?P<results>
      %[A-Za-z0-9_.$-]+
      (?:\s*:\s*\d+)?
    )
    \s*=\s*

    stablehlo[.]custom_call
    \s+
    @(?P<kernel>[A-Za-z_.$][A-Za-z0-9_.$-]*)
    \s*
    \(
      (?P<operands>[^()]*)       # SSA operands only; no nested parentheses.
    \)
    \s*

    (?P<attrs>
      \{
        [^{}]*
      \}
    )
    \s*

    :
    \s*
    \(
      (?P<operand_types>[^()]*)  # Tensor types have <...>, not (...).
    \)
    \s*
    ->
    \s*
    (?P<result_types>
      \(
        [^()]*
      \)
      |
      [^,\n\r]+
    )
    """,
    pyre.VERBOSE | pyre.DOTALL,
)


def first_result_type(result_types: str) -> str:
    """Return the first result type, stripping an optional outer tuple."""
    result_types = result_types.strip()

    if result_types.startswith("(") and result_types.endswith(")"):
        result_types = result_types[1:-1].strip()

    # The result types in your kernels are tensor<...>, which do not contain
    # top-level commas. A generic scanner keeps this robust for nested types.
    depth_angle = 0
    depth_paren = 0

    for index, char in enumerate(result_types):
        if char == "<":
            depth_angle += 1
        elif char == ">":
            depth_angle -= 1
        elif char == "(":
            depth_paren += 1
        elif char == ")":
            depth_paren -= 1
        elif char == "," and depth_angle == 0 and depth_paren == 0:
            return result_types[:index].strip()

    return result_types.strip()


def type_to_symbol_suffix(mlir_type: str) -> str:
    """Convert tensor<32768x12x128xbf16> to tensor_32768x12x128xbf16."""
    mlir_type = mlir_type.strip()

    if mlir_type.startswith("tensor<") and mlir_type.endswith(">"):
        contents = mlir_type[len("tensor<"):-1]
        return "tensor_" + contents

    # Fallback for non-tensor result types. Keep symbols valid and stable.
    suffix = pyre.sub(r"[^A-Za-z0-9_.$]", "_", mlir_type)
    suffix = pyre.sub(r"_+", "_", suffix).strip("_")
    return suffix


def target_function_name(kernel: str, result_types: str) -> str:
    """
    Map a custom-call target and result types to the injected func.func name.

    Example:
    vendor_rotary_embedding
    + tensor<32768x12x128xbf16>

    becomes:
    vendor_rotary_embedding_tensor_32768x12x128xbf16
    """
    return kernel + "_" + type_to_symbol_suffix(first_result_type(result_types))


def append_functions_before_module_end(
    mlir_text: str,
    functions: list[str],
) -> str:
    if not functions:
        return mlir_text

    closing_brace = mlir_text.rfind("}")

    if closing_brace < 0:
        raise ValueError("Could not find closing module brace")

    insertion = "\n\n" + "\n\n".join(functions) + "\n"

    return (
        mlir_text[:closing_brace]
        + insertion
        + mlir_text[closing_brace:]
    )


class KernelInjector:
    def __init__(self):
        self.kernels_to_insert = []
        self.kernels_to_insert_names = set()

    def rewrite_one_custom_call(self, match: pyre.Match) -> str:
        results = match.group("results").strip()
        kernel = match.group("kernel").strip()
        operands = match.group("operands").strip()
        operand_types = match.group("operand_types").strip()
        result_types = match.group("result_types").strip()

        callee = target_function_name(kernel, operand_types)

        kernel_data = {
            "kernel_name": kernel,
            "specialized_kernel_name": callee,
            "operand_types": operand_types.split(", ")
        }

        if callee not in self.kernels_to_insert_names:
            self.kernels_to_insert.append(kernel_data)
            self.kernels_to_insert_names.add(callee)

        return (
            results
            + " = func.call @"
            + callee
            + "("
            + operands
            + ") : ("
            + operand_types
            + ") -> "
            + result_types
        )


    def replace_custom_calls_with_func_calls(self, mlir_text: str) -> str:
        """
        Replace all custom calls with direct func.call operations.

        The original custom-call attribute dictionary is intentionally discarded:
        api_version, backend_config, called_computations, and related custom-call
        metadata do not belong on func.call.
        """
        return CUSTOM_CALL_PATTERN.sub(self.rewrite_one_custom_call, mlir_text)


    def inject_kernel(
        self,
        opaque_calls_module_path: str | Path,
        output_path: str | Path,
    ) -> None:
        opaque_calls_module_text = Path(opaque_calls_module_path).read_text()
        patched = self.replace_custom_calls_with_func_calls(opaque_calls_module_text)

        functions_to_insert = []
        for kernel_data in self.kernels_to_insert:
            function_body = instantiate_kernel(**kernel_data)
            functions_to_insert.append(function_body)

        patched = append_functions_before_module_end(patched, functions_to_insert)

        Path(output_path).write_text(str(patched))
