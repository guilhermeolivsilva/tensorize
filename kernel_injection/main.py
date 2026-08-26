from pathlib import Path

from absl import app, flags

import mlir_synth.synth
from mlir_synth.ir import *

from syntax_patcher import patch_custom_assembly_ops


def get_module_body(module: Module):
    """Return the top-level block of a builtin.module."""
    return module.operation.regions[0].blocks[0]


def get_symbol_name(op: Operation) -> str | None:
    """Return sym_name from a symbol op such as func.func."""
    attr = op.attributes.get("sym_name")
    if attr is None:
        return None

    # StringAttr prints as '"foo"'.
    text = str(attr).strip()

    if text.startswith('"') and text.endswith('"'):
        return text[1:-1]

    return text


def find_function(module: Module, name: str) -> Operation | None:
    """Find a top-level func.func with sym_name == name."""
    for op in get_module_body(module).operations:
        if op.name.value == name:
            return op

    return None


def is_target_custom_call(op: Operation,
                          target: str) -> bool:
    """Check whether _op is stablehlo.custom_call @target(...)."""
    if isinstance(op, OpView):
        _op = op.operation
    else:
        _op = op

    if not hasattr(_op, "name"):
        return False

    _op_name_attr = _op.name
    if hasattr(_op.name, "value"):
        _op_name_attr = _op.name.value


    if _op_name_attr != "stablehlo.custom_call":
        return False

    # Depending on the StableHLO revision, the target attribute may be named
    # call_target_name or called_computations / target_name. Print it once if
    # this does not match your version.
    target_attr = str(_op.attributes["call_target_name"])

    return (
        target == target_attr or
        f"@{target}" in target_attr or
        f'"{target}"' in target_attr
    )


def collect_operations(root: Operation) -> list[Operation]:
    """Collect operations recursively before mutating the IR."""
    result: list[Operation] = []

    def visit(op: Operation):
        result.append(op)

        for region in op.regions:
            for block in region.blocks:
                for nested_op in block.operations:
                    visit(nested_op)

    visit(root)
    return result


def walk_operations(op: Operation):
    """Yield every operation nested under `op`, including `op` itself."""
    yield op
    for region in op.regions:
        for block in region.blocks:
            for child in block.operations:
                yield from walk_operations(child)


def replace_all_operand_uses(
    root: Operation,
    old_value,
    new_value,
) -> None:
    """Redirect every operand equal to old_value below root."""
    for user in walk_operations(root):
        for index, operand in enumerate(user.operands):
            if operand == old_value:
                user.operands[index] = new_value


def replace_custom_call_with_func_call(
    module_op: Operation,
    custom_call: Operation,
    callee: str,
) -> Operation:
    """Replace `custom_call` and redirect all uses within module_op."""
    old_results = list(custom_call.results)

    with InsertionPoint(custom_call):
        call = Operation.create(
            "func.call",
            results=[result.type for result in old_results],
            operands=list(custom_call.operands),
            attributes={
                "callee": FlatSymbolRefAttr.get(callee),
            },
            loc=custom_call.location,
        )

    for old_result, new_result in zip(old_results, call.results):
        replace_all_operand_uses(module_op, old_result, new_result)

    custom_call.operation.erase()
    return call


def inject_kernel(
    opaque_calls_module_path: str | Path,
    lifted_kernel_path: str | Path,
    kernel_name: str,
    output_path: str | Path,
    patch_custom_assembly_form: bool = False
) -> None:
    opaque_calls_module_text = Path(opaque_calls_module_path).read_text()

    if patch_custom_assembly_form:
        opaque_calls_module_text = patch_custom_assembly_ops(opaque_calls_module_text)

    lifted_kernel_text = Path(lifted_kernel_path).read_text()

    if patch_custom_assembly_form:
        lifted_kernel_text = patch_custom_assembly_ops(lifted_kernel_text)

    with Context() as ctx:
        mlir_synth.synth.register_dialects()

        opaque_calls_module = Module.parse(opaque_calls_module_text)
        lifted_kernel_module = Module.parse(lifted_kernel_text)

        lifted_kernel_func = find_function(lifted_kernel_module, kernel_name)

        if lifted_kernel_func is None:
            raise RuntimeError(
                f"Could not find func.func @{kernel_name} "
                f"in {lifted_kernel_path}"
            )

        if find_function(opaque_calls_module, kernel_name) is not None:
            raise RuntimeError(
                f"Opaque calls module already contains @{kernel_name}"
            )

        # Detach the function from the kernel module and append it to the
        # golden module. The function remains in the same MLIR Context.
        lifted_kernel_func.detach_from_parent()

        opaque_calls_body = get_module_body(opaque_calls_module)

        with InsertionPoint(opaque_calls_body):
            opaque_calls_body.append(lifted_kernel_func)

        # Collect first, mutate second. This avoids invalidating Python
        # iteration while custom_call.erase() changes the IR.
        all_ops = collect_operations(opaque_calls_module.operation)

        matches = [
            op for op in all_ops
            if is_target_custom_call(op, kernel_name)
        ]

        if not matches:
            raise RuntimeError(
                f"Could not find stablehlo.custom_call "
                f"@{kernel_name} in {opaque_calls_module_path}"
            )

        for custom_call in matches:
            replace_custom_call_with_func_call(
                opaque_calls_module.operation,
                custom_call,
                kernel_name,
            )

        # Optional but useful: verify after structural mutation.
        opaque_calls_module.operation.verify()

        Path(output_path).write_text(str(opaque_calls_module))


FLAGS = flags.FLAGS
flags.DEFINE_string("opaque_path", None, "Path to the module with opaque stablehlo.custom_call ops.")
flags.DEFINE_string("kernel_path", None, "Path to the StableHLO MLIR implementation of a kernel.")
flags.DEFINE_string("kernel_name", None, "Name of the kernel of interest.")
flags.DEFINE_string("output_path", "injected.mlir", "Path to save the output. Defaults to `injected.mlir`.")
flags.DEFINE_boolean("patch_custom_assembly_form", True, "Whether custom MLIR assembly should have its syntax patched.")


def main(argv):
    inject_kernel(
        opaque_calls_module_path=FLAGS.opaque_path,
        lifted_kernel_path=FLAGS.kernel_path,
        kernel_name=FLAGS.kernel_name,
        output_path=FLAGS.output_path,
        patch_custom_assembly_form=FLAGS.patch_custom_assembly_form
    )

if __name__ == "__main__":
    app.run(main)
