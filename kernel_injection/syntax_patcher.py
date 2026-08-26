import re as pyre


def rewrite_custom_call_syntax(mlir_text: str) -> str:
    pattern = pyre.compile(
        r"""
        stablehlo\.custom_call
        \s+
        @(?P<target>[A-Za-z_.$][A-Za-z0-9_.$-]*)
        \(
          (?P<operands>[^)]*)
        \)
        \s*
        \{
          (?P<attrs>[^}]*)
        \}
        """,
        pyre.VERBOSE | pyre.DOTALL,
    )

    def replace(match: pyre.Match) -> str:
        target = match.group("target")
        operands = match.group("operands")
        attrs = match.group("attrs").strip()

        if attrs:
            attrs += ", "

        return (
            f'"stablehlo.custom_call"({operands}) {{'
            f'{attrs}call_target_name = "{target}", '
            f'called_computations = []'
            f'}}'
        )

    return pattern.sub(replace, mlir_text)


def rewrite_dot_general_syntax(mlir_text):
    DOT_GENERAL_PATTERN = pyre.compile(
        r"""
        stablehlo[.]dot_general
        \s+
        (?P<lhs>%[A-Za-z0-9_.$-]+)
        \s*,\s*
        (?P<rhs>%[A-Za-z0-9_.$-]+)
        \s*,\s*
        (?:
        batching_dims
        \s*=\s*
        \[
            (?P<lhs_batch>[^]]*)
        \]
        \s*x\s*
        \[
            (?P<rhs_batch>[^]]*)
        \]
        \s*,\s*
        )?
        contracting_dims
        \s*=\s*
        \[
        (?P<lhs_contract>[^]]*)
        \]
        \s*x\s*
        \[
        (?P<rhs_contract>[^]]*)
        \]
        \s*,\s*
        precision
        \s*=\s*
        \[
        (?P<precision>[^]]*)
        \]
        """,
        pyre.VERBOSE | pyre.DOTALL,
    )

    def rewrite_dot_general(match):
        lhs = match.group("lhs")
        rhs = match.group("rhs")

        lhs_batch = (match.group("lhs_batch") or "").strip()
        rhs_batch = (match.group("rhs_batch") or "").strip()
        lhs_contract = match.group("lhs_contract").strip()
        rhs_contract = match.group("rhs_contract").strip()

        precision_items = [
            item.strip()
            for item in match.group("precision").split(",")
            if item.strip()
        ]
        precision_attr = ", ".join(
            "#stablehlo<precision " + item + ">"
            for item in precision_items
        )

        return (
            '"stablehlo.dot_general"(' + lhs + ", " + rhs + ") {"
            "dot_dimension_numbers = #stablehlo.dot<"
            "lhs_batching_dimensions = [" + lhs_batch + "], "
            "rhs_batching_dimensions = [" + rhs_batch + "], "
            "lhs_contracting_dimensions = [" + lhs_contract + "], "
            "rhs_contracting_dimensions = [" + rhs_contract + "]"
            ">, "
            "precision_config = [" + precision_attr + "]"
            "}"
        )

    return DOT_GENERAL_PATTERN.sub(rewrite_dot_general, mlir_text)


def rewrite_slice_syntax(mlir_text):
    SLICE_PATTERN = pyre.compile(
        r"""
        stablehlo[.]slice
        \s+
        (?P<operand>%[A-Za-z0-9_.$-]+)
        \s+
        \[
        (?P<ranges>
            \s*
            \d+\s*:\s*\d+
            (?:\s*,\s*\d+\s*:\s*\d+)*
            \s*
        )
        \]
        """,
        pyre.VERBOSE | pyre.DOTALL,
    )

    def rewrite_slice(match):
        operand = match.group("operand")
        ranges = match.group("ranges")

        starts = []
        limits = []
        strides = []

        for item in ranges.split(","):
            start, limit = item.strip().split(":")
            starts.append(start.strip())
            limits.append(limit.strip())
            strides.append("1")

        return (
            '"stablehlo.slice"(' + operand + ') {'
            'start_indices = dense<[' + ", ".join(starts) + f']> : tensor<{len(starts)}xi64>, '
            'limit_indices = dense<[' + ", ".join(limits) + f']> : tensor<{len(limits)}xi64>, '
            'strides = dense<[' + ", ".join(strides) + f']> : tensor<{len(strides)} xi64>'
            '}'
        )

    return SLICE_PATTERN.sub(rewrite_slice, mlir_text)


def patch_custom_assembly_ops(module_text: str) -> str:
    module_text = rewrite_custom_call_syntax(module_text)
    module_text = rewrite_dot_general_syntax(module_text)
    return rewrite_slice_syntax(module_text)
