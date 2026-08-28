import re as pyre


def rewrite_properties_syntax(mlir_text: str) -> str:
    """Convert every balanced `<{ ... }>` property dictionary to `{ ... }`."""
    output = []
    i = 0

    while i < len(mlir_text):
        if mlir_text.startswith("<{", i):
            depth = 1
            j = i + 2

            while j < len(mlir_text) and depth:
                if mlir_text[j] == "{":
                    depth += 1
                elif mlir_text[j] == "}":
                    depth -= 1
                j += 1

            if depth != 0 or j >= len(mlir_text) or mlir_text[j] != ">":
                raise ValueError("Unbalanced MLIR property dictionary")

            output.append("{")
            output.append(mlir_text[i + 2:j - 1])
            output.append("}")
            i = j + 1
        else:
            output.append(mlir_text[i])
            i += 1

    return "".join(output)


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


def dense_i64(values):
    values = [value.strip() for value in values if value.strip()]

    return (
        "dense<[" + ", ".join(values) + "]> : "
        "tensor<" + str(len(values)) + "xi64>"
    )


def rewrite_gather_syntax(mlir_text):
    GATHER_PATTERN = pyre.compile(
        r"""
        (?P<prefix>
        "stablehlo[.]gather"
        \s*
        \(
            (?P<operand>%[A-Za-z0-9_.$-]+)
            \s*,\s*
            (?P<start_indices>%[A-Za-z0-9_.$-]+)
        \)
        \s*
        \{
            (?P<attrs>.*?)
        \}
        )
        (?P<suffix>
        \s*:
        )
        """,
        pyre.VERBOSE | pyre.DOTALL,
    )


    SLICE_SIZES_PATTERN = pyre.compile(
        r"""
        slice_sizes
        \s*=\s*
        array<i64:
        \s*(?P<values>[^>]*?)
        \s*
        >
        """,
        pyre.VERBOSE | pyre.DOTALL,
    )

    def rewrite_gather(match):
        attrs = match.group("attrs")

        def rewrite_slice_sizes(slice_match):
            values = slice_match.group("values").split(",")
            return "slice_sizes = " + dense_i64(values)

        attrs = SLICE_SIZES_PATTERN.sub(rewrite_slice_sizes, attrs)

        return (
            '"stablehlo.gather"('
            + match.group("operand")
            + ", "
            + match.group("start_indices")
            + ") {"
            + attrs
            + "}"
            + match.group("suffix")
        )

    return GATHER_PATTERN.sub(rewrite_gather, mlir_text)


def patch_custom_assembly_ops(module_text: str) -> str:
    module_text = rewrite_properties_syntax(module_text)
    module_text = rewrite_custom_call_syntax(module_text)
    module_text = rewrite_dot_general_syntax(module_text)
    module_text = rewrite_gather_syntax(module_text)
    return rewrite_slice_syntax(module_text)
