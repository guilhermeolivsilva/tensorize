from absl import app, flags

from kernel_injector import KernelInjector


FLAGS = flags.FLAGS
flags.DEFINE_string("opaque_path", None, "Path to the module with opaque stablehlo.custom_call ops.")
flags.DEFINE_string("output_path", "injected.mlir", "Path to save the output. Defaults to `injected.mlir`.")


def main(argv):
    opaque_path = FLAGS.opaque_path

    injector = KernelInjector()
    injector.inject_kernel(
        opaque_calls_module_path=opaque_path,
        output_path=FLAGS.output_path,
    )

if __name__ == "__main__":
    app.run(main)
