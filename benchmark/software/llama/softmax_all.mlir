module {
  func.func @softmax(%input: memref<1000xf64>, %output: memref<1000xf64>) {
    %max = memref.alloca() : memref<f64>
    %0 = affine.load %input[0] : memref<1000xf64>
    affine.store %0, %max[] : memref<f64>

    // Compute the max value in the input array
    affine.for %arg2 = 1 to 1000 {
      %1 = affine.load %max[] : memref<f64>
      %2 = affine.load %input[%arg2] : memref<1000xf64>
      %3 = arith.maxf %1, %2 : f64
      affine.store %3, %max[] : memref<f64>
    }

    // Compute the softmax divisor: y = exp(x - max)
    %max_val = affine.load %max[] : memref<f64>
    affine.for %arg3 = 0 to 1000 {
      %4 = affine.load %input[%arg3] : memref<1000xf64>
      %5 = arith.subf %4, %max_val : f64
      %6 = math.exp %5 : f64
      affine.store %6, %output[%arg3] : memref<1000xf64>
    }

    // Compute the tensor-wise sum of the input array
    %sum = memref.alloca() : memref<f64>
    %cst = arith.constant 0.000000e+00 : f64
    affine.store %cst, %sum[] : memref<f64>
    affine.for %arg4 = 0 to 1000 {
      %7 = affine.load %sum[] : memref<f64>
      %8 = affine.load %input[%arg4] : memref<1000xf64>
      %9 = arith.addf %7, %8 : f64
      affine.store %9, %sum[] : memref<f64>
    }

    // Compute the final softmax output by dividing each element by the sum
    %sum_val = affine.load %sum[] : memref<f64>
    affine.for %arg5 = 0 to 1000 {
      %10 = affine.load %output[%arg5] : memref<1000xf64>
      %11 = arith.divf %10, %sum_val : f64
      affine.store %11, %output[%arg5] : memref<1000xf64>
    }

    return
  }
}
