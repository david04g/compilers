module {
  func.func @main(%x: tensor<7x257xf32>, %residual: tensor<7x257xf32>, %gamma: tensor<257xf32>) -> tensor<7x257xf32> {
    %sum = "fusion.add"(%x, %residual) : (tensor<7x257xf32>, tensor<7x257xf32>) -> tensor<7x257xf32>
    %out = "fusion.rms_norm"(%sum, %gamma) {epsilon = 1.00000000e-05 : f32} : (tensor<7x257xf32>, tensor<257xf32>) -> tensor<7x257xf32>
    return %out : tensor<7x257xf32>
  }
}
