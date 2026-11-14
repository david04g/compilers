module {
  func.func @main(%arg0: tensor<7x257xf32>, %arg1: tensor<7x257xf32>, %arg2: tensor<257xf32>) -> tensor<7x257xf32> {
    %0 = "fusion.execute"(%arg0, %arg1, %arg2) {epsilon = 9.99999974E-6 : f32, mode = "cached", threads = 256 : i64} : (tensor<7x257xf32>, tensor<7x257xf32>, tensor<257xf32>) -> tensor<7x257xf32>
    return %0 : tensor<7x257xf32>
  }
}
