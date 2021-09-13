// (c) Copyright 2021 Xilinx Inc.

// RUN: air-opt %s -air-linalg-codegen | FileCheck %s
// NOTE: Assertions have been autogenerated by utils/generate-test-checks.py


// CHECK-LABEL:   func @call_mmult(
// CHECK-SAME:                     %[[VAL_0:.*]]: memref<128x128xi32>, %[[VAL_1:.*]]: memref<128x128xi32>,
// CHECK-SAME:                     %[[VAL_2:.*]]: memref<128x128xi32>) {
// CHECK:           %[[VAL_3:.*]] = constant 128 : index
// CHECK:           %[[VAL_4:.*]] = constant 0 : index
// CHECK:           %[[VAL_5:.*]] = constant 32 : index
// CHECK:           %[[VAL_6:.*]] = constant 64 : index
// CHECK:           scf.for %[[VAL_7:.*]] = %[[VAL_4]] to %[[VAL_3]] step %[[VAL_6]] {
// CHECK:             scf.for %[[VAL_8:.*]] = %[[VAL_4]] to %[[VAL_3]] step %[[VAL_6]] {
// CHECK:               scf.for %[[VAL_9:.*]] = %[[VAL_4]] to %[[VAL_3]] step %[[VAL_6]] {
// CHECK:                 scf.parallel (%[[VAL_10:.*]], %[[VAL_11:.*]]) = (%[[VAL_4]], %[[VAL_4]]) to (%[[VAL_6]], %[[VAL_6]]) step (%[[VAL_5]], %[[VAL_5]]) {
// CHECK:                   scf.for %[[VAL_12:.*]] = %[[VAL_4]] to %[[VAL_6]] step %[[VAL_5]] {
// CHECK:                     %[[VAL_13:.*]] = addi %[[VAL_9]], %[[VAL_10]] : index
// CHECK:                     %[[VAL_14:.*]] = addi %[[VAL_7]], %[[VAL_12]] : index
// CHECK:                     %[[VAL_15:.*]] = memref.subview %[[VAL_0]]{{\[}}%[[VAL_13]], %[[VAL_14]]] {{\[}}%[[VAL_5]], %[[VAL_5]]] [1, 1] : memref<128x128xi32> to memref<?x?xi32, #map>
// CHECK:                     %[[VAL_16:.*]] = addi %[[VAL_7]], %[[VAL_12]] : index
// CHECK:                     %[[VAL_17:.*]] = addi %[[VAL_8]], %[[VAL_11]] : index
// CHECK:                     %[[VAL_18:.*]] = memref.subview %[[VAL_1]]{{\[}}%[[VAL_16]], %[[VAL_17]]] {{\[}}%[[VAL_5]], %[[VAL_5]]] [1, 1] : memref<128x128xi32> to memref<?x?xi32, #map>
// CHECK:                     %[[VAL_19:.*]] = addi %[[VAL_9]], %[[VAL_10]] : index
// CHECK:                     %[[VAL_20:.*]] = addi %[[VAL_8]], %[[VAL_11]] : index
// CHECK:                     %[[VAL_21:.*]] = memref.subview %[[VAL_2]]{{\[}}%[[VAL_19]], %[[VAL_20]]] {{\[}}%[[VAL_5]], %[[VAL_5]]] [1, 1] : memref<128x128xi32> to memref<?x?xi32, #map>
// CHECK:                     %[[VAL_22:.*]] = memref.alloc(%[[VAL_5]], %[[VAL_5]]) : memref<?x?xi32, 2>
// CHECK:                     %[[VAL_23:.*]] = memref.alloc(%[[VAL_5]], %[[VAL_5]]) : memref<?x?xi32, 2>
// CHECK:                     %[[VAL_24:.*]] = memref.alloc(%[[VAL_5]], %[[VAL_5]]) : memref<?x?xi32, 2>
// CHECK:                     linalg.copy(%[[VAL_15]], %[[VAL_22]]) : memref<?x?xi32, #map>, memref<?x?xi32, 2>
// CHECK:                     linalg.copy(%[[VAL_18]], %[[VAL_23]]) : memref<?x?xi32, #map>, memref<?x?xi32, 2>
// CHECK:                     linalg.copy(%[[VAL_21]], %[[VAL_24]]) : memref<?x?xi32, #map>, memref<?x?xi32, 2>
// CHECK:                     linalg.matmul ins(%[[VAL_22]], %[[VAL_23]] : memref<?x?xi32, 2>, memref<?x?xi32, 2>) outs(%[[VAL_24]] : memref<?x?xi32, 2>)
// CHECK:                     linalg.copy(%[[VAL_24]], %[[VAL_21]]) : memref<?x?xi32, 2>, memref<?x?xi32, #map>
// CHECK:                     memref.dealloc %[[VAL_22]] : memref<?x?xi32, 2>
// CHECK:                     memref.dealloc %[[VAL_23]] : memref<?x?xi32, 2>
// CHECK:                     memref.dealloc %[[VAL_24]] : memref<?x?xi32, 2>
// CHECK:                   }
// CHECK:                   scf.yield
// CHECK:                 }
// CHECK:               }
// CHECK:             }
// CHECK:           }
// CHECK:           return
// CHECK:         }
module  {
  func @task(%arg0: tensor<128x128xi32>, %arg1: tensor<128x128xi32>) -> tensor<128x128xi32> {
    %0 = "aten.type_cast"(%arg0) : (tensor<128x128xi32>) -> memref<128x128xi32>
    %1 = "aten.type_cast"(%arg1) : (tensor<128x128xi32>) -> memref<128x128xi32>
    %2 = memref.alloc() : memref<128x128xi32>
    linalg.matmul {__internal_linalg_transform__ = "ACDC_mmult"} ins(%0, %1 : memref<128x128xi32>, memref<128x128xi32>) outs(%2 : memref<128x128xi32>)
    %3 = "aten.type_cast"(%2) : (memref<128x128xi32>) -> tensor<128x128xi32>
    return %3 : tensor<128x128xi32>
  }
}

