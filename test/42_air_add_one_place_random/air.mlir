// (c) Copyright 2021 Xilinx Inc. All Rights Reserved.

module {

func @graph(%arg0 : memref<16xi32>, %arg1 : memref<16xi32>) -> () {
  %herd_cols = constant 1 : index
  %herd_rows = constant 1 : index
  air.launch_herd tile(%tx, %ty) in (%size_x = %herd_cols, %size_y = %herd_rows) args(%ext0 = %arg0, %ext1 = %arg1) : memref<16xi32>, memref<16xi32> attributes { sym_name="herd_0"} {
    %c0 = constant 0 : index
    %c16 = constant 16 : index
    %c1_32 = constant 1 : i32
    %buf0 = memref.alloc() {sym_name = "scratch"}: memref<16xi32, 2>
    %buf1 = memref.alloc() {sym_name = "scratch_copy"}: memref<16xi32, 2>
    air.dma_memcpy_nd (%buf0[%c0][%c16][%c0], %ext0[%c0][%c16][%c0]) {id = 1 : i32} : (memref<16xi32, 2>, memref<16xi32>)
    affine.for %i = 0 to 16 {
        %0 = affine.load %buf0[%i] : memref<16xi32, 2>
        %1 = addi %0, %c1_32 : i32
        affine.store %1, %buf1[%i] : memref<16xi32, 2>
    }
    air.dma_memcpy_nd (%ext1[%c0][%c16][%c0], %buf1[%c0][%c16][%c0]) {id = 2 : i32} : (memref<16xi32>, memref<16xi32, 2>)
    memref.dealloc %buf1 : memref<16xi32, 2>
    memref.dealloc %buf0 : memref<16xi32, 2>
    air.herd_terminator
  }
  return
}

}

