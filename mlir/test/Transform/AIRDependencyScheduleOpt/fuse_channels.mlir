//===- fuse_channels.mlir --------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: MIT
//
//===----------------------------------------------------------------------===//

// RUN: air-opt %s -air-fuse-channels="aggressive-mode=true" --split-input-file | FileCheck %s
// RUN: air-opt %s -air-fuse-channels="aggressive-mode=false" --split-input-file | FileCheck %s --check-prefix=FUSELOOP

// Have multiple channel put-get pairs share the same symbolic channels.
// CHECK-LABEL: func0
// CHECK: air.launch
// CHECK: air.channel.put @channel_0
// CHECK: air.channel.put @channel_0
// CHECK: air.segment
// CHECK: air.channel.get @channel_0
// CHECK: air.channel.get @channel_0
// CHECK: air.segment_terminator
// CHECK: air.launch_terminator
// FUSELOOP-LABEL: func0
// FUSELOOP: air.launch
// FUSELOOP: air.channel.put @channel_0
// FUSELOOP: air.channel.put @channel_1
// FUSELOOP: air.segment
// FUSELOOP: air.channel.get @channel_0
// FUSELOOP: air.channel.get @channel_1
// FUSELOOP: air.segment_terminator
// FUSELOOP: air.launch_terminator

module {
  air.channel @channel_0 [1, 1]
  air.channel @channel_1 [1, 1]
  func.func @func0(){
    %c1 = arith.constant 1 : index
    air.launch (%arg3, %arg4) in (%arg5=%c1, %arg6=%c1) {
      %alloc_0 = memref.alloc() : memref<4x4xi32>
      %alloc_1 = memref.alloc() : memref<4x4xi32>
      air.channel.put @channel_0[%arg3, %arg4] (%alloc_0[] [] []) : (memref<4x4xi32>)
      air.channel.put @channel_1[%arg3, %arg4] (%alloc_1[] [] []) : (memref<4x4xi32>)
      air.segment {
        %c2 = arith.constant 2 : index
        %alloc_2 = memref.alloc() : memref<4x4xi32, 1>
        %alloc_3 = memref.alloc() : memref<4x4xi32, 1>
        air.channel.get @channel_0[] (%alloc_2[] [] []) : (memref<4x4xi32, 1>)
        air.channel.get @channel_1[] (%alloc_3[] [] []) : (memref<4x4xi32, 1>)
        air.herd @herd_0 tile (%arg12, %arg13) in (%arg14=%c2, %arg15=%c2) {
          air.herd_terminator
        }
        memref.dealloc %alloc_2 : memref<4x4xi32, 1>
        memref.dealloc %alloc_3 : memref<4x4xi32, 1>
        air.segment_terminator
      }
      memref.dealloc %alloc_0 : memref<4x4xi32>
      memref.dealloc %alloc_1 : memref<4x4xi32>
      air.launch_terminator
    }
    return
  }
}

// -----

// CHECK-LABEL: func1
// CHECK: air.launch
// CHECK: air.segment
// CHECK: air.channel.put @channel_2
// CHECK: scf.for
// CHECK: air.channel.put @channel_0
// CHECK: air.channel.put @channel_0
// CHECK: air.herd
// CHECK: air.channel.get @channel_2
// CHECK: scf.for
// CHECK: air.channel.get @channel_0
// CHECK: air.channel.get @channel_0
// CHECK: air.herd_terminator
// CHECK: air.segment_terminator
// CHECK: air.launch_terminator
// FUSELOOP-LABEL: func1
// FUSELOOP: air.launch
// FUSELOOP: air.segment
// FUSELOOP: air.channel.put @channel_2
// FUSELOOP: scf.for
// FUSELOOP: air.channel.put @channel_0
// FUSELOOP: air.channel.put @channel_1
// FUSELOOP: air.herd
// FUSELOOP: air.channel.get @channel_2
// FUSELOOP: scf.for
// FUSELOOP: air.channel.get @channel_0
// FUSELOOP: air.channel.get @channel_1
// FUSELOOP: air.herd_terminator
// FUSELOOP: air.segment_terminator
// FUSELOOP: air.launch_terminator

module {
  air.channel @channel_0 [1, 1]
  air.channel @channel_1 [1, 1]
  air.channel @channel_2 [1, 1]
  func.func @func1(){
    %c1 = arith.constant 1 : index
    air.launch (%arg3, %arg4) in (%arg5=%c1, %arg6=%c1) {
      air.segment {
        %c0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1_1 = arith.constant 1 : index
        %alloc_1 = memref.alloc() : memref<4x4xi32, 1>
        %alloc_2 = memref.alloc() : memref<4x4xi32, 1>
        %alloc_3 = memref.alloc() : memref<4x4xi32, 1>
        air.channel.put @channel_2[] (%alloc_1[] [] []) : (memref<4x4xi32, 1>)
        scf.for %arg0 = %c0 to %c2 step %c1_1 {
          air.channel.put @channel_0[] (%alloc_2[] [] []) : (memref<4x4xi32, 1>)
        }
        scf.for %arg0 = %c0 to %c2 step %c1_1 {
          air.channel.put @channel_1[] (%alloc_3[] [] []) : (memref<4x4xi32, 1>)
        }
        memref.dealloc %alloc_1 : memref<4x4xi32, 1>
        memref.dealloc %alloc_2 : memref<4x4xi32, 1>
        memref.dealloc %alloc_3 : memref<4x4xi32, 1>
        air.herd @herd_0 tile (%arg12, %arg13) in (%arg14=%c2, %arg15=%c2) {
          %c0_2 = arith.constant 0 : index
          %c2_2 = arith.constant 2 : index
          %c1_2 = arith.constant 1 : index
          %alloc_4 = memref.alloc() : memref<4x4xi32, 2>
          air.channel.get @channel_2[] (%alloc_4[] [] []) : (memref<4x4xi32, 2>)
          scf.for %arg0 = %c0_2 to %c2_2 step %c1_2 {
            %alloc_5 = memref.alloc() : memref<4x4xi32, 2>
            %alloc_6 = memref.alloc() : memref<4x4xi32, 2>
            air.channel.get @channel_0[] (%alloc_5[] [] []) : (memref<4x4xi32, 2>)
            air.channel.get @channel_1[] (%alloc_6[] [] []) : (memref<4x4xi32, 2>)
            memref.dealloc %alloc_5 : memref<4x4xi32, 2>
            memref.dealloc %alloc_6 : memref<4x4xi32, 2>
          }
          memref.dealloc %alloc_4 : memref<4x4xi32, 2>
          air.herd_terminator
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }
}

// -----

// CHECK-LABEL: func2
// CHECK: air.launch
// CHECK: air.channel.put{{.*}}@channel_0
// CHECK: air.channel.put{{.*}}@channel_0
// CHECK: air.segment
// CHECK: air.channel.get{{.*}}@channel_0
// CHECK: air.channel.get{{.*}}@channel_0
// CHECK: air.segment_terminator
// CHECK: air.launch_terminator
// FUSELOOP-LABEL: func2
// FUSELOOP: air.launch
// FUSELOOP: air.channel.put{{.*}}@channel_0
// FUSELOOP: air.channel.put{{.*}}@channel_1
// FUSELOOP: air.segment
// FUSELOOP: air.channel.get{{.*}}@channel_0
// FUSELOOP: air.channel.get{{.*}}@channel_1
// FUSELOOP: air.segment_terminator
// FUSELOOP: air.launch_terminator

module {
  air.channel @channel_0 [1, 1]
  air.channel @channel_1 [1, 1]
  func.func @func2() {
    %c1 = arith.constant 1 : index
    %0 = air.launch async (%arg0, %arg1) in (%arg2=%c1, %arg3=%c1) attributes {id = 1 : i32} {
      %async_token, %results = air.execute -> (memref<4x4xi32>) {
        %alloc = memref.alloc() : memref<4x4xi32>
        air.execute_terminator %alloc : memref<4x4xi32>
      }
      %async_token_0, %results_1 = air.execute -> (memref<4x4xi32>) {
        %alloc = memref.alloc() : memref<4x4xi32>
        air.execute_terminator %alloc : memref<4x4xi32>
      }
      %1 = air.channel.put async [%async_token]  @channel_0[%arg0, %arg1] (%results[] [] []) {id = 1 : i32} : (memref<4x4xi32>)
      %2 = air.channel.put async [%async_token_0]  @channel_1[%arg0, %arg1] (%results_1[] [] []) {id = 2 : i32} : (memref<4x4xi32>)
      %3 = air.segment async  attributes {id = 2 : i32} {
        %c2 = arith.constant 2 : index
        %async_token_4, %results_5 = air.execute -> (memref<4x4xi32, 1>) {
          %alloc = memref.alloc() : memref<4x4xi32, 1>
          air.execute_terminator %alloc : memref<4x4xi32, 1>
        }
        %async_token_6, %results_7 = air.execute -> (memref<4x4xi32, 1>) {
          %alloc = memref.alloc() : memref<4x4xi32, 1>
          air.execute_terminator %alloc : memref<4x4xi32, 1>
        }
        %4 = air.channel.get async [%async_token_4]  @channel_0[] (%results_5[] [] []) {id = 3 : i32} : (memref<4x4xi32, 1>)
        %5 = air.channel.get async [%async_token_6]  @channel_1[] (%results_7[] [] []) {id = 4 : i32} : (memref<4x4xi32, 1>)
        %6 = air.herd @herd_0 async  tile (%arg4, %arg5) in (%arg6=%c2, %arg7=%c2) attributes {id = 3 : i32} 
        %async_token_8 = air.execute [%4] {
          memref.dealloc %results_5 : memref<4x4xi32, 1>
        }
        %async_token_9 = air.execute [%5] {
          memref.dealloc %results_7 : memref<4x4xi32, 1>
        }
        air.segment_terminator
      }
      %async_token_2 = air.execute [%1] {
        memref.dealloc %results : memref<4x4xi32>
      }
      %async_token_3 = air.execute [%2] {
        memref.dealloc %results_1 : memref<4x4xi32>
      }
      air.launch_terminator
    }
    return
  }
}

// -----

// CHECK-LABEL: func3
// CHECK: air.launch
// CHECK: air.channel.put{{.*}}@channel_1
// CHECK: air.channel.put{{.*}}@channel_1
// CHECK: air.segment @segment_0
// CHECK: air.herd @herd_0
// CHECK: air.channel.get{{.*}}@channel_1
// CHECK: air.channel.get{{.*}}@channel_1
// CHECK: air.herd_terminator
// CHECK: air.segment_terminator
// CHECK: air.launch_terminator
// FUSELOOP-LABEL: func3
// FUSELOOP: air.launch
// FUSELOOP: air.channel.put{{.*}}@channel_0
// FUSELOOP: air.channel.put{{.*}}@channel_1
// FUSELOOP: air.segment @segment_0
// FUSELOOP: air.herd @herd_0
// FUSELOOP: air.channel.get{{.*}}@channel_0
// FUSELOOP: air.channel.get{{.*}}@channel_1
// FUSELOOP: air.herd_terminator
// FUSELOOP: air.segment_terminator
// FUSELOOP: air.launch_terminator

#map = affine_map<()[s0] -> (s0 * 32)>
module {
  air.channel @channel_1 [2, 2]
  air.channel @channel_0 [2, 2]
  func.func @func3(%arg0: memref<64x64xi32>, %arg1: memref<64x64xi32>, %arg2: memref<64x64xi32>) {
    %c1 = arith.constant 1 : index
    air.launch (%arg3, %arg4) in (%arg5=%c1, %arg6=%c1) args(%arg7=%arg0, %arg8=%arg1) : memref<64x64xi32>, memref<64x64xi32> {
      %c32 = arith.constant 32 : index
      %c64 = arith.constant 64 : index
      %c1_0 = arith.constant 1 : index
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %0 = air.wait_all async 
      %1 = scf.parallel (%arg9, %arg10) = (%c0, %c0) to (%c2, %c2) step (%c1_0, %c1_0) init (%0) -> !air.async.token {
        %4 = affine.apply #map()[%arg9]
        %5 = arith.addi %arg3, %4 : index
        scf.for %arg11 = %c0 to %c64 step %c32 {
          air.channel.put  @channel_0[%arg9, %arg10] (%arg7[%5, %arg11] [%c32, %c32] [%c64, %c1_0]) : (memref<64x64xi32>)
        }
        %6 = air.wait_all async 
        scf.reduce(%6 : !air.async.token) {
        ^bb0(%arg11: !air.async.token, %arg12: !air.async.token):
          %7 = air.wait_all async [%arg11, %arg12] 
          scf.reduce.return %7 : !air.async.token
        }
      }
      %2 = air.wait_all async 
      %3 = scf.parallel (%arg9, %arg10) = (%c0, %c0) to (%c2, %c2) step (%c1_0, %c1_0) init (%2) -> !air.async.token {
        %4 = affine.apply #map()[%arg10]
        %5 = arith.addi %arg4, %4 : index
        scf.for %arg11 = %c0 to %c64 step %c32 {
          air.channel.put  @channel_1[%arg9, %arg10] (%arg8[%arg11, %5] [%c32, %c32] [%c64, %c1_0]) : (memref<64x64xi32>)
        }
        %6 = air.wait_all async 
        scf.reduce(%6 : !air.async.token) {
        ^bb0(%arg11: !air.async.token, %arg12: !air.async.token):
          %7 = air.wait_all async [%arg11, %arg12] 
          scf.reduce.return %7 : !air.async.token
        }
      }
      air.segment @segment_0  {
        %c2_1 = arith.constant 2 : index
        air.herd @herd_0  tile (%arg9, %arg10) in (%arg11=%c2_1, %arg12=%c2_1) {
          %c0_2 = arith.constant 0 : index
          %c64_3 = arith.constant 64 : index
          %c32_4 = arith.constant 32 : index
          scf.for %arg13 = %c0_2 to %c64_3 step %c32_4 {
            %alloc = memref.alloc() : memref<32x32xi32, 2>
            %alloc_5 = memref.alloc() : memref<32x32xi32, 2>
            air.channel.get  @channel_0[%arg9, %arg10] (%alloc[] [] []) : (memref<32x32xi32, 2>)
            air.channel.get  @channel_1[%arg9, %arg10] (%alloc_5[] [] []) : (memref<32x32xi32, 2>)
            memref.dealloc %alloc : memref<32x32xi32, 2>
            memref.dealloc %alloc_5 : memref<32x32xi32, 2>
          }
          air.herd_terminator
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    return
  }
}

// -----

// CHECK-LABEL: func4
// CHECK: air.launch
// CHECK: air.segment @segment_0
// CHECK: air.herd @herd_0
// CHECK: air.channel.get{{.*}}@channel_2
// CHECK: air.channel.get{{.*}}@channel_3
// CHECK: air.herd_terminator
// CHECK: scf.for
// CHECK-NEXT: scf.for
// CHECK: air.channel.put{{.*}}@channel_2
// CHECK: scf.for
// CHECK-NEXT: scf.for
// CHECK: air.channel.put{{.*}}@channel_3
// CHECK: air.herd @herd_0
// CHECK: air.channel.get{{.*}}@channel_2
// CHECK: air.channel.get{{.*}}@channel_3
// CHECK: air.herd_terminator
// CHECK: air.segment_terminator
// CHECK: air.launch_terminator
// FUSELOOP-LABEL: func4
// FUSELOOP: air.launch
// FUSELOOP: air.segment @segment_0
// FUSELOOP: air.herd @herd_0
// FUSELOOP: air.channel.get{{.*}}@channel_2
// FUSELOOP: air.channel.get{{.*}}@channel_3
// FUSELOOP: air.herd_terminator
// FUSELOOP: scf.for
// FUSELOOP-NEXT: scf.for
// FUSELOOP: air.channel.put{{.*}}@channel_2
// FUSELOOP: scf.for
// FUSELOOP-NEXT: scf.for
// FUSELOOP: air.channel.put{{.*}}@channel_3
// FUSELOOP: air.herd @herd_0
// FUSELOOP: air.channel.get{{.*}}@channel_2
// FUSELOOP: air.channel.get{{.*}}@channel_3
// FUSELOOP: air.herd_terminator
// FUSELOOP: air.segment_terminator
// FUSELOOP: air.launch_terminator

#map = affine_map<(d0) -> (d0 * 8)>
#set = affine_set<()[s0, s1] : (s0 == 0, s1 >= 0, -s1 + 1 >= 0)>
module {
  air.channel @channel_3 [1, 1] {broadcast_shape = [1, 2]}
  air.channel @channel_2 [1, 1] {broadcast_shape = [1, 2]}
  air.channel @channel_1 [1, 1] {broadcast_shape = [1, 2]}
  air.channel @channel_0 [1, 1] {broadcast_shape = [1, 2]}
  func.func @func4(%arg0: memref<1024x2048xi32>, %arg1: memref<2048x512xi32>, %arg2: memref<1024x512xi32>) {
    %c16 = arith.constant 16 : index
    %c8 = arith.constant 8 : index
    %async_token, %results = air.execute -> (memref<1x1x4x8x4x8xi32, 2>) {
      %alloc = memref.alloc() : memref<1x1x4x8x4x8xi32, 2>
      air.execute_terminator %alloc : memref<1x1x4x8x4x8xi32, 2>
    }
    %async_token_0, %results_1 = air.execute -> (memref<2x1x32x256xi32, 1>) {
      %alloc = memref.alloc() : memref<2x1x32x256xi32, 1>
      air.execute_terminator %alloc : memref<2x1x32x256xi32, 1>
    }
    %0 = air.launch async [%async_token, %async_token_0] (%arg3, %arg4) in (%arg5=%c16, %arg6=%c8) args(%arg7=%results_1, %arg8=%results) : memref<2x1x32x256xi32, 1>, memref<1x1x4x8x4x8xi32, 2> attributes {id = 1 : i32} {
      %1 = air.segment @segment_0 async  args(%arg9=%arg7, %arg10=%arg8) : memref<2x1x32x256xi32, 1>, memref<1x1x4x8x4x8xi32, 2> attributes {id = 2 : i32} {
        %c8192 = arith.constant 8192 : index
        %c4 = arith.constant 4 : index
        %c1024 = arith.constant 1024 : index
        %c32 = arith.constant 32 : index
        %c256 = arith.constant 256 : index
        %c0 = arith.constant 0 : index
        %c2 = arith.constant 2 : index
        %c1 = arith.constant 1 : index
        %c8_4 = arith.constant 8 : index
        %2 = air.wait_all async 
        %3 = scf.for %arg11 = %c0 to %c32 step %c4 iter_args(%arg12 = %2) -> (!air.async.token) {
          %async_token_5, %results_6 = air.execute [%arg12] -> (index) {
            %13 = affine.apply #map(%arg11)
            air.execute_terminator %13 : index
          }
          %12 = air.channel.put async [%async_token_5]  @channel_0[] (%arg9[%c0, %c0, %c0, %c0, %c0, %results_6] [%c1, %c1, %c4, %c8_4, %c4, %c8_4] [%c8192, %c8192, %c8_4, %c1024, %c256, %c1]) {id = 9 : i32} : (memref<2x1x32x256xi32, 1>)
          scf.yield %12 : !air.async.token
        }
        %4 = air.wait_all async 
        %5 = scf.for %arg11 = %c0 to %c32 step %c4 iter_args(%arg12 = %4) -> (!air.async.token) {
          %async_token_5, %results_6 = air.execute [%arg12] -> (index) {
            %13 = affine.apply #map(%arg11)
            air.execute_terminator %13 : index
          }
          %12 = air.channel.put async [%async_token_5]  @channel_1[] (%arg9[%c0, %c0, %c0, %c0, %c0, %results_6] [%c1, %c1, %c4, %c8_4, %c4, %c8_4] [%c8192, %c8192, %c8_4, %c1024, %c256, %c1]) {id = 10 : i32} : (memref<2x1x32x256xi32, 1>)
          scf.yield %12 : !air.async.token
        }
        %6 = air.herd @herd_0 async  tile (%arg11, %arg12) in (%arg13=%c2, %arg14=%c2) args(%arg15=%arg10) : memref<1x1x4x8x4x8xi32, 2> attributes {id = 3 : i32} {
          %c0_5 = arith.constant 0 : index
          %c32_6 = arith.constant 32 : index
          %c4_7 = arith.constant 4 : index
          %12 = air.wait_all async 
          %13 = scf.for %arg16 = %c0_5 to %c32_6 step %c4_7 iter_args(%arg17 = %12) -> (!air.async.token) {
            %14 = affine.if #set()[%arg11, %arg12] -> !air.async.token {
              %15 = air.channel.get async [%arg17]  @channel_0[%arg11, %arg12] (%arg15[] [] []) {id = 13 : i32} : (memref<1x1x4x8x4x8xi32, 2>)
              affine.yield %15 : !air.async.token
            } else {
              %15 = air.channel.get async [%arg17]  @channel_1[%arg11, %arg12] (%arg15[] [] []) {id = 14 : i32} : (memref<1x1x4x8x4x8xi32, 2>)
              affine.yield %15 : !air.async.token
            }
            scf.yield %14 : !air.async.token
          }
          air.herd_terminator
        }
        %7 = air.wait_all async 
        %8 = scf.for %arg11 = %c1 to %c8_4 step %c1 iter_args(%arg12 = %7) -> (!air.async.token) {
          %12 = scf.for %arg13 = %c0 to %c32 step %c4 iter_args(%arg14 = %arg12) -> (!air.async.token) {
            %async_token_5, %results_6 = air.execute [%arg14] -> (index) {
              %14 = affine.apply #map(%arg13)
              air.execute_terminator %14 : index
            }
            %13 = air.channel.put async [%async_token_5]  @channel_2[] (%arg9[%c0, %c0, %c0, %c0, %c0, %results_6] [%c1, %c1, %c4, %c8_4, %c4, %c8_4] [%c8192, %c8192, %c8_4, %c1024, %c256, %c1]) {id = 19 : i32} : (memref<2x1x32x256xi32, 1>)
            scf.yield %13 : !air.async.token
          }
          scf.yield %12 : !air.async.token
        }
        %9 = air.wait_all async 
        %10 = scf.for %arg11 = %c1 to %c8_4 step %c1 iter_args(%arg12 = %9) -> (!air.async.token) {
          %12 = scf.for %arg13 = %c0 to %c32 step %c4 iter_args(%arg14 = %arg12) -> (!air.async.token) {
            %async_token_5, %results_6 = air.execute [%arg14] -> (index) {
              %14 = affine.apply #map(%arg13)
              air.execute_terminator %14 : index
            }
            %13 = air.channel.put async [%async_token_5]  @channel_3[] (%arg9[%c0, %c0, %c0, %c0, %c0, %results_6] [%c1, %c1, %c4, %c8_4, %c4, %c8_4] [%c8192, %c8192, %c8_4, %c1024, %c256, %c1]) {id = 20 : i32} : (memref<2x1x32x256xi32, 1>)
            scf.yield %13 : !air.async.token
          }
          scf.yield %12 : !air.async.token
        }
        %11 = air.herd @herd_0 async  tile (%arg11, %arg12) in (%arg13=%c2, %arg14=%c2) args(%arg15=%arg10) : memref<1x1x4x8x4x8xi32, 2> attributes {id = 4 : i32} {
          %c4_5 = arith.constant 4 : index
          %c32_6 = arith.constant 32 : index
          %c0_7 = arith.constant 0 : index
          %c1_8 = arith.constant 1 : index
          %c8_9 = arith.constant 8 : index
          scf.for %arg16 = %c1_8 to %c8_9 step %c1_8 {
            %12 = air.wait_all async 
            %13 = scf.for %arg17 = %c0_7 to %c32_6 step %c4_5 iter_args(%arg18 = %12) -> (!air.async.token) {
              %14 = affine.if #set()[%arg11, %arg12] -> !air.async.token {
                %15 = air.channel.get async [%arg18]  @channel_2[%arg11, %arg12] (%arg15[] [] []) {id = 23 : i32} : (memref<1x1x4x8x4x8xi32, 2>)
                affine.yield %15 : !air.async.token
              } else {
                %15 = air.channel.get async [%arg18]  @channel_3[%arg11, %arg12] (%arg15[] [] []) {id = 24 : i32} : (memref<1x1x4x8x4x8xi32, 2>)
                affine.yield %15 : !air.async.token
              }
              scf.yield %14 : !air.async.token
            }
          }
          air.herd_terminator
        }
        air.segment_terminator
      }
      air.launch_terminator
    }
    %async_token_2 = air.execute [%0] {
      memref.dealloc %results_1 : memref<2x1x32x256xi32, 1>
    }
    %async_token_3 = air.execute [%0] {
      memref.dealloc %results : memref<1x1x4x8x4x8xi32, 2>
    }
    return
  }
}
