// =============================================================================
// Test 00: Basic AIR hierarchy without air.rank
//
// Purpose: Establish baseline hierarchy (launch → segment → herd) before
//          introducing multi-device primitives.
//
// This test demonstrates:
// - air.launch: single device context
// - air.segment: tile rectangle with L2 memory
// - air.herd: array of compute units
// - Simple GEMV operation representative of attention/FFN computations
//
// RUN: air-opt %s | FileCheck %s
// =============================================================================

module @test_basic_hierarchy {

// CHECK-LABEL: func.func @gemv_single_device
func.func @gemv_single_device(
    %arg_in  : memref<1x2880xf16>,    // Input vector (like hidden state)
    %arg_w   : memref<2880x1440xf16>, // Weight matrix (like Q projection shard)
    %arg_out : memref<1x1440xf16>     // Output vector
) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  %c2880 = arith.constant 2880 : index
  %c1440 = arith.constant 1440 : index

  // CHECK: air.launch
  air.launch (%lx) in (%ls = %c1)
    args(%in_l = %arg_in, %w_l = %arg_w, %out_l = %arg_out) :
    memref<1x2880xf16>, memref<2880x1440xf16>, memref<1x1440xf16>
  {
    // CHECK: air.segment
    air.segment @gemv_seg
      args(%in_s = %in_l, %w_s = %w_l, %out_s = %out_l) :
      memref<1x2880xf16>, memref<2880x1440xf16>, memref<1x1440xf16>
    {
      // Allocate L2 buffer for tiled output accumulation
      %c90 = arith.constant 90 : index
      %output_tiles = memref.alloc() : memref<16x1x90xf32, 2>

      // CHECK: air.herd
      // Herd: 4x4 = 16 compute units
      // Each CU computes a 90-element tile of the output (1440 / 16 = 90)
      air.herd @gemv_herd tile (%tx, %ty) in (%sx = %c4, %sy = %c4)
        args(%in_h = %in_s, %w_h = %w_s, %out_h = %output_tiles) :
        memref<1x2880xf16>, memref<2880x1440xf16>, memref<16x1x90xf32, 2>
      {
        %c128 = arith.constant 128 : index

        // Compute which output tile this CU owns
        %cu_id = arith.addi (arith.muli %ty, %c4 : index), %tx : index
        %tile_size = arith.constant 90 : index
        %out_base = arith.muli %cu_id, %tile_size : index

        // LDS accumulator for this tile (f32 for precision)
        %acc = memref.alloc() : memref<1x90xf32, 3>

        // Initialize accumulator to zero
        air.execute {
          linalg.fill ins(%cst_zero : f32) outs(%acc : memref<1x90xf32, 3>)
        } where {
          %cst_zero = arith.constant 0.0 : f32
        }

        // Tiled GEMV: input[1x2880] × weight[2880x1440][out_base:out_base+90]
        // Loop over K dimension (2880) in tiles of 128
        air.execute {
          scf.for %ki = %c0 to %c2880 step %c128 {
            // Allocate LDS for input tile and weight tile
            %in_tile = memref.alloc() : memref<1x128xf16, 3>
            %w_tile  = memref.alloc() : memref<128x90xf16, 3>

            // DMA input tile from DDR → LDS
            air.dma_memcpy_nd (
              %in_tile[] [] [],
              %in_h[%c0, %ki] [%c1, %c128] [%c2880, %c1]
            ) : memref<1x128xf16, 3>, memref<1x2880xf16>

            // DMA weight tile from DDR → LDS
            // weight[ki:ki+128, out_base:out_base+90]
            air.dma_memcpy_nd (
              %w_tile[] [] [],
              %w_h[%ki, %out_base] [%c128, %tile_size] [%c1440, %c1]
            ) : memref<128x90xf16, 3>, memref<2880x1440xf16>

            // Compute: acc += in_tile × w_tile
            linalg.matmul
              ins(%in_tile, %w_tile : memref<1x128xf16, 3>, memref<128x90xf16, 3>)
              outs(%acc : memref<1x90xf32, 3>)

            memref.dealloc %in_tile : memref<1x128xf16, 3>
            memref.dealloc %w_tile  : memref<128x90xf16, 3>
          }
        }

        // Store accumulated result to L2 output buffer
        air.dma_memcpy_nd (
          %out_h[%cu_id, %c0, %c0] [%c1, %c1, %tile_size] [%tile_size, %tile_size, %c1],
          %acc[] [] []
        ) : memref<16x1x90xf32, 2>, memref<1x90xf32, 3>

        memref.dealloc %acc : memref<1x90xf32, 3>

        air.herd_terminator
      } // end @gemv_herd

      // Coalesce L2 output tiles and downcast f32 → f16
      air.execute {
        scf.for %tile_id = %c0 to arith.constant 16 : index step %c1 {
          %tile_off = arith.muli %tile_id, arith.constant 90 : index : index
          linalg.generic {
            indexing_maps = [
              affine_map<(d0, d1) -> (d0, 0, d1)>,  // src in L2
              affine_map<(d0, d1) -> (d0, d1)>      // dst in DDR
            ],
            iterator_types = ["parallel", "parallel"]
          }
          ins(%output_tiles : memref<16x1x90xf32, 2>)
          outs(%out_s[%c0, %tile_off] [%c1, arith.constant 90 : index] [arith.constant 90 : index, %c1]
               : memref<1x1440xf16>)
          {
            ^bb0(%in_val : f32, %out_val : f16):
              %truncated = arith.truncf %in_val : f32 to f16
              linalg.yield %truncated : f16
          }
        }
      }

      memref.dealloc %output_tiles : memref<16x1x90xf32, 2>

      air.segment_terminator
    } // end @gemv_seg

    air.launch_terminator
  } // end air.launch

  return
}

} // end module
