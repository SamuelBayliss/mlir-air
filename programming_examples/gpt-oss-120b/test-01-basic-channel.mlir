// =============================================================================
// Test 01: Basic channel communication without air.rank
//
// Purpose: Demonstrate air.channel for communication between segments before
//          introducing multi-device concepts.
//
// This test demonstrates:
// - air.channel declaration
// - air.channel.put and air.channel.get for data transfer
// - Two segments communicating within a single air.launch
// - Pattern: producer segment → channel → consumer segment
//
// This establishes the channel communication pattern that will be used for
// cross-rank communication in later tests.
//
// RUN: air-opt %s | FileCheck %s
// =============================================================================

module @test_basic_channel {

// CHECK-LABEL: func.func @two_segment_pipeline
func.func @two_segment_pipeline(
    %arg_in  : memref<1x2880xf16>,    // Input data
    %arg_out : memref<1x2880xf16>     // Output data
) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2880 = arith.constant 2880 : index

  // CHECK: air.channel @pipe
  air.channel @pipe [1, 1]

  // CHECK: air.launch
  air.launch (%lx) in (%ls = %c1)
    args(%in_l = %arg_in, %out_l = %arg_out) :
    memref<1x2880xf16>, memref<1x2880xf16>
  {
    // =========================================================================
    // Producer segment: reads input, processes, sends via channel
    // =========================================================================
    // CHECK: air.segment @producer
    air.segment @producer
      args(%in_s = %in_l) : memref<1x2880xf16>
    {
      // Allocate L2 buffer for intermediate result
      %intermediate = memref.alloc() : memref<1x2880xf16, 2>

      // Simple processing (element-wise operation)
      air.execute {
        linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1) -> (d0, d1)>,  // input
            affine_map<(d0, d1) -> (d0, d1)>   // output
          ],
          iterator_types = ["parallel", "parallel"]
        }
        ins(%in_s : memref<1x2880xf16>)
        outs(%intermediate : memref<1x2880xf16, 2>)
        {
          ^bb0(%in_val : f16, %out_val : f16):
            // Simple transformation: multiply by 2
            %c2 = arith.constant 2.0 : f16
            %result = arith.mulf %in_val, %c2 : f16
            linalg.yield %result : f16
        }
      }

      // CHECK: air.channel.put @pipe
      // Send intermediate result through channel
      air.channel.put @pipe[%c0, %c0]
        (%intermediate[%c0, %c0] [%c1, %c2880] [%c2880, %c1])
        : memref<1x2880xf16, 2>

      memref.dealloc %intermediate : memref<1x2880xf16, 2>

      air.segment_terminator
    } // end @producer

    // =========================================================================
    // Consumer segment: receives via channel, processes, writes output
    // =========================================================================
    // CHECK: air.segment @consumer
    air.segment @consumer
      args(%out_s = %out_l) : memref<1x2880xf16>
    {
      // Allocate L2 buffer to receive data
      %received = memref.alloc() : memref<1x2880xf16, 2>

      // CHECK: air.channel.get @pipe
      // Receive data from channel
      air.channel.get @pipe[%c0, %c0]
        (%received[%c0, %c0] [%c1, %c2880] [%c2880, %c1])
        : memref<1x2880xf16, 2>

      // Simple processing and write to output
      air.execute {
        linalg.generic {
          indexing_maps = [
            affine_map<(d0, d1) -> (d0, d1)>,  // input
            affine_map<(d0, d1) -> (d0, d1)>   // output
          ],
          iterator_types = ["parallel", "parallel"]
        }
        ins(%received : memref<1x2880xf16, 2>)
        outs(%out_s : memref<1x2880xf16>)
        {
          ^bb0(%in_val : f16, %out_val : f16):
            // Simple transformation: add 1
            %c1_f16 = arith.constant 1.0 : f16
            %result = arith.addf %in_val, %c1_f16 : f16
            linalg.yield %result : f16
        }
      }

      memref.dealloc %received : memref<1x2880xf16, 2>

      air.segment_terminator
    } // end @consumer

    air.launch_terminator
  } // end air.launch

  return
}

// =============================================================================
// Test with async tokens for ordered execution
// =============================================================================

// CHECK-LABEL: func.func @two_segment_pipeline_async
func.func @two_segment_pipeline_async(
    %arg_in  : memref<1x2880xf16>,
    %arg_out : memref<1x2880xf16>
) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c2880 = arith.constant 2880 : index

  air.channel @pipe_async [1, 1]

  // CHECK: air.launch async
  %t0 = air.launch async (%lx) in (%ls = %c1)
    args(%in_l = %arg_in, %out_l = %arg_out) :
    memref<1x2880xf16>, memref<1x2880xf16>
  {
    // Producer segment with async token
    %t_producer = air.segment async @producer_async
      args(%in_s = %in_l) : memref<1x2880xf16>
    {
      %intermediate = memref.alloc() : memref<1x2880xf16, 2>

      %t_compute = air.execute {
        linalg.fill ins(%cst : f16) outs(%intermediate : memref<1x2880xf16, 2>)
      } where {
        %cst = arith.constant 1.0 : f16
      }

      %t_put = air.channel.put async [%t_compute] @pipe_async[%c0, %c0]
        (%intermediate[%c0, %c0] [%c1, %c2880] [%c2880, %c1])
        : memref<1x2880xf16, 2>

      memref.dealloc %intermediate : memref<1x2880xf16, 2>
      air.segment_terminator
    }

    // Consumer segment depends on producer via channel
    %t_consumer = air.segment async @consumer_async
      args(%out_s = %out_l) : memref<1x2880xf16>
    {
      %received = memref.alloc() : memref<1x2880xf16, 2>

      %t_get = air.channel.get async @pipe_async[%c0, %c0]
        (%received[%c0, %c0] [%c1, %c2880] [%c2880, %c1])
        : memref<1x2880xf16, 2>

      %t_write = air.execute [%t_get] {
        linalg.copy ins(%received : memref<1x2880xf16, 2>)
                    outs(%out_s : memref<1x2880xf16>)
      }

      memref.dealloc %received : memref<1x2880xf16, 2>
      air.segment_terminator
    }

    air.launch_terminator
  }

  air.wait_all [%t0]
  return
}

} // end module
