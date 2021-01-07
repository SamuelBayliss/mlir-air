
#include <cstdio>
#include <cassert>
#include <stdlib.h>
#include <fcntl.h>
#include <sys/mman.h>

#include <xaiengine.h>

//#include "acdc_agent.h"
#include "acdc_queue.h"
#include "hsa_defs.h"

#define SHMEM_BASE 0x020100000000LL

#define XAIE_NUM_ROWS            8
#define XAIE_NUM_COLS           50
#define XAIE_ADDR_ARRAY_OFF     0x800

#define HIGH_ADDR(addr)	((addr & 0xffffffff00000000) >> 32)
#define LOW_ADDR(addr)	(addr & 0x00000000ffffffff)

namespace {

XAieGbl_Config *AieConfigPtr;	                          /**< AIE configuration pointer */
XAieGbl AieInst;	                                      /**< AIE global instance */
XAieGbl_HwCfg AieConfig;                                /**< AIE HW configuration instance */
XAieGbl_Tile TileInst[XAIE_NUM_COLS][XAIE_NUM_ROWS+1];  /**< Instantiates AIE array of [XAIE_NUM_COLS] x [XAIE_NUM_ROWS] */
XAieDma_Tile TileDMAInst[XAIE_NUM_COLS][XAIE_NUM_ROWS+1];

#include "aie_inc.cpp"

}

hsa_status_t queue_create(uint32_t size, uint32_t type, queue_t **queue)
{
  int fd = open("/dev/mem", O_RDWR | O_SYNC);
  if (fd == -1)
    return HSA_STATUS_ERROR_INVALID_QUEUE_CREATION;

  uint64_t *bram_ptr = (uint64_t *)mmap(NULL, 0x8000, PROT_READ|PROT_WRITE, MAP_SHARED, fd, SHMEM_BASE);
  // I have no idea if this does anything
  __clear_cache((void*)bram_ptr, (void*)(bram_ptr+0x1000));
  //for (int i=0; i<20; i++)
  //  printf("%p %llx\n", &bram_ptr[i], bram_ptr[i]);

  printf("Opened shared memory paddr: %p vaddr: %p\n", SHMEM_BASE, bram_ptr);
  uint64_t q_paddr = bram_ptr[0];
  uint64_t q_offset = q_paddr - SHMEM_BASE;
  queue_t *q = (queue_t*)( ((size_t)bram_ptr) + q_offset );
  printf("Queue location at paddr: %p vaddr: %p\n", bram_ptr[0], q);

  if (q->id !=  0xacdc) {
    printf("%s error invalid id %x\n", __func__, q->id);
    return HSA_STATUS_ERROR_INVALID_QUEUE_CREATION;
  }

  if (q->size != size) {
    printf("%s error size mismatch %d\n", __func__, q->size);
    return HSA_STATUS_ERROR_INVALID_QUEUE_CREATION;
  }

  if (q->type != type) {
    printf("%s error type mismatch %d\n", __func__, q->type);
    return HSA_STATUS_ERROR_INVALID_QUEUE_CREATION;
  }

  uint64_t base_address_offset = q->base_address - SHMEM_BASE;
  q->base_address_vaddr = ((size_t)bram_ptr) + base_address_offset;

  *queue = q;
  return HSA_STATUS_SUCCESS;
}

void setup_aie(void)
{
  auto col = 7;

  size_t aie_base = XAIE_ADDR_ARRAY_OFF << 14;
  XAIEGBL_HWCFG_SET_CONFIG((&AieConfig), XAIE_NUM_ROWS, XAIE_NUM_COLS, XAIE_ADDR_ARRAY_OFF);
  XAieGbl_HwInit(&AieConfig);
  AieConfigPtr = XAieGbl_LookupConfig(XPAR_AIE_DEVICE_ID);
  XAieGbl_CfgInitialize(&AieInst, &TileInst[0][0], AieConfigPtr);

  // reset cores and locks
  for (int i = 1; i <= XAIE_NUM_ROWS; i++) {
    for (int j = 0; j < XAIE_NUM_COLS; j++) {
      XAieTile_CoreControl(&(TileInst[j][i]), XAIE_DISABLE, XAIE_ENABLE);
      for (int l=0; l<16; l++)
        XAieTile_LockRelease(&(TileInst[j][i]), l, 0x0, 0);
    }
  }
  // cores
  //

  //  mlir_initialize_cores();
  XAieTile_CoreControl(&(TileInst[col][1]), XAIE_DISABLE, XAIE_DISABLE);
  XAieTile_CoreControl(&(TileInst[col][2]), XAIE_DISABLE, XAIE_DISABLE);

  XAieTile_ShimColumnReset(&(TileInst[col][0]), XAIE_RESETENABLE);
  XAieTile_ShimColumnReset(&(TileInst[col][0]), XAIE_RESETDISABLE);

  // configure switchboxes
  //
  mlir_configure_switchboxes();
  XAieTile_ShimStrmMuxConfig(&(TileInst[col][0]), XAIETILE_SHIM_STRM_MUX_SOUTH7, XAIETILE_SHIM_STRM_MUX_DMA);

  // locks
  //
  //mlir_initialize_locks();

  // dmas
  //

  XAieTile_LockRelease(&(TileInst[col][2]), 0, 1, 0);
  u8 lock_ret = XAieTile_LockAcquire(&(TileInst[col][2]), 0, 1, 10000);
  assert(lock_ret);

  mlir_configure_dmas();
}

int main(int argc, char *argv[])
{
  auto col = 7;

  // setup the aie array
  setup_aie();

  XAieTile_DmWriteWord(&(TileInst[col][2]), 24*4, -1);

  // create the queue
  queue_t *q = nullptr;
  auto ret = queue_create(MB_QUEUE_SIZE, HSA_QUEUE_TYPE_SINGLE, &q);
  assert(ret == 0 && "failed to create queue!");

  // setup the shim dma descriptors
  XAieDma_Shim ShimDmaInst1;
  uint32_t *bram_ptr;

  #define DMA_COUNT 256

  auto burstlen = 4;
  XAieDma_ShimInitialize(&(TileInst[col][0]), &ShimDmaInst1);
  XAieDma_ShimBdSetAddr(&ShimDmaInst1, 1, HIGH_ADDR((u64)SHMEM_BASE), LOW_ADDR((u64)SHMEM_BASE), sizeof(u32) * DMA_COUNT);
  XAieDma_ShimBdSetAxi(&ShimDmaInst1, 1 , 0, burstlen, 0, 0, XAIE_ENABLE);
  XAieDma_ShimBdWrite(&ShimDmaInst1, 1);
  XAieDma_ShimSetStartBd((&ShimDmaInst1), XAIEDMA_SHIM_CHNUM_MM2S1, 1);

  auto cnt = XAieDma_ShimPendingBdCount(&ShimDmaInst1, XAIEDMA_SHIM_CHNUM_MM2S1);
  if (cnt)
    printf("%s %d Warn %d\n", __FUNCTION__, __LINE__, cnt);

  XAieDma_ShimChControl((&ShimDmaInst1), XAIEDMA_SHIM_CHNUM_MM2S1, XAIE_DISABLE, XAIE_DISABLE, XAIE_ENABLE);

  // reserve a packet in the queue
  uint64_t wr_idx = queue_add_write_index(q, 1);
  uint64_t packet_id = wr_idx % q->size;

  // herd_setup packet
  dispatch_packet_t *herd_pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  initialize_packet(herd_pkt);
  herd_pkt->type = HSA_PACKET_TYPE_AGENT_DISPATCH;

  // Set up the worlds smallest herd at 7,2
  herd_pkt->arg[0]  = AIR_PKT_TYPE_HERD_INITIALIZE;
  herd_pkt->arg[0] |= (AIR_ADDRESS_ABSOLUTE_RANGE << 48);
  herd_pkt->arg[0] |= (1L << 40);
  herd_pkt->arg[0] |= (7L << 32);
  herd_pkt->arg[0] |= (1L << 24);
  herd_pkt->arg[0] |= (2L << 16);
  
  herd_pkt->arg[1] = 0;  // Herd ID 0
  herd_pkt->arg[2] = 0;
  herd_pkt->arg[3] = 0;

  // dispatch packet
  signal_create(1, 0, NULL, (signal_t*)&herd_pkt->completion_signal);
  signal_create(0, 0, NULL, (signal_t*)&q->doorbell);
  signal_store_release((signal_t*)&q->doorbell, wr_idx);

  // wait for packet completion
  while (signal_wait_aquire((signal_t*)&herd_pkt->completion_signal, HSA_SIGNAL_CONDITION_EQ, 0, 0x80000, HSA_WAIT_STATE_ACTIVE) != 0) {
    printf("packet completion signal timeout on herd initialization!\n");
    printf("%x\n", herd_pkt->header);
    printf("%x\n", herd_pkt->type);
    printf("%x\n", herd_pkt->completion_signal);
  }

  // reserve another packet in the queue
  wr_idx = queue_add_write_index(q, 1);
  packet_id = wr_idx % q->size;

  // lock packet
  dispatch_packet_t *lock_pkt = (dispatch_packet_t*)(q->base_address_vaddr) + packet_id;
  initialize_packet(lock_pkt);
  lock_pkt->type = HSA_PACKET_TYPE_AGENT_DISPATCH;

  // Release lock 0 in 0,0 with value 0
  lock_pkt->arg[0]  = AIR_PKT_TYPE_XAIE_LOCK;
  lock_pkt->arg[0] |= (AIR_ADDRESS_HERD_RELATIVE << 48);
  lock_pkt->arg[1]  = 0;
  lock_pkt->arg[2]  = 1;
  lock_pkt->arg[3]  = 0;

  // dispatch packet
  signal_create(1, 0, NULL, (signal_t*)&lock_pkt->completion_signal);
  signal_create(0, 0, NULL, (signal_t*)&q->doorbell);
  signal_store_release((signal_t*)&q->doorbell, wr_idx);

  // wait for packet completion
  while (signal_wait_aquire((signal_t*)&lock_pkt->completion_signal, HSA_SIGNAL_CONDITION_EQ, 0, 0x80000, HSA_WAIT_STATE_ACTIVE) != 0) {
    printf("packet completion signal timeout on lock release!\n");
    printf("%x\n", lock_pkt->header);
    printf("%x\n", lock_pkt->type);
    printf("%x\n", lock_pkt->completion_signal);
  }

  //XAieTile_LockRelease(&(TileInst[col][2]), 0, 0, 0);

  // wait for shim dma to finish
  auto count = 0;
  while (XAieDma_ShimPendingBdCount(&ShimDmaInst1, XAIEDMA_SHIM_CHNUM_MM2S1)) {
    XAieLib_usleep(1000);
    count++;
    if (!(count % 1000)) {
      printf("%d seconds\n",count/1000);
      if (count == 2000) break;
    }
  }

  // we copied the start of the shared bram into tile memory,
  // fish out the queue id and check it
  uint32_t d = XAieTile_DmReadWord(&(TileInst[col][2]), 24*4);
  printf("ID %x\n", d);

  if (d == 0xacdc)
    printf("PASS!\n");
  else
    printf("fail.\n");
}