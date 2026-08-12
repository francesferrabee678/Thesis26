#added timing calls for debugging, and autosaving checkpoints each epoch


import os
import time
import torch
import hydra
import lightning.pytorch as pl
from m3drefclip.data.data_module import DataModule
from lightning.pytorch.callbacks import LearningRateMonitor, Callback, ModelCheckpoint
from m3drefclip.callback.gpu_cache_clean_callback import GPUCacheCleanCallback
from m3drefclip.callback.lr_decay_callback import LrDecayCallback
import torch.multiprocessing as mp

mp.set_sharing_strategy('file_system')

class TimingCallback(Callback):
    def __init__(self):
        super().__init__()
        self.last_batch_end_time = 0
        self.batch_start_time = 0
        self.forward_backward_time = 0

    def on_train_epoch_start(self, trainer, pl_module):
        print(f"\n[TIMING] >>> Starting Epoch {trainer.current_epoch}")
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.last_batch_end_time = time.time()

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Time spent between the end of the last batch and the start of this one
        self.data_load_time = time.time() - self.last_batch_end_time
        self.batch_start_time = time.time()

    def on_before_optimizer_step(self, trainer, pl_module, optimizer):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Time spent doing model(x) and loss.backward()
        self.forward_backward_time = time.time() - self.batch_start_time

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        current_time = time.time()
        
        # Time spent applying gradients to weights
        optimizer_time = current_time - (self.batch_start_time + self.forward_backward_time)
        
        # Total time for this specific step cycle
        total_step_time = current_time - self.last_batch_end_time

        # Print every step for the first 30 steps so we can watch the degradation happen live
        if batch_idx <= 30 or batch_idx % 5 == 0:
            device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
            if device != 'cpu':
                allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)
                reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)
            else:
                allocated, reserved = 0, 0

            print(
                f"\n[SUB-STEP PROFILER] Epoch {trainer.current_epoch} | Batch {batch_idx}\n"
                f"  ├─ Data Loading Wait: {self.data_load_time:.3f}s\n"
                f"  ├─ Forward/Backward:  {self.forward_backward_time:.3f}s\n"
                f"  ├─ Optimizer Step:    {optimizer_time:.3f}s\n"
                f"  ├─ TOTAL CYCLE TIME:  {total_step_time:.3f}s\n"
                f"  └─ GPU VRAM: Allocated {allocated:.2f}GB | Reserved {reserved:.2f}GB"
            )
            
            # Automated Diagnosis Warnings
            if self.data_load_time > 10.0 and self.data_load_time > self.forward_backward_time * 2:
                print("  => ⚠️ BOTTLENECK: Data Loading (CPU/Disk I/O is starving the GPU).")
            elif self.forward_backward_time > 10.0:
                print("  => ⚠️ BOTTLENECK: Math/GPU (Forward/Backward pass is dragging).")
            elif allocated > 70.0: # Adjust this based on your A100/H100 VRAM limits
                print("  => ⚠️ BOTTLENECK: Memory Pressure (GPU is close to OOM).")

        # Reset the clock for the next batch's data loading phase
        self.last_batch_end_time = time.time()

def init_callbacks(cfg):
    checkpoint_monitor = hydra.utils.instantiate(cfg.checkpoint_monitor)
    gpu_cache_clean_monitor = GPUCacheCleanCallback()
    lr_decay_callback = LrDecayCallback()
    lr_monitor = LearningRateMonitor(logging_interval="epoch")
    timing_monitor = TimingCallback() # <-- Added our new tracker
    autosave = ModelCheckpoint(
    dirpath=os.path.join(cfg.experiment_output_path, "training"),
    filename="autosave-{epoch}", save_top_k=1,
    every_n_epochs=1, save_on_train_epoch_end=True,   # no monitor → always fires
    )
    return [checkpoint_monitor, autosave, gpu_cache_clean_monitor, lr_decay_callback, lr_monitor, timing_monitor]

@hydra.main(version_base=None, config_path="config", config_name="global_config")
def main(cfg):
    print("\n[PROFILER] Starting script...")
    t0 = time.time()
    
    pl.seed_everything(cfg.train_seed, workers=True)
    os.makedirs(os.path.join(cfg.experiment_output_path, "training"), exist_ok=True)

    print(f"[PROFILER] Initializing DataModule... ({(time.time()-t0):.2f}s)")
    t1 = time.time()
    data_module = DataModule(cfg.data)

    print(f"[PROFILER] Initializing Model... ({(time.time()-t1):.2f}s)")
    t2 = time.time()
    model = hydra.utils.instantiate(cfg.model.model_name, cfg)

    if "detector_path" in cfg:
        print("[PROFILER] Loading Detector Weights...")
        detector_weights = torch.load(cfg.detector_path)["state_dict"]
        model.detector.load_state_dict(detector_weights)

    logger = hydra.utils.instantiate(cfg.logger)
    callbacks = init_callbacks(cfg)

    print(f"[PROFILER] Initializing Trainer... ({(time.time()-t2):.2f}s)")
    trainer = pl.Trainer(callbacks=callbacks, logger=logger, **cfg.trainer)

    if cfg.ckpt_path is not None:
        assert os.path.exists(cfg.ckpt_path), "Error: Checkpoint path does not exist."

    print("\n[PROFILER] HANDING OFF TO PYTORCH LIGHTNING TRAINER.FIT()...")
    print("\n[PROFILER] --- CURRENT NVIDIA-SMI STATE BEFORE TRAINING ---")
    os.system("nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.total,memory.used --format=csv")
    print("-----------------------------------------------------------\n")

    print("\n[PROFILER] HANDING OFF TO PYTORCH LIGHTNING TRAINER.FIT()...")
    trainer.fit(model=model, datamodule=data_module, ckpt_path=cfg.ckpt_path)


if __name__ == '__main__':
    main()