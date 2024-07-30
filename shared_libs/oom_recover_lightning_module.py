import gc
import torch
# import lightning as L
import pytorch_lightning as L


from abc import abstractmethod


class OOMRecoverable(L.LightningModule):

    def __init__(self, grad_acc:int=1, memory_safe_margin:int=3*1024**3) -> None:
        super().__init__()
        self.grad_acc = grad_acc
        self.total_oom_count = 0
        self.automatic_optimization = False
        self.memory_safe_margin = memory_safe_margin
        
    @abstractmethod
    def compute_loss(self, data):
        pass
    
    @abstractmethod
    def modify_data_on_oom(self, data):
        pass
    
    def on_train_epoch_start(self) -> None:
        # the memory of grad is preallocated for performance and stability.
        # except in the first step, the grad is set to none to detect untrained parameters.
        opt:torch.optim.Optimizer = self.optimizers()
        opt.zero_grad(set_to_none=(self.global_step == 0))
        
        # grad_acc does not expand over two training epochs.
        self.epoch_step = 0

    def training_step(self, data, batch_idx) -> None:

        opt:torch.optim.Optimizer = self.optimizers()
        
        # since the problem size of each scene is variable,
        # out of memory is frequently happen if the memory
        # budget is tight. Here we capture OOM exceptions,
        # and then drop some scene from the data_list to lower
        # the memory demands.
        
        while True:  # the retry when oom loop
            try:
                if len(data):
                    loss = self.compute_loss(data)
                else:
                    loss = 0. * sum(p.mean() for p in self.parameters())
                if get_free_memory(self.local_rank) >= self.memory_safe_margin:
                    break
                elif get_available_memory(self.local_rank) >= 1.8 * self.memory_safe_margin:
                    break
                else:
                    torch.cuda.empty_cache()
                    if get_free_memory(self.local_rank) < self.memory_safe_margin:
                        raise RuntimeError("out of memory (in terms of safe margin)")
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    loss = None
                    self.total_oom_count += 1
                    print(f'[WARNING] ran out of memory, used batch with less data (total oom {self.total_oom_count} times on rank {self.global_rank})')
                    data = self.modify_data_on_oom(data)
                    gc.collect()
                    torch.cuda.empty_cache()
                else:
                    raise e

        # currently it is not very possible to recover from an oom in backward in DDP.
        self.manual_backward(loss / self.grad_acc)

        # detect untrained parameters
        if self.global_step == 0:
            for name, param in self.named_parameters():
                if param.grad is None and param.requires_grad:
                    print(f"[WARNING] {name} is not trained.")
        
        # optimizer.step() every self.grad_acc steps
        self.epoch_step += 1
        if (self.epoch_step + 1) % self.grad_acc == 0:
            opt.step()
            opt.zero_grad(set_to_none=False)

    def on_train_epoch_end(self) -> None:
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step()
        
        # release some memory so that validation loop can be more safe.
        # when grad_acc > 1, it also means ignoring last samples in an training epoch.
        opt:torch.optim.Optimizer = self.optimizers()
        opt.zero_grad(set_to_none=True)


def get_reserved_but_unallocated_memory(device=None):
    reserved = torch.cuda.memory_reserved(device)
    allocated = torch.cuda.memory_allocated(device)
    return reserved - allocated

def get_free_memory(device=None):
    free = torch.cuda.mem_get_info(device)[0]
    return free

def get_available_memory(device=None):
    free = torch.cuda.mem_get_info(device)[0]
    reserved = torch.cuda.memory_reserved(device)
    allocated = torch.cuda.memory_allocated(device)
    return free + reserved - allocated
