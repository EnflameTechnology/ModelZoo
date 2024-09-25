import time
import os

class LoraUtils:
    def __init__(
        self,
        pipe=None,
        lora_list=None,
        adapter_weights=None,
        lora_scale=1.0,
        merge_lora_flag=True
    ):
        self.pipe = pipe
        self.lora_list = lora_list
        self.adapter_weights = adapter_weights
        self.lora_scale = lora_scale
        self.merge_lora_flag = merge_lora_flag

    def load_lora(self, lora_list, adapter_names=[]):
        num_lora = len(lora_list)
        if(num_lora > 0):
            if(len(adapter_names) == num_lora):
                adapter_name_list = adapter_names
            elif(len(adapter_names) > num_lora):    
                adapter_name_list = adapter_names[:num_lora]
            else:
                adapter_name_list = adapter_names + ["adapter-" + str(i) for i in range(num_lora - len(adapter_names))]
            t_0 = time.time()
            for j in range(num_lora):
                lora_weight_name = os.path.basename(lora_list[j])
                cur_adapter_name = adapter_name_list[j]
                self.pipe.load_lora_weights(lora_list[j], weight_name=lora_weight_name, adapter_name=cur_adapter_name)
            t_1 = time.time()
            print(f'loaded {num_lora} lora(s) successfully, costing time: {t_1 - t_0}')
        return adapter_name_list

    def unload_lora(self):
        self.pipe.unload_lora_weights()
        print(f'lora unloaded successfully!')

    def set_active_adapters(self, adapter_name_list, adapter_weights=None):
        if(adapter_weights is None):
            adapter_weights = [1.0] * len(adapter_name_list)
        elif(len(adapter_weights) < len(adapter_name_list)):
            adapter_weights = adapter_weights + [1.0] * (len(adapter_name_list) - len(adapter_weights))
        elif(len(adapter_weights) > len(adapter_name_list)):
            adapter_weights = adapter_weights[:len(adapter_name_list)]
        self.pipe.set_adapters(adapter_name_list, adapter_weights=adapter_weights)

    def get_active_adapters(self):
        '''
        gets the list of current activate adapters
        '''
        return self.pipe.get_active_adapters()

    def get_list_adapters(self):
        '''
        gets the list of all available adapters
        '''
        return self.pipe.get_list_adapters()

    def merge_lora_weights(self, merge_lora_flag=True, lora_scale=1.0):
        try:
            if(merge_lora_flag):
                t0 = time.time()
                self.pipe.fuse_lora(lora_scale=lora_scale)
                t1 = time.time()
                cur_active_adapters = self.get_active_adapters()
                num_active_adapters = len(cur_active_adapters)
                print(f'it took { t1-t0 } seconds to merge {num_active_adapters} lora(s): {cur_active_adapters}')
            else:
                print('lora(s) not merged into SD model, and will be used during inference per layer')
            return merge_lora_flag
        except Exception as e:
            print(e)
            print('failed to merge lora weights before inference, will merge lora outputs per layer during inference!')
            return False

    def unmerge_lora_weights(self):
        t0 = time.time()
        self.pipe.unfuse_lora()
        t1 = time.time()
        cur_active_adapters = self.get_active_adapters()
        num_active_adapters = len(cur_active_adapters)
        print(f'it took { t1-t0 } seconds to unmerge {num_active_adapters} lora(s): {cur_active_adapters}')

    def disable_lora(self):
        self.pipe.disable_lora()

    def enable_lora(self):
        self.pipe.enable_lora()

    def delete_adapters(self, adapter_names):
        self.pipe.delete_adapters(adapter_names)

def replace_prompt_str(prompt_str):
    prompt_str = prompt_str.replace(' ', '_')
    prompt_str = prompt_str.replace(',', '_')
    prompt_str = prompt_str.replace('.', '_')
    prompt_str = prompt_str.replace('"', '_')
    prompt_str = prompt_str.replace("'", '_')
    prompt_str = prompt_str.replace("(", '-')
    prompt_str = prompt_str.replace(")", '-')
    prompt_str = prompt_str.replace(">", '-')
    prompt_str = prompt_str.replace("<", '-')
    prompt_str = prompt_str.replace(":", '-')
    return prompt_str

def prompt_process(prompt_a, prompt_b):
    batch_size = len(prompt_a)
    prompt_b_len = len(prompt_b)
    prompt_b = prompt_b[:batch_size] if prompt_b_len>batch_size else prompt_b + ['']*(batch_size - prompt_b_len)

    return prompt_b

def prompt_align(prompt_a, prompt_list):
    num_prompt_list = len(prompt_list)
    res_prompt_list = []
    for i in range(num_prompt_list):
        prompt_b = prompt_process(prompt_a, prompt_list[i])
        res_prompt_list.append(prompt_b)

    return res_prompt_list[:]

def seed_everything(seed, device='gcu'):
    """
    Set a random seed to ensure reproducibility.

    Parameters:
        seed (int): The random seed
    """
    import random
    import numpy as np
    import torch

    # Set the seed for the Python random number generator
    random.seed(seed)

    # Set the seed for the NumPy random number generator
    np.random.seed(seed)

    # Set the seed for the PyTorch random number generator
    torch.manual_seed(seed)

    # If using CUDA, set the seed for the CUDA random number generator
    if device == 'cuda':
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # Set the seed for all GPUs

        # Set reproducibility (optional)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    elif device == 'gcu':
        import torch_gcu
        if torch.gcu.is_available():
            torch.gcu.manual_seed(seed)
            torch.gcu.manual_seed_all(seed)  # Set the seed for all GCUs