import torch
from torchsde._brownian.derived import BrownianTree
from torchsde._brownian import brownian_interval
from torchsde.types import Optional, Scalar, Tensor
from diffusers.schedulers.scheduling_dpmsolver_sde import BrownianTreeNoiseSampler
from diffusers.configuration_utils import ConfigMixin
from typing import Any, Dict

def diffusers_gcu_getattr(self, name: str) -> Any:
    """
    The only reason we overwrite `getattr` here is to gracefully deprecate accessing
    config attributes directly. See https://github.com/huggingface/diffusers/pull/3129

    Tihs funtion is mostly copied from PyTorch's __getattr__ overwrite:
    https://pytorch.org/docs/stable/_modules/torch/nn/modules/module.html#Module
    """

    is_in_config = "_internal_dict" in self.__dict__ and hasattr(self.__dict__["_internal_dict"], name)
    is_attribute = name in self.__dict__

    if is_in_config and not is_attribute:
        deprecation_message = f"Accessing config attribute `{name}` directly via '{type(self).__name__}' object attribute is deprecated. Please access '{name}' over '{type(self).__name__}'s config object instead, e.g. 'scheduler.config.{name}'."
        return self._internal_dict[name]

    raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")

def BrownianTree_init(self, t0: Scalar,
                 w0: Tensor,
                 t1: Optional[Scalar] = None,
                 w1: Optional[Tensor] = None,
                 entropy: Optional[int] = None,
                 tol: float = 1e-6,
                 pool_size: int = 24,
                 cache_depth: int = 9,
                 safety: Optional[float] = None):

    if t1 is None:
        t1 = t0 + 1
    if w1 is None:
        W = None
    else:
        W = w1 - w0
    self._w0 = w0
    self._interval = brownian_interval.BrownianInterval(t0=t0,
                                                        t1=t1,
                                                        size=w0.shape,
                                                        dtype=w0.dtype,
                                                        device='cpu',
                                                        entropy=entropy,
                                                        tol=tol,
                                                        pool_size=pool_size,
                                                        halfway_tree=True,
                                                        W=W)
    super(BrownianTree, self).__init__()

def BrownianTreeNoiseSampler_call(self, sigma, sigma_next):
    t0, t1 = self.transform(torch.as_tensor(sigma)), self.transform(torch.as_tensor(sigma_next))
    return self.tree(t0, t1).to('gcu') / (t1 - t0).abs().sqrt()

def get_scheduler(scheduler_type, config_path, *args):

    if len(args) > 1:
        raise ValueError("Too many arguments provided.")

    if scheduler_type == "dpm++_2m":
        from diffusers import DPMSolverMultistepScheduler

        scheduler = DPMSolverMultistepScheduler.from_pretrained(config_path)
    elif scheduler_type == "dpm++_2m_karras":
        from diffusers import DPMSolverMultistepScheduler

        scheduler = DPMSolverMultistepScheduler.from_pretrained(
            config_path, use_karras_sigmas=True
        )
    elif scheduler_type == "dpm++_2m_sde":
        from diffusers import DPMSolverMultistepScheduler

        scheduler = DPMSolverMultistepScheduler.from_pretrained(
            config_path, algorithm_type="sde-dpmsolver++"
        )
    elif scheduler_type == "euler_a":
        from diffusers import EulerAncestralDiscreteScheduler

        scheduler = EulerAncestralDiscreteScheduler.from_pretrained(
            config_path
        )
    elif scheduler_type == "lms":
        from diffusers import LMSDiscreteScheduler

        scheduler = LMSDiscreteScheduler.from_pretrained(
            config_path, use_karras_sigmas=True
        )
    elif scheduler_type == "lms_karras":
        from diffusers import LMSDiscreteScheduler

        scheduler = LMSDiscreteScheduler.from_pretrained(config_path)
    elif scheduler_type == "unipc":
        from diffusers import UniPCMultistepScheduler

        scheduler = UniPCMultistepScheduler.from_pretrained(config_path)
    elif scheduler_type == "ddim":
        from diffusers import DDIMScheduler

        scheduler = DDIMScheduler.from_pretrained(config_path)
    elif scheduler_type == "heun":
        from diffusers import HeunDiscreteScheduler

        scheduler = HeunDiscreteScheduler.from_pretrained(config_path)
    elif scheduler_type == "dpm2":
        from diffusers import KDPM2DiscreteScheduler

        scheduler = KDPM2DiscreteScheduler.from_pretrained(config_path)
    elif scheduler_type == "dpm2-a":
        from diffusers import KDPM2AncestralDiscreteScheduler

        scheduler = KDPM2AncestralDiscreteScheduler.from_pretrained(
            config_path
        )
    elif scheduler_type == "euler":
        from diffusers import EulerDiscreteScheduler

        scheduler = EulerDiscreteScheduler.from_pretrained(config_path)

    elif scheduler_type == "euler_t":
        from diffusers import EulerDiscreteScheduler

        scheduler = EulerDiscreteScheduler.from_pretrained(config_path, timestep_spacing="trailing")

    elif scheduler_type == "ddpm":
        from diffusers import DDPMScheduler

        scheduler = DDPMScheduler.from_pretrained(config_path)
    elif scheduler_type == "dpm++_2s":
        from diffusers import DPMSolverSinglestepScheduler

        scheduler = DPMSolverSinglestepScheduler.from_pretrained(config_path)
    elif scheduler_type == "dpm++_2s_karras":
        from diffusers import DPMSolverSinglestepScheduler

        scheduler = DPMSolverSinglestepScheduler.from_pretrained(
            config_path, use_karras_sigmas=True
        )
    elif scheduler_type == "pndm":
        from diffusers import PNDMScheduler

        scheduler = PNDMScheduler.from_pretrained(config_path)
    elif scheduler_type == "deis":
        from diffusers import DEISMultistepScheduler

        scheduler = DEISMultistepScheduler.from_pretrained(config_path)
    elif scheduler_type == "dpm_2m_karras":
        from diffusers import DPMSolverMultistepScheduler

        scheduler = DPMSolverMultistepScheduler.from_pretrained(config_path, use_karras_sigmas=True, algorithm_type="sde-dpmsolver++")
    elif scheduler_type == "dpm_sde_karras":

        from diffusers import DPMSolverSDEScheduler
        BrownianTree.__init__ = BrownianTree_init
        BrownianTreeNoiseSampler.__call__ = BrownianTreeNoiseSampler_call
        ConfigMixin.__getattr__=diffusers_gcu_getattr
        if len(args) == 1:
            for seed in args:
                scheduler = DPMSolverSDEScheduler.from_pretrained(config_path, use_karras_sigmas=True, noise_sampler_seed=seed)
        else:
            scheduler = DPMSolverSDEScheduler.from_pretrained(config_path, use_karras_sigmas=True)
    else:
        raise ValueError(f"unsupported scheduler type: {scheduler_type}")

    return scheduler
