"""Training callbacks: adaptive KL penalty and the stage-2 reward curriculum."""

import wandb
from transformers import TrainerCallback

from .rewards import is_rank0


class AdaptiveKLScheduler(TrainerCallback):
    """Nudge ``beta`` up or down to hold the observed KL near ``target``.

    A fixed KL coefficient either strangles exploration early or lets the policy
    drift late, so we track the logged ``train/kl`` and adjust multiplicatively
    inside [``beta_min``, ``beta_max``].
    """

    def __init__(self, target=0.13, tol=0.03, up=1.5, down=0.66,
                 beta_min=5e-4, beta_max=5e-2):
        self.tgt, self.tol, self.up, self.down = target, tol, up, down
        self.beta_min, self.beta_max = beta_min, beta_max

    def on_log(self, args, state, control, **kw):
        logs = (state.log_history or [])[-1] if state.log_history else {}
        kl = logs.get("train/kl", None)
        if kl is None:
            return
        beta = args.beta
        if kl > self.tgt + self.tol:
            beta = min(self.beta_max, beta * self.up)
        elif kl < self.tgt - self.tol:
            beta = max(self.beta_min, beta * self.down)
        args.beta = beta
        if is_rank0():
            wandb.log({"sched/beta": beta}, commit=False)


class RewardWeightScheduler(TrainerCallback):
    """Linearly move the reward mix from ``start`` to ``end`` weights.

    Mutates ``weights`` in place; the reward wrappers read it through a closure,
    so the change takes effect on the next reward call. The interpolation
    finishes at ``finish_frac`` of the estimated run, leaving the tail of
    training at the final mix.
    """

    def __init__(self, weights: dict, start: dict, end: dict,
                 total_steps_estimate: int, finish_frac: float = 0.6):
        self.weights = weights
        self.start, self.end = start, end
        self.smax = max(1, total_steps_estimate)
        self.finish_frac = finish_frac

    def on_step_end(self, args, state, control, **kw):
        p = min(1.0, state.global_step / (self.finish_frac * self.smax))
        for k in self.start:
            self.weights[k] = self.start[k] + p * (self.end[k] - self.start[k])
        if is_rank0() and wandb.run is not None:
            wandb.log({f"sched/w_{k}": v for k, v in self.weights.items()}, commit=False)
