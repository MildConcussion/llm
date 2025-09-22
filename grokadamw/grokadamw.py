import math
import torch
from torch.optim import Optimizer
from typing import Iterable, Callable, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GrokAdamW(Optimizer):
    def __init__(self, params: Iterable[torch.Tensor], lr: float = 1e-3, betas: tuple[float, float] = (0.9, 0.999),
                 eps: float = 1e-8, weight_decay: float = 1e-2, alpha_init: float = 0.98, lamb: float = 2.0,
                 gamma: float = 0.1, grokking_signal_fns: Optional[list[Callable[[], float]]] = None,
                 grokking_signal_decay_rate: float = 0.1, gradient_clipping: float = 1.0):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= alpha_init <= 1.0:
            raise ValueError(f"Invalid alpha_init value: {alpha_init}")

        # Add device caching for performance
        self._device = None

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                        alpha_init=alpha_init, lamb=lamb, gamma=gamma,
                        grokking_signal_fns=grokking_signal_fns,
                        grokking_signal_decay_rate=grokking_signal_decay_rate,
                        gradient_clipping=gradient_clipping)
        super(GrokAdamW, self).__init__(params, defaults)

    def _get_device(self, params):
        """Cache device to avoid repeated checks for MPS optimization"""
        if self._device is None and params:
            self._device = params[0].device
        return self._device

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], float]] = None) -> Optional[float]:
        return self._step_impl(closure)

    def _step_impl(self, closure: Optional[Callable[[], float]]) -> Optional[float]:
        """Performs a single optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            grokking_signal = self._compute_grokking_signal(group)

            params_with_grad = [p for p in group['params'] if p.grad is not None]
            if not params_with_grad:
                continue

            grads = [p.grad for p in params_with_grad]

            self._update_group(group, params_with_grad, grads, grokking_signal)

        return loss

    @staticmethod
    def _default_grokking_signal(train_loss: Optional[float], eval_loss: Optional[float]) -> float:
        """Default grokking signal function based on loss difference."""
        if train_loss is None or eval_loss is None:
            return 0.0
        diff = max(0, eval_loss - train_loss)
        max_loss = max(eval_loss, train_loss)
        return diff / max_loss if max_loss > 0 else 0.0

    def _compute_grokking_signal(self, group: dict) -> Optional[float]:
        """Computes a combined grokking signal from multiple functions."""
        if group['grokking_signal_fns'] is None:
            train_loss = group.get('train_loss', None)
            eval_loss = group.get('eval_loss', None)
            # print(f"train_loss: {train_loss}, eval_loss: {eval_loss}")
            return self._default_grokking_signal(train_loss, eval_loss)

        signals = []
        for fn in group['grokking_signal_fns']:
            try:
                signal = fn()
                if signal is not None:
                    signals.append(signal)
            except Exception as e:
                logger.warning(f"Error in grokking_signal_fn: {e}. Ignoring this function.")

        return sum(signals) / len(signals) if signals else None

    def _update_group(self, group: dict, params: list[torch.Tensor], grads: list[torch.Tensor],
                      grokking_signal: Optional[float]) -> None:
        # Early exit if no parameters
        if not params:
            return

        # Get device from first parameter (assume all params on same device)
        device = self._get_device(params)

        # Pre-compute scalars outside loop for MPS performance
        beta1, beta2 = group['betas']
        lr = group['lr']
        weight_decay = group['weight_decay']
        eps = group['eps']
        lamb = group['lamb']
        gamma = group['gamma']

        # Compute grokking alpha once outside loop
        alpha = group['alpha_init']
        if grokking_signal is not None and grokking_signal > 0.3:
            # Sharper response to strong grokking signals
            decay_factor = group['grokking_signal_decay_rate'] * (1 + grokking_signal * 2)
            alpha = alpha * math.exp(-decay_factor * grokking_signal)

        # Batch gradient clipping for all parameters at once (MPS optimization)
        if group['gradient_clipping'] > 0:
            torch.nn.utils.clip_grad_norm_(params, group['gradient_clipping'])

        # Use autocast outside parameter loop for MPS efficiency
        with torch.amp.autocast(device_type='mps'):
            for i, (p, grad) in enumerate(zip(params, grads)):
                state = group.get('state', {}).get(p, {})

                # Initialize state on device if needed (keep on MPS device)
                if not state:
                    state = {
                        'step': 0,
                        'exp_avg': torch.zeros_like(p, device=device),
                        'exp_avg_sq': torch.zeros_like(p, device=device),
                        'grok_ema': torch.zeros_like(p, device=device)
                    }
                    if 'state' not in group:
                        group['state'] = {}
                    group['state'][p] = state

                exp_avg = state['exp_avg']
                exp_avg_sq = state['exp_avg_sq']
                grok_ema = state['grok_ema']

                state['step'] += 1
                step = state['step']

                # Layer-wise beta1 adjustment
                layer_beta1 = beta1 * (1 - gamma)**i

                # Update grok_ema (fused operations for MPS performance)
                grok_ema.mul_(alpha).add_(grad, alpha=1 - alpha)
                grok_grad = grad.add(grok_ema, alpha=lamb)  # In-place add

                # Update moments (fused operations)
                exp_avg.mul_(layer_beta1).add_(grok_grad, alpha=1 - layer_beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grok_grad, grok_grad, value=1 - beta2)

                # Bias correction (pre-compute for efficiency)
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                # Use numerically stable ordering per bnb Adam reasoning
                sqrt_bias_correction2 = math.sqrt(bias_correction2)
                step_size = lr * sqrt_bias_correction2 / bias_correction1

                # Update parameters (fused operations)
                p.mul_(1 - lr * weight_decay)
                # Compute denom as sqrt(v) + eps * sqrt(bias_correction2) for better FP stability
                denom = exp_avg_sq.sqrt().add_(eps * sqrt_bias_correction2)
                p.addcdiv_(exp_avg, denom, value=-step_size)

    def state_dict(self):
        state_dict = super().state_dict()
        for group in state_dict['param_groups']:
            group['grokking_signal_fns'] = None  # Cannot serialize functions
        return state_dict

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        for group in self.param_groups:
            group['grokking_signal_fns'] = self.defaults['grokking_signal_fns']

    def __setstate__(self, state: dict) -> None:
        super().__setstate__(state)
        for group in self.param_groups:
            group.setdefault('grokking_signal_fns', [])
            group.setdefault('grokking_signal_decay_rate', 0.1)
            group.setdefault('gradient_clipping', 1.0)