import torch
import torch.nn as nn
import os
import torch.autograd as autograd

# Ensure the compiler can be found
try:
    from .compiler import compile_and_load_kernel
except ImportError:
    # If running as a script, modify path to import from parent directory
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from mps.compiler import compile_and_load_kernel


# JIT compile and load the custom MPS kernel
# This will be done only once when the module is first imported.
script_dir = os.path.dirname(os.path.abspath(__file__))
# Only include compilable sources; headers are included automatically
sources = [os.path.join(script_dir, 'CustomRMSNorm.mm')]
custom_rmsnorm_lib = compile_and_load_kernel("CustomRMSNorm", sources)


class CustomRMSNormFunction(autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, eps, half_precision, cpp_impl):
        output = custom_rmsnorm_lib.rms_norm_forward(cpp_impl, input, weight)
        ctx.save_for_backward(input, weight)
        ctx.cpp_impl = cpp_impl
        ctx.eps = eps
        ctx.half_precision = half_precision
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, weight = ctx.saved_tensors
        dx, dw = custom_rmsnorm_lib.rms_norm_backward(ctx.cpp_impl, grad_output, input, weight)
        return dx, dw, None, None, None


class CustomRMSNorm(nn.Module):
    """
    A custom RMSNorm module that uses a hand-written Metal kernel for the forward pass.
    """
    def __init__(self, dim: int, eps: float = 1e-6, half_precision: bool = False):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.half_precision = half_precision
        # Learnable scaling parameter (gamma)
        self.weight = nn.Parameter(torch.ones(dim))
        self._weight_device = None

        # We will lazily initialize the C++ implementation on the first forward pass
        # to ensure it's on the correct device.
        self._cpp_impl = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the custom RMSNorm.
        """
        # The C++ implementation expects a 2D tensor, so we reshape.
        original_shape = x.shape
        is_3d = len(original_shape) == 3
        if is_3d:
            B, T, C = original_shape
            x_reshaped = x.view(-1, self.dim)
        else:
            x_reshaped = x  # Assume 2D input is already (N, dim)

        # Ensure tensors are on the MPS device and are memory-contiguous.
        # This is a requirement for passing their data pointers to Metal.
        device = x.device
        if device.type != 'mps':
            raise RuntimeError("CustomRMSNorm only supports MPS device.")

        input_tensor = x_reshaped if x_reshaped.is_contiguous() else x_reshaped.contiguous()

        # Cache weight on device
        if self._weight_device is None or self._weight_device.device != device:
            self._weight_device = self.weight.to(device)
        weight_tensor = self._weight_device if self._weight_device.is_contiguous() else self._weight_device.contiguous()

        # Lazy initialization of the C++ object.
        # This ensures the Metal device is correctly captured from the current context.
        if self._cpp_impl is None:
            self._cpp_impl = custom_rmsnorm_lib.rms_norm_new(self.dim, self.eps, self.half_precision)
            if self._cpp_impl is None:
                raise RuntimeError("Failed to initialize CustomRMSNorm C++ implementation.")

        import time
        import os
        if os.environ.get('BENCHMARK_RMSNORM', '0') == '1':
            torch.mps.synchronize()
            start_mem = torch.mps.current_allocated_memory()
            start_time = time.perf_counter()

        # Call the forward function from our custom C++ library
        output_reshaped = CustomRMSNormFunction.apply(input_tensor, weight_tensor, self.eps, self.half_precision, self._cpp_impl)

        if os.environ.get('BENCHMARK_RMSNORM', '0') == '1':
            torch.mps.synchronize()
            end_time = time.perf_counter()
            end_mem = torch.mps.current_allocated_memory()
            print(f"CustomRMSNorm forward time: {(end_time - start_time)*1000:.2f} ms")
            print(f"Memory usage: {end_mem - start_mem} bytes")

        # Reshape the output back to the original 3D shape
        return output_reshaped.view(original_shape)

    def __del__(self):
        """
        Destructor to free the C++ object and prevent memory leaks.
        """
        if hasattr(self, '_cpp_impl') and self._cpp_impl is not None:
            try:
                custom_rmsnorm_lib.rms_norm_free(self._cpp_impl)
            except (AttributeError, TypeError):
                pass  # Silently ignore errors during shutdown
            object.__setattr__(self, '_cpp_impl', None)

    def extra_repr(self) -> str:
        return f'dim={self.dim}, eps={self.eps}'

if __name__ == '__main__':
    import os
    import time
    os.environ['BENCHMARK_RMSNORM'] = '1'

    print("Running enhanced verification test for CustomRMSNorm...")

    class NativeRMSNorm(nn.Module):
        def __init__(self, dim, eps=1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))

        def _norm(self, x):
            return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        def forward(self, x):
            output = self._norm(x)
            return output * self.weight

    device = 'mps'
    configs = [
        {'dim': 32, 'batch_size': 32, 'seq_len': 32, 'half': False},
        {'dim': 64, 'batch_size': 4, 'seq_len': 16, 'half': False},
        {'dim': 64, 'batch_size': 4, 'seq_len': 16, 'half': True},
        {'dim': 512, 'batch_size': 32, 'seq_len': 128, 'half': False},
        {'dim': 512, 'batch_size': 32, 'seq_len': 128, 'half': True},
    ]

    for cfg in configs:
        print(f"\nTesting config: dim={cfg['dim']}, batch={cfg['batch_size']}, seq={cfg['seq_len']}, half={cfg['half']}")

        dtype = torch.float16 if cfg['half'] else torch.float32
        input_data = torch.randn(cfg['batch_size'], cfg['seq_len'], cfg['dim'], device=device, dtype=dtype)
        input_data.requires_grad = True

        custom_norm = CustomRMSNorm(cfg['dim'], half_precision=cfg['half']).to(device)
        native_norm = NativeRMSNorm(cfg['dim']).to(device)

        native_norm.weight.data.copy_(custom_norm.weight.data)
        if cfg['half']:
            native_norm.weight.data = native_norm.weight.data.half()
            custom_norm.weight.data = custom_norm.weight.data.half()
        custom_norm.weight.requires_grad = True
        native_norm.weight.requires_grad = True

        # Forward and backward for custom
        start_time = time.perf_counter()
        output_custom = custom_norm(input_data)
        torch.mps.synchronize()
        custom_fwd_time = (time.perf_counter() - start_time) * 1000

        grad_output = torch.randn_like(output_custom)
        start_time = time.perf_counter()
        output_custom.backward(grad_output)
        dx_custom = input_data.grad.clone()
        dw_custom = custom_norm.weight.grad.clone()
        torch.mps.synchronize()
        custom_bwd_time = (time.perf_counter() - start_time) * 1000

        # Forward and backward for native
        input_data.grad = None
        start_time = time.perf_counter()
        output_native = native_norm(input_data)
        torch.mps.synchronize()
        native_fwd_time = (time.perf_counter() - start_time) * 1000

        start_time = time.perf_counter()
        output_native.backward(grad_output)
        dx_native = input_data.grad.clone()
        dw_native = native_norm.weight.grad.clone()
        torch.mps.synchronize()
        native_bwd_time = (time.perf_counter() - start_time) * 1000

        # Compare forward
        fwd_close = torch.allclose(output_custom.float(), output_native.float(), atol=1e-2 if cfg['half'] else 1e-6)
        print(f"Forward outputs close: {fwd_close}")
        if not fwd_close:
            print(f"Max forward diff: {(output_custom - output_native).abs().max().item()}")

        # Compare backward
        dx_close = torch.allclose(dx_custom.float(), dx_native.float(), atol=5e-1 if cfg['half'] else 5e-4)
        dw_close = torch.allclose(dw_custom.float(), dw_native.float(), atol=5e-1 if cfg['half'] else 5e-4)
        print(f"Input gradients close: {dx_close}")
        if not dx_close:
            print(f"Max dx diff: {(dx_custom - dx_native).abs().max().item()}")
        print(f"Weight gradients close: {dw_close}")
        if not dw_close:
            print(f"Max dw diff: {(dw_custom - dw_native).abs().max().item()}")

        print(f"Custom fwd time: {custom_fwd_time:.2f} ms, bwd time: {custom_bwd_time:.2f} ms")
        print(f"Native fwd time: {native_fwd_time:.2f} ms, bwd time: {native_bwd_time:.2f} ms")
        print(f"Fwd speedup: {native_fwd_time / custom_fwd_time:.2f}x")
        print(f"Bwd speedup: {native_bwd_time / custom_bwd_time:.2f}x")

    print("Enhanced verification with backward complete!")