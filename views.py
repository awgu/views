import functools
import itertools
from typing import Any, List

import torch

LR = 1e-3


class FlatParameter(torch.nn.Parameter):
    ...


class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.linear = torch.nn.Linear(5, 7)
        self.relu = torch.nn.ReLU()

        # This is constructed as a dict once per iteration in the first
        # pre-backward hook to run, which iterates over this module's
        # parameters and adds an entry to mapping parameters with
        # `requires_grad=True` to `False` to indicate to wait for their
        # gradient to be ready before running the post-backward hook logic
        self._param_to_is_grad_ready = None

    @torch.no_grad()
    def flatten(self):
        """Flattens the weight and bias into a flattened parameter and modifies
        their storage to point into the flattened parameter's storage."""
        self.flat_param = FlatParameter(torch.cat(
            [self.linear.weight.reshape(-1), self.linear.bias.reshape(-1)], 0,
        ))
        views = torch.split(
            self.flat_param, [self.linear.weight.numel(), self.linear.bias.numel()], dim=0,
        )
        # Operations on `weight` and `bias` do not propagate through the
        # autograd graph
        self.linear.weight.data = views[0].view(self.linear.weight.shape)
        self.linear.bias.data = views[1].view(self.linear.bias.shape)

    def forward(self, x):
        z = self.linear(x)
        return self.relu(z)

    def register_pre_bwd_hooks(self, verbose: bool = False):
        def pre_bwd_hook(param, param_name: str, *unused: Any):
            """Allocates a gradient for the flattened parameter if needed and
            ensures on the first pre-backward hook of this iteration that each
            original parameter's ``.grad`` points into this allocated
            gradient."""
            if verbose:
                print(f"Pre bwd hook from {param_name}!")
            if self._param_to_is_grad_ready is None:
                self._param_to_is_grad_ready = {}
                for p in self.parameters():
                    if p.requires_grad:
                        self._param_to_is_grad_ready[p] = False
            if any(self._param_to_is_grad_ready.values()):
                self._param_to_is_grad_ready[param] = True
                assert self.flat_param.grad is not None
                return
            # self._param_to_is_grad_ready[param] = True
            if self.flat_param.grad is None:
                # Must be initialized as zero for mathematical correctness
                self.flat_param.grad = torch.zeros_like(self.flat_param)
            assert len(self.flat_param.grad.shape) == 1
            # Set the original parameters' `.grad` as views into the flattened
            # gradient
            offset = 0
            for param in (self.linear.weight, self.linear.bias):
                param.grad = torch.narrow(
                    self.flat_param.grad, 0, offset, param.numel(),
                ).view(param.shape)
                offset += param.numel()

        self.linear.weight.register_hook(functools.partial(pre_bwd_hook, self.linear.weight, "weight"))
        self.linear.bias.register_hook(functools.partial(pre_bwd_hook, self.linear.bias, "bias"))

    def register_post_bwd_hooks(self, verbose: bool = False):
        def post_bwd_hook(param_name: str, *unused: Any):
            """Resets internal data for the next iteration if this hook is the
            last one to be called for this module."""
            if verbose:
                print(f"Post bwd hook from {param_name}!")
            if all(self._param_to_is_grad_ready.values()):
                if verbose:
                    print(f"\"Reduce scatter\" from {param_name}!")
                self._param_to_is_grad_ready = None  # reset for next iteration

        for p, n in [(self.linear.weight, "weight"), (self.linear.bias, "bias")]:
            p_tmp = p.expand_as(p)
            grad_acc = p_tmp.grad_fn.next_functions[0][0]
            grad_acc.register_hook(functools.partial(post_bwd_hook, n))

    def named_parameters(self, *args, **kwargs):
        for param_name, param in super().named_parameters(*args, **kwargs):
            if not isinstance(param, FlatParameter):
                yield (param_name, param)

    def fsdp_parameters(self, *args, **kwargs):
        for _, param in self.named_fsdp_parameters(*args, **kwargs):
            yield param

    def named_fsdp_parameters(self, *args, **kwargs):
        for param_name, param in super().named_parameters(*args, **kwargs):
            if isinstance(param, FlatParameter):
                yield (param_name, param)


def check_model_parameters(models: List[torch.nn.Module]):
    for model1, model2 in itertools.combinations(models, 2):
        assert torch.allclose(model1.linear.weight, model2.linear.weight)
        assert torch.allclose(model1.linear.bias, model2.linear.bias)


def check_model_gradients(models: List[torch.nn.Module]):
    for model1, model2 in itertools.combinations(models, 2):
        assert torch.allclose(model1.linear.weight.grad, model2.linear.weight.grad)
        assert torch.allclose(model1.linear.bias.grad, model2.linear.bias.grad)


def fwd_bwd(models: List[torch.nn.Module], inp):
    for model in models:
        out = model(inp)
        loss = out.sum()
        loss.backward()


def opt(optims: List[torch.optim.Optimizer]):
    for optim in optims:
        optim.step()


def zero(optims: List[torch.optim.Optimizer]):
    for optim in optims:
        optim.zero_grad()


def run_iter(models, optims, inp, zero_grad: bool):
    if zero_grad:
        zero(optims)
    fwd_bwd(models, inp)
    check_model_gradients(models)
    check_model_parameters(models)
    opt(optims)
    check_model_parameters(models)


def main():
    # Model without flattening
    model1 = Model()
    optim1 = torch.optim.Adam(model1.parameters(), lr=LR)
    # Model with flattening, to optimize in terms of original parameters
    model2 = Model()
    model2.flatten()
    model2.register_pre_bwd_hooks()
    model2.register_post_bwd_hooks()
    optim2 = torch.optim.Adam([model2.linear.weight, model2.linear.bias], lr=LR)
    # Model with flattening, to optimize in terms of the flattened parameter
    model3 = Model()
    model3.flatten()
    model3.register_pre_bwd_hooks()
    model3.register_post_bwd_hooks()
    optim3 = torch.optim.Adam([model3.flat_param], lr=LR)

    models = [model1, model2, model3]
    optims = [optim1, optim2, optim3]
    inp = torch.randn(8, 5)
    run_iter(models, optims, inp, False)
    run_iter(models, optims, inp, True)
    run_iter(models, optims, inp, True)

    # Check that `.parameters()` returns the original parameters
    assert len(list(model1.parameters())) == 2
    assert len(list(model2.parameters())) == 2
    assert len(list(model3.parameters())) == 2

    # Check that the new `.fsdp_parameters()` returns only flattened parameters
    assert len(list(model1.fsdp_parameters())) == 0
    assert len(list(model2.fsdp_parameters())) == 1
    assert len(list(model3.fsdp_parameters())) == 1

    print("Yay!")


if __name__ == "__main__":
    main()
