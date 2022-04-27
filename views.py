import functools
import itertools
from typing import Any, List, Set

import torch

LR = 1e-3


class FlatParameter(torch.nn.Parameter):
    ...


class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.linear0 = torch.nn.Linear(5, 7)
        self.linear1 = torch.nn.Linear(7, 3)
        self.relu = torch.nn.ReLU()

        # This is constructed as a dict once per iteration in the first
        # pre-backward hook to run, which iterates over this module's
        # parameters and adds an entry to mapping parameters with
        # `requires_grad=True` to `False` to indicate to wait for their
        # gradient to be ready before running the post-backward hook logic
        self._param_to_is_grad_ready = None

        # Save the `AccumulateGrad` objects to ensure the registered post-hooks
        # run (or else new `AccumulateGrad` objects without the hooks may be
        # created); use a set to avoid the data structure from using increasing
        # memory over the iterations
        self.grad_accs: Set[torch.AccumulateGrad] = set()

    def flatten(self):
        """Flattens the weight and bias into a flattened parameter and modifies
        their storage to point into the flattened parameter's storage."""
        with torch.no_grad():
            self.flat_param = FlatParameter(torch.cat(
                [p.reshape(-1) for p in self.parameters()], 0,
            ))
            views = torch.split(
                self.flat_param, list(p.numel() for p in self.parameters()),
                dim=0,
            )
            for param, view in zip(self.parameters(), views):
                param.data = view.view(param.shape)
        for param in self.parameters():
            param.requires_grad_(True)

    def forward(self, x):
        z = self.linear0(x)
        z = self.relu(z)
        z = self.linear1(z)
        z = self.relu(z)
        return z

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
            if self.flat_param.grad is None:
                # Must be initialized as zero for mathematical correctness
                self.flat_param.grad = torch.zeros_like(self.flat_param)
            assert len(self.flat_param.grad.shape) == 1
            # Set the original parameters' `.grad` as views into the flattened
            # gradient
            offset = 0
            for param in self.parameters():
                param.grad = torch.narrow(
                    self.flat_param.grad, 0, offset, param.numel(),
                ).view(param.shape)
                offset += param.numel()

        for param_name, param in self.named_parameters():
            param.register_hook(functools.partial(pre_bwd_hook, param, param_name))

    def register_post_bwd_hooks(self, verbose: bool = False):
        def post_bwd_hook(param, param_name: str, *unused: Any):
            """Resets internal data for the next iteration if this hook is the
            last one to be called for this module."""
            if verbose:
                print(f"Post bwd hook from {param_name}!")
            self._param_to_is_grad_ready[param] = True
            if all(self._param_to_is_grad_ready.values()):
                if verbose:
                    print(f"\"Reduce scatter\" from {param_name}!")
                self._param_to_is_grad_ready = None  # reset for next iteration

        for n, p in self.named_parameters():
            assert p.requires_grad
            p_tmp = p.expand_as(p)
            grad_acc = p_tmp.grad_fn.next_functions[0][0]
            grad_acc.register_hook(functools.partial(post_bwd_hook, p, n))
            # Must keep a reference to the `AccumulateGrad` object
            self.grad_accs.add(grad_acc)

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
        assert torch.allclose(model1.linear0.weight, model2.linear0.weight)
        assert torch.allclose(model1.linear0.bias, model2.linear0.bias)
        assert torch.allclose(model1.linear1.weight, model2.linear1.weight), \
            f"{model1.linear1.weight}\n{model2.linear1.weight}"
        assert torch.allclose(model1.linear1.bias, model2.linear1.bias)


def check_model_gradients(models: List[torch.nn.Module]):
    for model1, model2 in itertools.combinations(models, 2):
        assert torch.allclose(model1.linear0.weight.grad, model2.linear0.weight.grad)
        assert torch.allclose(model1.linear0.bias.grad, model2.linear0.bias.grad)
        assert torch.allclose(model1.linear1.weight.grad, model2.linear1.weight.grad)
        assert torch.allclose(model1.linear1.bias.grad, model2.linear1.bias.grad)


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
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    verbose = True

    # Model without flattening
    model1 = Model()
    optim1 = torch.optim.Adam(model1.parameters(), lr=LR)
    # Model with flattening, to optimize in terms of original parameters
    model2 = Model()
    model2.flatten()
    model2.register_pre_bwd_hooks(verbose)
    model2.register_post_bwd_hooks(verbose)
    optim2 = torch.optim.Adam(
        [
            model2.linear0.weight,
            model2.linear0.bias,
            model2.linear1.weight,
            model2.linear1.bias,
        ], lr=LR,
    )
    # Model with flattening, to optimize in terms of the flattened parameter
    model3 = Model()
    model3.flatten()
    model3.register_pre_bwd_hooks(verbose)
    model3.register_post_bwd_hooks(verbose)
    optim3 = torch.optim.Adam([model3.flat_param], lr=LR)

    models = [model1, model2, model3]
    optims = [optim1, optim2, optim3]
    inp = torch.randn(8, 5)
    check_model_parameters(models)
    run_iter(models, optims, inp, False)
    run_iter(models, optims, inp, True)
    run_iter(models, optims, inp, True)

    # Check that `.parameters()` returns the original parameters
    assert len(list(model1.parameters())) == 4
    assert len(list(model2.parameters())) == 4
    assert len(list(model3.parameters())) == 4

    # Check that the new `.fsdp_parameters()` returns only flattened parameters
    assert len(list(model1.fsdp_parameters())) == 0
    assert len(list(model2.fsdp_parameters())) == 1
    assert len(list(model3.fsdp_parameters())) == 1

    # Check that `grad_accs` do not grow in size over the iterations
    assert len(model2.grad_accs) == len(list(model2.parameters()))
    assert len(model3.grad_accs) == len(list(model3.parameters()))

    print()
    print("Yay! Gradients and parameters match!")


if __name__ == "__main__":
    main()
