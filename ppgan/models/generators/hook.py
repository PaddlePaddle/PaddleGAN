# code was based on https://github.com/fastai/fastai

import numpy as np

import paddle
import paddle.nn as nn


def is_listy(x):
    return isinstance(x, (tuple, list))


class Hook():
    "Create a hook on `m` with `hook_func`."

    def __init__(self, m, hook_func, is_forward=True, detach=True):
        self.hook_func, self.detach, self.stored = hook_func, detach, None
        f = m.register_forward_post_hook if is_forward else m.register_backward_hook
        self.hook = f(self.hook_fn)
        self.removed = False

    def hook_fn(self, module, input, output):
        "Applies `hook_func` to `module`, `input`, `output`."
        if self.detach:
            input = (o.detach()
                     for o in input) if is_listy(input) else input.detach()
            output = (o.detach()
                      for o in output) if is_listy(output) else output.detach()
        self.stored = self.hook_func(module, input, output)

    def remove(self):
        "Remove the hook from the model."
        if not self.removed:
            self.hook.remove()
            self.removed = True

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.remove()


class Hooks():
    "Create several hooks on the modules in `ms` with `hook_func`."

    def __init__(self, ms, hook_func, is_forward=True, detach=True):
        self.hooks = []
        try:
            for m in ms:
                self.hooks.append(Hook(m, hook_func, is_forward, detach))
        except Exception as e:
            pass

    def __getitem__(self, i: int) -> Hook:
        return self.hooks[i]

    def __len__(self) -> int:
        return len(self.hooks)

    def __iter__(self):
        return iter(self.hooks)

    @property
    def stored(self):
        return [o.stored for o in self]

    def remove(self):
        "Remove the hooks from the model."
        for h in self.hooks:
            h.remove()

    def __enter__(self, *args):
        return self

    def __exit__(self, *args):
        self.remove()


def _hook_inner(m, i, o):
    return o if isinstance(
        o, paddle.static.Variable) else o if is_listy(o) else list(o)


def hook_output(module, detach=True, grad=False):
    "Return a `Hook` that stores activations of `module` in `self.stored`"
    return Hook(module, _hook_inner, detach=detach, is_forward=not grad)


def hook_outputs(modules, detach=True, grad=False):
    "Return `Hooks` that store activations of all `modules` in `self.stored`"
    return Hooks(modules, _hook_inner, detach=detach, is_forward=not grad)


def model_sizes(m, size=(64, 64)):
    "Pass a dummy input through the model `m` to get the various sizes of activations."
    with hook_outputs(m) as hooks:
        x = dummy_eval(m, size)
        return [o.stored.shape for o in hooks]


def dummy_eval(m, size=(64, 64)):
    "Pass a `dummy_batch` in evaluation mode in `m` with `size`."
    m.eval()
    return m(dummy_batch(size))


def dummy_batch(size=(64, 64), ch_in=3):
    "Create a dummy batch to go through `m` with `size`."
    arr = np.random.rand(1, ch_in, *size).astype('float32') * 2 - 1
    return paddle.to_tensor(arr)
