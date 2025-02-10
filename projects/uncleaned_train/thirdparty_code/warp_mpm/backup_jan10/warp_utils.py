import warp as wp
import ctypes
from typing import Optional

from warp.torch import (
    dtype_from_torch,
    device_from_torch,
    dtype_is_compatible,
    from_torch,
)


def from_torch_safe(t, dtype=None, requires_grad=None, grad=None):
    """Wrap a PyTorch tensor to a Warp array without copying the data.

    Args:
        t (torch.Tensor): The torch tensor to wrap.
        dtype (warp.dtype, optional): The target data type of the resulting Warp array. Defaults to the tensor value type mapped to a Warp array value type.
        requires_grad (bool, optional): Whether the resulting array should wrap the tensor's gradient, if it exists (the grad tensor will be allocated otherwise). Defaults to the tensor's `requires_grad` value.

    Returns:
        warp.array: The wrapped array.
    """
    if dtype is None:
        dtype = dtype_from_torch(t.dtype)
    elif not dtype_is_compatible(t.dtype, dtype):
        raise RuntimeError(f"Incompatible data types: {t.dtype} and {dtype}")

    # get size of underlying data type to compute strides
    ctype_size = ctypes.sizeof(dtype._type_)

    shape = tuple(t.shape)
    strides = tuple(s * ctype_size for s in t.stride())

    # if target is a vector or matrix type
    # then check if trailing dimensions match
    # the target type and update the shape
    if hasattr(dtype, "_shape_"):
        dtype_shape = dtype._shape_
        dtype_dims = len(dtype._shape_)
        if dtype_dims > len(shape) or dtype_shape != shape[-dtype_dims:]:
            raise RuntimeError(
                f"Could not convert Torch tensor with shape {shape} to Warp array with dtype={dtype}, ensure that source inner shape is {dtype_shape}"
            )

        # ensure the inner strides are contiguous
        stride = ctype_size
        for i in range(dtype_dims):
            if strides[-i - 1] != stride:
                raise RuntimeError(
                    f"Could not convert Torch tensor with shape {shape} to Warp array with dtype={dtype}, because the source inner strides are not contiguous"
                )
            stride *= dtype_shape[-i - 1]

        shape = tuple(shape[:-dtype_dims]) or (1,)
        strides = tuple(strides[:-dtype_dims]) or (ctype_size,)

    requires_grad = t.requires_grad if requires_grad is None else requires_grad
    if grad is not None:
        if not isinstance(grad, wp.array):
            import torch

            if isinstance(grad, torch.Tensor):
                grad = from_torch(grad, dtype=dtype)
            else:
                raise ValueError(f"Invalid gradient type: {type(grad)}")
    elif requires_grad:
        # wrap the tensor gradient, allocate if necessary
        if t.grad is None:
            # allocate a zero-filled gradient tensor if it doesn't exist
            import torch

            t.grad = torch.zeros_like(t, requires_grad=False)
        grad = from_torch(t.grad, dtype=dtype)

    a = wp.types.array(
        ptr=t.data_ptr(),
        dtype=dtype,
        shape=shape,
        strides=strides,
        device=device_from_torch(t.device),
        copy=False,
        owner=False,
        grad=grad,
        requires_grad=requires_grad,
    )

    # save a reference to the source tensor, otherwise it will be deallocated
    a._tensor = t
    return a


class MyTape(wp.Tape):
    # returns the adjoint of a kernel parameter
    def get_adjoint(self, a):
        if not wp.types.is_array(a) and not isinstance(a, wp.codegen.StructInstance):
            # if input is a simple type (e.g.: float, vec3, etc) then
            # no gradient needed (we only return gradients through arrays and structs)
            return a

        elif wp.types.is_array(a) and a.grad:
            # keep track of all gradients used by the tape (for zeroing)
            # ignore the scalar loss since we don't want to clear its grad
            self.gradients[a] = a.grad
            return a.grad

        elif isinstance(a, wp.codegen.StructInstance):
            adj = a._cls()
            for name, _ in a._cls.ctype._fields_:
                if name.startswith("_"):
                    continue
                if isinstance(a._cls.vars[name].type, wp.array):
                    arr = getattr(a, name)
                    if arr is None:
                        continue
                    if arr.grad:
                        grad = self.gradients[arr] = arr.grad
                    else:
                        grad = wp.zeros_like(arr)
                    setattr(adj, name, grad)
                else:
                    setattr(adj, name, getattr(a, name))

            self.gradients[a] = adj
            return adj

        return None


# from https://github.com/PingchuanMa/NCLaw/blob/main/nclaw/warp/tape.py
class CondTape(object):
    def __init__(self, tape: Optional[MyTape], cond: bool = True) -> None:
        self.tape = tape
        self.cond = cond

    def __enter__(self):
        if self.tape is not None and self.cond:
            self.tape.__enter__()

    def __exit__(self, exc_type, exc_value, traceback):
        if self.tape is not None and self.cond:
            self.tape.__exit__(exc_type, exc_value, traceback)