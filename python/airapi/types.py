class DType:
    def __init__(self, name, itemsize):
        self.name = name
        self.itemsize = itemsize
    def __repr__(self):
        return f"airapi.{self.name}"

bf16  = DType("bf16",  2)
f32   = DType("f32",   4)
f16   = DType("f16",   2)
i8    = DType("i8",    1)
i4    = DType("i4",    0.5)
e8m0  = DType("e8m0",  1)
fp8   = DType("fp8",   1)
