class AffineLayout:
    def __init__(self, strides=None):
        self.strides = strides

class LinearLayout:
    def __init__(self, matrix):
        self.matrix = matrix

class AffineLinearLayout:
    def __init__(self, matrix, offset):
        self.matrix = matrix
        self.offset = offset

def RowMajor():
    return AffineLayout(strides=None)

def ColMajor():
    return AffineLayout(strides=None)
