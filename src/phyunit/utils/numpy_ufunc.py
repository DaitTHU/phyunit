try:
    import warnings

    import numpy as np  # type: ignore

    ufuncs = {f for name in dir(np) if isinstance(f := getattr(np, name), np.ufunc)}

    ufunc_names = {f.__name__ for f in ufuncs}

    ufunc_dict: dict[str, set[str]] = {
        # must be dimensionless
        'dimless': {
            'sin', 'cos', 'tan', 'sinh', 'cosh', 'tanh',
            'arcsin', 'arccos', 'arctan', 'arcsinh', 'arccosh', 'arctanh',
            'exp', 'exp2', 'expm1',
            'log', 'log2', 'log10', 'log1p',
            'logaddexp', 'logaddexp2',
            # input must be bool
            'logical_and', 'logical_not', 'logical_or', 'logical_xor',
            # input must be integer
            'invert', 'bitwise_and', 'bitwise_or', 'bitwise_xor', 'bitwise_count',
            'left_shift', 'right_shift',
            'gcd', 'lcm',
            # output must contain integer
            'floor', 'ceil', 'rint', 'trunc',
            'floor_divide', 'remainder', 'fmod', 'modf', 'divmod',
        },
        # output is bool
        'bool': {
            'isfinite', 'isinf', 'isnan', 'isnat',
        },
        # comparison
        'comparison': {
            'equal', 'not_equal',
            'greater', 'greater_equal', 'less', 'less_equal',
        },
        # input dimension must be the same
        'dimsame': {
            'maximum', 'fmax', 'minimum', 'fmin',
            # addition/subtraction
            'add', 'subtract', 'nextafter', 
            # others
            'arctan2', 'hypot', 
        },
        # single input & output unit == input unit
        'preserve': {
            'absolute', 'fabs', 'conjugate', 'positive', 'negative', 
            'spacing', 
        },
        # angle unit conversion
        'angle': {
            'deg2rad', 'degrees', 'rad2deg', 'radians',
        },
        # physical quantity product
        'product': {
            'multiply', 'matmul', 'vecdot', 'vecmat', 'matvec',
        },
        # physical quantity operations
        'nonlinear': {
            'divide', 'reciprocal', 
            'square', 'sqrt', 'cbrt', 
        },
        # others
        'other': {
            'copysign', 'heaviside', 'sign', 'signbit', 
            'power', 'float_power', 'frexp', 'ldexp', 
        },
    }

    if ufunc_inter := set.intersection(*ufunc_dict.values()):
        warnings.warn(f"ufunc duplicates in multiple categories: {ufunc_inter}")
    if ufunc_diff := ufunc_names - set.union(*ufunc_dict.values()):
        warnings.warn(f"ufuncs not categorized: {ufunc_diff}")

except ImportError:
    ufunc_dict: dict[str, set[str]] = {}

