try:
    import warnings

    import numpy as np  # type: ignore

    ufuncs = {f for name in dir(np) if isinstance(f := getattr(np, name), np.ufunc)}

    ufunc_dict: dict[str, set[np.ufunc]] = {
        # must be dimensionless
        'dimless': {
            np.sin, np.cos, np.tan, np.sinh, np.cosh, np.tanh,
            np.arcsin, np.arccos, np.arctan, np.arcsinh, np.arccosh, np.arctanh, 
            np.exp, np.exp2, np.expm1, 
            np.log, np.log2, np.log10, np.log1p, 
            np.logaddexp, np.logaddexp2, 
            # input must be bool
            np.logical_and, np.logical_not, np.logical_or, np.logical_xor,
            # input must be integer
            np.invert, np.bitwise_and, np.bitwise_or, np.bitwise_xor, np.left_shift, np.right_shift, 
            np.gcd, np.lcm,
            # output must contain integer
            np.floor, np.ceil, np.rint, np.trunc,
            np.floor_divide, np.remainder, np.fmod, np.modf, np.divmod
        },
        # output is bool
        'bool': {
            np.isfinite, np.isinf, np.isnan, np.isnat
        },
        # comparison
        'comparison': {
            np.equal, np.not_equal,
            np.greater, np.greater_equal, np.less, np.less_equal,
        },
        # input dimension must be the same
        'dimsame': {
            np.maximum, np.fmax, np.minimum, np.fmin,
            # addition/subtraction
            np.add, np.subtract, np.nextafter, 
            # others
            np.arctan2, np.hypot, 
        },
        # single input & output unit == input unit
        'preserve': {
            np.absolute, np.fabs, np.conjugate, np.positive, np.negative, 
            np.spacing, 
        },
        # angle unit conversion
        'angle': {
            np.deg2rad, np.degrees, np.rad2deg, np.radians
        },
        # physical quantity operations
        'nonlinear': {
            np.multiply, np.matmul, np.true_divide, np.reciprocal, 
            np.square, np.sqrt, np.cbrt, 
        },
        # others
        'other': {
            np.copysign, np.heaviside, np.sign, np.signbit, 
            np.power, np.float_power, np.frexp, np.ldexp, 
        },
    }

    if ufunc_inter := set.intersection(*ufunc_dict.values()):
        warnings.warn(f"ufunc duplicates in multiple categories: {ufunc_inter}")
    if ufunc_diff := ufuncs - set.union(*ufunc_dict.values()):
        warnings.warn(f"ufuncs not categorized: {ufunc_diff}")

except ImportError:
    pass

