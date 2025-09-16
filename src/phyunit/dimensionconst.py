from .utils.constclass import ConstClass

from .dimension import Dimension, DIMENSIONLESS

class DimensionConst(ConstClass):
    '''
    `DimensionConst` is a class containing constant `Dimension` objects
    objects, like dimensionless, 7 SI base units, and other derived units.
    Units with different physical meanings sharing the same dimension,
    like energy and work, have to share the same name `ENERGY`.
    This constraint exists because dimensions are mathematical abstractions
    that represent physical quantities, and multiple quantities with the same
    dimensional formula (e.g., energy and work) are treated as equivalent
    in dimensional analysis. This ensures consistency in calculations and
    avoids ambiguity when performing operations involving these quantities.

    Example usage:
        >>> print(DimensionConst.VELOCITY)
        T⁻¹L

        >>> print(DimensionConst.FORCE)
        T⁻²LM
    '''
    DIMENSIONLESS = DIMENSIONLESS  # default singleton
    # 7 eigen SI base units
    TIME = Dimension(T=1)
    LENGTH = Dimension(L=1)
    MASS = Dimension(M=1)
    ELECTRIC_CURRENT = Dimension(I=1)
    THERMODYNAMIC_TEMPERATURE = Dimension(Theta=1)
    AMOUNT_OF_SUBSTANCE = Dimension(N=1)
    LUMINOUS_INTENSITY = Dimension(J=1)
    # straight derived
    ANGLE = PHASE_ANGLE = PLANE_ANGLE = SOLID_ANGLE = DIMENSIONLESS
    WAVENUMBER = 1 / LENGTH
    AREA = LENGTH**2
    VOLUME = LENGTH**3
    FREQUENCY = 1 / TIME
    ACTIVITY = FREQUENCY  # of a radionuclide
    TEMPERATURE = THERMODYNAMIC_TEMPERATURE
    # kinematics and dynamic
    VELOCITY = LENGTH / TIME
    ACCELERATION = VELOCITY / TIME
    FORCE = MASS * ACCELERATION
    WEIGHT = FORCE
    PRESSURE = FORCE / AREA
    STRESS = PRESSURE
    ENERGY = FORCE * LENGTH
    WORK = HEAT = ENERGY
    POWER = ENERGY / TIME
    RADIANT_FLUX = POWER
    MOMENTUM = MASS * VELOCITY
    DYNAMIC_VISCOSITY = PRESSURE * TIME
    KINEMATIC_VISCOSITY = AREA / TIME
    # electrodynamics
    ELECTRIC_CHARGE = ELECTRIC_CURRENT * TIME
    VOLTAGE = POWER / ELECTRIC_CURRENT
    ELECTRIC_POTENTIAL = ELECTROMOTIVE_FORCE = VOLTAGE
    CAPACITANCE = ELECTRIC_CHARGE / VOLTAGE
    RESISTANCE = VOLTAGE / ELECTRIC_CURRENT
    IMPEDANCE = REACTANCE = RESISTANCE
    CONDUCTANCE = 1 / RESISTANCE
    MAGNETIC_FLUX = VOLTAGE * TIME
    MAGNETIC_FLUX_DENSITY = MAGNETIC_FLUX / AREA
    MAGNETIC_INDUCTION = MAGNETIC_FLUX_DENSITY
    MAGNETIC_FIELD_STRENGTH = ELECTRIC_CURRENT / LENGTH
    INDUCTANCE = MAGNETIC_FLUX / ELECTRIC_CURRENT
    # luminous
    LUMINOUS_FLUX = LUMINOUS_INTENSITY * SOLID_ANGLE
    ILLUMINANCE = LUMINOUS_FLUX / AREA
    LUMINANCE = LUMINOUS_INTENSITY / AREA
    # nuclear radiation
    KERMA = ENERGY / MASS
    ABSORBED_DOSE = EQUIVALENT_DOSE = KERMA  # of ionising radiation
    EXPOSURE = ELECTRIC_CHARGE / MASS  # X-ray and γ-ray
    # chemistry
    CATALYTIC_ACTIVITY = AMOUNT_OF_SUBSTANCE / TIME