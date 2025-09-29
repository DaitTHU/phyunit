from .dimension import DIMENSIONLESS, Dimension
from .utils.constclass import ConstClass


class DimensionConst(ConstClass):
    '''
    use this class to access commonly used `Dimension` constants.

    >>> print(DimensionConst.VELOCITY)
    T⁻¹L

    >>> print(DimensionConst.FORCE)
    T⁻²LM

    NOTE
    different physical quantities may have the same dimension,
    like _energy_ and _work_.

    Be careful when using these dimensions,
    especially in `set` or as keys in `dict`:

    >>> {DimensionConst.ENERGY: 1, DimensionConst.WORK: 2}  # same key!
    {Dimension(T=-2, L=2, M=1): 2}
    '''

    DIMENSIONLESS = DIMENSIONLESS

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

    