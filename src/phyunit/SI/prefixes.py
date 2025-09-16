from ..utils.constclass import ConstClass

__all__ = ['prefix']


class prefix(ConstClass):
    '''The `prefix` constclass offers all prefix factors, 
    which are just numbers.
    '''

    # SI prefixes

    quetta = Q = 1e30
    '''quetta, 10^30'''
    ronna = R = 1e27
    '''ronna, 10^27'''
    yotta = Y = 1e24
    '''yotta, 10^24'''
    zetta = Z = 1e21
    '''zetta, 10^21'''
    exa = E = 1e18
    '''exa, 10^18'''
    peta = P = 1e15
    '''peta, 10^15'''
    tera = T = 1e12
    '''tera, 10^12'''
    giga = G = 1e9
    '''giga, 10^9'''
    mega = M = 1e6
    '''mega, 10^6'''
    kilo = k = 1e3
    '''kilo, 10^3'''
    hecto = h = 1e2
    '''hecto, 10^2'''
    deca = da = 1e1
    '''deca, 10^1'''
    deci = d = 1e-1
    '''deci, 10^-1'''
    centi = c = 1e-2
    '''centi, 10^-2'''
    milli = m = 1e-3
    '''milli, 10^-3'''
    micro = u = 1e-6
    '''micro, 10^-6'''
    nano = n = 1e-9
    '''nano, 10^-9'''
    pico = p = 1e-12
    '''pico, 10^-12'''
    femto = f = 1e-15
    '''femto, 10^-15'''
    atto = a = 1e-18
    '''atto, 10^-18'''
    zepto = z = 1e-21
    '''zepto, 10^-21'''
    yocto = y = 1e-24
    '''yocto, 10^-24'''
    ronto = r = 1e-27
    '''ronto, 10^-27'''
    quecto = q = 1e-30
    '''quecto, 10^-30'''

    # binary prefixes

    kibi = ki = 2**10
    '''kibi, 2^10'''
    mebi = Mi = 2**20
    '''mebi, 2^20'''
    gibi = Gi = 2**30
    '''gibi, 2^30'''
    tebi = Ti = 2**40
    '''tebi, 2^40'''
    pebi = Pi = 2**50
    '''pebi, 2^50'''
    exbi = Ei = 2**60
    '''exbi, 2^60'''
    zebi = Zi = 2**70
    '''zebi, 2^70'''
    yobi = Yi = 2**80
    '''yobi, 2^80'''
