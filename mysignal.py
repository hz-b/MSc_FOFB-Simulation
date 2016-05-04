
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import sympy as sy

def poly_to_sympy(num, den, symbol='s', simplify=True):
    """ Convert Scipy's LTI instance to Sympy expression """
    s = sy.Symbol(symbol)
    G = sy.Poly(num, s) / sy.Poly(den, s)
    return sy.simplify(G) if simplify else G

def poly_from_sympy(xpr, symbol='s'):
    """ Convert Sympy transfer function polynomial to Scipy LTI """
    s = sy.Symbol(symbol)
    num, den = sy.simplify(xpr).as_numer_denom()  # expressions
    p_num_den = sy.poly(num, s), sy.poly(den, s)  # polynomials
    c_num_den = [sy.expand(p).all_coeffs() for p in p_num_den]  # coefficients
    l_num, l_den = [sy.lambdify((), c)() for c in c_num_den]  # convert to floats
    return l_num, l_den

class TF(signal.TransferFunction):
    def __init__(self, num, den):
        super().__init__(num, den)

    def __neg__(self):
        return TF(-self.num,self.den)

    def __mul__(self, other):
        self_s = self.to_sympy()
        if type(other) in [int, float]:
            other_s = other
        else:
            other_s = other.to_sympy()
        return TF.from_sympy(self_s * other_s)

    def __truediv__(self, other):
        self_s = self.to_sympy()
        if type(other) in [int, float]:
            other_s = other
        else:
            other_s = other.to_sympy()
        return TF.from_sympy(self_s / other_s)

    def __rtruediv__(self, other):
        self_s = self.to_sympy()
        if type(other) in [int, float]:
            other_s = other
        else:
            other_s = other.to_sympy()
        return TF.from_sympy(other_s / self_s)

    def __add__(self,other):
        self_s = self.to_sympy()
        if type(other) in [int, float]:
            other_s = other
        else:
            other_s = other.to_sympy()
        return TF.from_sympy(self_s + other_s)

    def __sub__(self,other):
        self_s = self.to_sympy()
        if type(other) in [int, float]:
            other_s = other
        else:
            other_s = other.to_sympy()
        return TF.from_sympy(self_s - other_s)

    def __rsub__(self,other):
        self_s = self.to_sympy()
        if type(other) in [int, float]:
            other_s = other
        else:
            other_s = other.to_sympy()
        return TF.from_sympy(other_s - self_s)

    # symmetric behaviour for commutative operators
    __rmul__ = __mul__
    __radd__ = __add__

    def to_sympy(self, symbol='s', simplify=True):
        """ Convert Scipy's LTI instance to Sympy expression """
        return poly_to_sympy(self.num, self.den, 's', simplify)

    def from_sympy(xpr, symbol='s'):
        """ Convert Sympy transfer function polynomial to Scipy LTI """
        num, den = poly_from_sympy(xpr, symbol)
        return TF(num, den)

    def as_poly_s(self):
        return self.to_sympy()

    def as_poly_z(self, Ts):
        [numz], denz, _ = signal.cont2discrete((self.num, self.den), Ts,
                                               method='bilinear')
        return poly_to_sympy(numz, denz, 'z')

    def apply_f(self, inputs, outputs, Ts):
        [num], den, dt = signal.cont2discrete((self.num, self.den), Ts,
                                              method='bilinear')

        Nn = num.size
        Nd = den.size - 1

        x = np.zeros(Nn)
        y = np.zeros(Nd)

        if Nn >= inputs.size:
            x[:inputs.size] = np.flipud(inputs)
        else:
            x = np.flipud(inputs[-Nn:])

        if Nd >= outputs.size:
            y[:outputs.size] = np.flipud(outputs)
        else:
            y = np.flipud(outputs[-Nd:])

        return den[0]*(sum(num*x) - sum(den[1:]*y))

    def plotHw(self, w=None, ylabel=None):
        w, H = self.freqresp(w)
        fig = plt.figure()
        plt.subplot(2, 1, 1)
        plt.plot(w/(2*np.pi), np.abs(H))
        plt.xscale('log')
        plt.yscale('log')
        plt.grid(which="both")
        plt.xlabel("Frequency [in Hz]")
        plt.ylabel(ylabel if ylabel is not None else "Amplitude")

        plt.subplot(2, 1, 2)
        plt.plot(w/(2*np.pi), np.unwrap(np.angle(H)))
        plt.xscale('log')
        plt.grid(which="both")
        plt.xlabel("Frequency [in Hz]")
        plt.ylabel("Phase")
        fig.subplots_adjust(hspace=.5)


class TransferFunction_1stOrder(TF):
    def __init__(self, A, a):
        super().__init__([A], [1, a])
        self.A = A
        self.a = a


    def apply_f(self, xk_0, xk_1, yk_1, Ts):
        """
            H(s) = A / (1 + a*s)
            H(z) = A / (1 + a * 2/Ts * [z-1]/[z+1])
                 = A * (z+1) / ( [z+1] + 2*a/Ts * [z-1] )
                 = A * (z+1) / ( [1 - 2*a/Ts] + [1 + 2*a/Ts]*z )

            H(z) = A * (1 + z^{-1}) / ( [1 + 2*a/Ts] + [1 - 2*a/Ts] * z^{-1} )

            y = H(z) * x
            (1 + 2*a/Ts) * y[k] + (1 - 2*a/Ts) * y[k-1] = A * (x[k] + x[k-1])
            y[k] = ( A * (x[k] + x[k-1]) - (1 - 2*a/Ts) * y[k-1] ) / (1 + 2*a/Ts)
        """
        y = (self.A * (xk_0 + xk_1) - (1 - 2*self.a/Ts) * yk_1 ) / (1 + 2*self.a/Ts)
        return y


class PID(TF):
    def __init__(self, P, I, D):
        tf = TF([P],[1])
        if I != 0:
            tf += TF([I],[1,0])
        if D != 0:
            tau_d = P/D
            tf += TF([D, 0],[tau_d/8, 1])
        super().__init__(tf.num, tf.den)

        self.kP = P
        self.kI = I
        self.kD = D

    def apply_f(self, e):
        return self.kP*e[-1] + self.kD*(e[-1]-e[-2]) + self.kI*sum(e)
