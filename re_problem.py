import numpy as np
from pymoo.core.problem import Problem


def obtain_problem(name, *args, **kwargs):
    name = name.lower()

    PROBLEM = {
        're21': RE21,
        're22': RE22,
        're23': RE23,
        're24': RE24,
        're25': RE25,
        're31': RE31,
        're32': RE32,
        're33': RE33,
        're34': RE34,
        're35': RE35,
        're36': RE36,
        're37': RE37,
        're41': RE41,
        're42': RE42,
        're61': RE61,
        're91': RE91
    }

    if name not in PROBLEM:
        raise Exception("Problem not found.")

    return PROBLEM[name](*args, **kwargs)


def div(x1, x2):
    '''
    Divide x1 / x2, return 0 where x2 == 0

    '''
    results = x1 * 0.0
    results[x2 != 0.0] = x1[x2 != 0.0] / x2[x2 != 0.0]

    return results

class RE21(Problem):
    def __init__(self):
        self.ideal = np.array([1237.8414230005742, 0.002761423749158419])
        self.nadir = np.array([2886.3695604236013, 0.039999999999998245])

        F = 10.0
        sigma = 10.0
        tmp_val = F / sigma

        self.n_dim = 4
        self.n_obj = 2
        self.lbound = np.array([tmp_val, np.sqrt(2.0) * tmp_val, np.sqrt(2.0) * tmp_val, tmp_val])
        self.ubound = np.ones(self.n_dim) * 3 * tmp_val

        xl = np.zeros(self.n_dim)
        xu = np.ones(self.n_dim)

        super().__init__(n_var=4, n_obj=2, xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        X = X * (self.ubound - self.lbound) + self.lbound
        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]
        x4 = X[:, 3]

        F = 10.0
        E = 2.0 * 1e5
        L = 200.0

        f1 = L * (2 * x1 + np.sqrt(2.0) * x2 + np.sqrt(x3) + x4)  

        term1 = 2.0 / x1
        term2 = 2.0 * np.sqrt(2.0) / x2
        term3 = -2.0 * np.sqrt(2.0) / x3  
        term4 = 2.0 / x4
        f2 = (F * L / E) * (term1 + term2 + term3 + term4)  
        f_normalized = np.column_stack([f1, f2])
        f_norm = (f_normalized - self.ideal) / (self.nadir - self.ideal)
        out["F"] = f_norm

class RE22(Problem):
    def __init__(self):

        self.ideal = np.array([5.88, 0.0])
        self.nadir = np.array([361.262944647, 180.01547])
        self.n_dim = 3
        self.n_obj = 2
        self.lbound = np.array([0.2, 0.0, 0.0], dtype=np.float64)
        self.ubound = np.array([15, 20, 40], dtype=np.float64)

        xl = np.zeros(self.n_dim)
        xu = np.ones(self.n_dim)
        super().__init__(n_var=3, n_obj=2, xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        X = X * (self.ubound - self.lbound) + self.lbound
        feasible_vals = np.array([
            0.20, 0.31, 0.40, 0.44, 0.60, 0.62, 0.79, 0.80, 0.88, 0.93, 1.0,
            1.20, 1.24, 1.32, 1.40, 1.55, 1.58, 1.60, 1.76, 1.80, 1.86, 2.0,
            2.17, 2.20, 2.37, 2.40, 2.48, 2.60, 2.64, 2.79, 2.80, 3.0, 3.08,
            3.10, 3.16, 3.41, 3.52, 3.60, 3.72, 3.95, 3.96, 4.0, 4.03, 4.20,
            4.34, 4.40, 4.65, 4.74, 4.80, 4.84, 5.0, 5.28, 5.40, 5.53, 5.72,
            6.0, 6.16, 6.32, 6.60, 7.11, 7.20, 7.80, 7.90, 8.0, 8.40, 8.69,
            9.0, 9.48, 10.27, 11.0, 11.06, 11.85, 12.0, 13.0, 14.0, 15.0
        ])

        idx_arr = [np.abs(np.asarray(feasible_vals) - x0).argmin() for x0 in X[:,0]]
        x1_feasible = np.array([feasible_vals[idx] for idx in idx_arr])

        x2 = X[:, 1]
        x3 = X[:, 2]


        f1 = 29.4 * x1_feasible + 0.6 * x2 * x3


        g1 = (x1_feasible * x3) - 7.735 * div(x1_feasible * x1_feasible, x2) - 180.0
        g2 = 4.0 - div(x3, x2)
        violation_g1 = np.where(g1 < 0, -g1, 0)
        violation_g2 = np.where(g2 < 0, -g2, 0)
        f2 = violation_g1 + violation_g2
        f = np.column_stack([f1, f2])
        f_norm = (f - self.ideal) / (self.nadir - self.ideal)
        out["F"] = f_norm

class RE23(Problem):
    def __init__(self):
        self.n_dim = 4
        self.n_obj = 2
        self.lbound = np.array([1, 1, 10, 10], dtype=np.float64)
        self.ubound = np.array([100, 100, 200, 240],dtype=np.float64)

        self.ideal = np.array([15.9018007813, 0.0])
        self.nadir = np.array([5852.05896876, 1288669.78054])
        xl = np.zeros(self.n_dim)
        xu = np.ones(self.n_dim)
        super().__init__(n_var=4, n_obj=2, xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        X = X * (self.ubound - self.lbound) + self.lbound
        x1 = 0.0625 * np.round(X[:, 0]).astype(np.int32)
        x2 = 0.0625 * np.round(X[:, 1]).astype(np.int32)
        x3 = X[:, 2]  
        x4 = X[:, 3]  

        f1 = (
                0.6224 * x1 * x3 * x4
                + 1.7781 * x2 * x3 ** 2
                + 3.1661 * x1 ** 2 * x4
                + 19.84 * x1 ** 2 * x3
        )

        g1 = x1 - 0.0193 * x3  
        g2 = x2 - 0.00954 * x3  
        g3 = (np.pi * x3 ** 2 * x4
              + (4 / 3) * np.pi * x3 ** 3
              - 1296000)  

        g1 = np.where(g1 < 0, -g1, 0)
        g2 = np.where(g2 < 0, -g2, 0)
        g3 = np.where(g3 < 0, -g3, 0)

        f2 = g1 + g2 + g3

        f = np.column_stack([f1, f2])
        f_norm = (f - self.ideal) / (self.nadir - self.ideal)
        out["F"] = f_norm

class RE24(Problem):
    def __init__(self):
        self.ideal = np.array([60.5, 0.0])  
        self.nadir = np.array([481.608088535, 44.2819047619])  

        self.n_dim = 2
        self.n_obj = 2
        self.lbound = np.ones(self.n_dim).astype(np.float64) * 0.5
        self.ubound = np.array([4, 50], dtype=np.float64)

        xl = np.zeros(self.n_dim)
        xu = np.ones(self.n_dim)

        super().__init__(n_var=2, n_obj=2, n_ieq_constr=0, xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        X = X * (self.ubound - self.lbound) + self.lbound
        x1 = X[:, 0]
        x2 = X[:, 1]

        g = np.zeros((len(X), 4))

        f1 = x1 + 120 * x2

        E = 700000
        sigma_b_max = 700
        tau_max = 450
        delta_max = 1.5

        sigma_k = (E * x1 * x1) / 100
        sigma_b = 4500 / (x1 * x2)
        tau = 1800 / x2
        delta = (56.2 * 10000) / (E * x1 * x2 * x2)

        g[:,0] = 1 - (sigma_b / sigma_b_max)
        g[:,1] = 1 - (tau / tau_max)
        g[:,2] = 1 - (delta / delta_max)
        g[:,3] = 1 - (sigma_b / sigma_k)
        g = np.where(g < 0, -g, 0)
        f2 = g[:,0] + g[:,1] + g[:,2] + g[:,3]
        f = np.column_stack([f1, f2])
        f_norm = (f - self.ideal) / (self.nadir - self.ideal)
        out["F"] = f_norm

class RE25(Problem):
    def __init__(self):
        self.feasible_vals = np.array([
            0.009, 0.0095, 0.0104, 0.0118, 0.0128, 0.0132, 0.014, 0.015, 0.0162,
            0.0173, 0.018, 0.02, 0.023, 0.025, 0.028, 0.032, 0.035, 0.041, 0.047,
            0.054, 0.063, 0.072, 0.08, 0.092, 0.105, 0.12, 0.135, 0.148, 0.162,
            0.177, 0.192, 0.207, 0.225, 0.244, 0.263, 0.283, 0.307, 0.331, 0.362,
            0.394, 0.4375, 0.5
        ])

        self.ideal = np.array([0.037591349242869145, 0.0])
        self.nadir = np.array([0.40397042546, 2224669.22419])

        self.n_dim = 3
        self.n_obj = 2
        self.lbound = np.array([1, 0.6, 0.09], dtype=np.float64)
        self.ubound = np.array([70, 30, 0.5], dtype=np.float64)

        xl = np.zeros(self.n_dim)
        xu = np.ones(self.n_dim)

        super().__init__(n_var=3, n_obj=2, xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        X = X * (self.ubound - self.lbound) + self.lbound
        n_sub = len(X)
        f = np.zeros((n_sub, self.n_obj))
        g = np.zeros((n_sub, 6))
        x1 = np.round(X[:, 0])
        x2 = X[:, 1]

        # Reference: getNearestValue_sample2.py (https://gist.github.com/icchi-h/1d0bb1c52ebfdd31f14b3e811328390a)
        idx_array = np.array([np.abs(np.asarray(self.feasible_vals) - x_temp).argmin() for x_temp in X[:, 2]])

        x3 = np.array([self.feasible_vals[idx] for idx in idx_array])

        # first original objective function
        f[:, 0] = (np.pi * np.pi * x2 * x3 * x3 * (x1 + 2)) / 4.0

        # constraint functions
        Cf = ((4.0 * (x2 / x3) - 1) / (4.0 * (x2 / x3) - 4)) + (0.615 * x3 / x2)
        Fmax = 1000.0
        S = 189000.0
        G = 11.5 * 1e+6
        K = (G * x3 * x3 * x3 * x3) / (8 * x1 * x2 * x2 * x2)
        lmax = 14.0
        lf = (Fmax / K) + 1.05 * (x1 + 2) * x3
        dmin = 0.2
        Dmax = 3
        Fp = 300.0
        sigmaP = Fp / K
        sigmaPM = 6
        sigmaW = 1.25

        g[:, 0] = -((8 * Cf * Fmax * x2) / (np.pi * x3 * x3 * x3)) + S
        g[:, 1] = -lf + lmax
        g[:, 2] = -3 + (x2 / x3)
        g[:, 3] = -sigmaP + sigmaPM
        g[:, 4] = -sigmaP - ((Fmax - Fp) / K) - 1.05 * (x1 + 2) * x3 + lf
        g[:, 5] = sigmaW - ((Fmax - Fp) / K)

        g = np.where(g < 0, -g, 0)
        f[:, 1] = g[:, 0] + g[:, 1] + g[:, 2] + g[:, 3] + g[:, 4] + g[:, 5]

        f_norm = (f - self.ideal) / (self.nadir - self.ideal)

        out["F"] = f_norm

class RE31(Problem):
    def __init__(self):
        self.ideal = np.array([5.53731918799e-05, 0.333333333333, 0.0])
        self.nadir = np.array([500.002668442, 8246211.25124, 19359919.7502])

        self.n_dim = 3
        self.n_obj = 3
        self.lbound = np.array([1e-5, 1e-5, 1], dtype=np.float64)
        self.ubound = np.array([100, 100, 3], dtype=np.float64)

        xl = np.zeros(self.n_dim)
        xu = np.ones(self.n_dim)

        super().__init__(n_var=3, n_obj=3, xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        X = X * (self.ubound - self.lbound) + self.lbound
        n_sub = len(X)
        f = np.zeros((n_sub, self.n_obj))
        g = np.zeros((n_sub, 3))
        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]
        # First original objective function
        f[:, 0] = x1 * np.sqrt(16.0 + (x3 * x3)) + x2 * np.sqrt(1.0 + x3 * x3)
        # Second original objective function
        f[:, 1] = (20.0 * np.sqrt(16.0 + (x3 * x3))) / (x1 * x3)
        # Constraint functions
        g[:, 0] = 0.1 - f[:, 0]
        g[:, 1] = 100000.0 - f[:, 1]
        g[:, 2] = 100000 - ((80.0 * np.sqrt(1.0 + x3 * x3)) / (x3 * x2))
        g = np.where(g < 0, -g, 0)
        f[:, 2] = g[:, 0] + g[:, 1] + g[:, 2]

        f_norm = (f - self.ideal) / (self.nadir - self.ideal)
        out["F"] = f_norm

class RE32(Problem):
    def __init__(self):
        self.ideal = np.array([0.010205496875, 0.00043904, 0.0])
        self.nadir = np.array([37.7831517014, 17561.6, 425062976.628])

        self.n_dim = 4
        self.n_obj = 3
        self.lbound = np.array([0.125, 0.1, 0.1, 0.125], dtype=np.float64)
        self.ubound = np.array([5, 10, 10, 5], dtype=np.float64)

        xl = np.zeros(self.n_dim)
        xu = np.ones(self.n_dim)

        super().__init__(n_var=4, n_obj=3, xl=xl, xu=xu)


    def _evaluate(self, X, out, *args, **kwargs):
        X =X * (self.ubound - self.lbound) + self.lbound
        n_sub = len(X)
        f = np.zeros((n_sub, self.n_obj))
        g = np.zeros((n_sub, 4))
        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]
        x4 = X[:, 3]

        P = 6000
        L = 14
        E = 30 * 1e6

        # // deltaMax = 0.25
        G = 12 * 1e6
        tauMax = 13600
        sigmaMax = 30000
        # First original objective function
        f[:, 0] = (1.10471 * x1 * x1 * x2) + (0.04811 * x3 * x4) * (14.0 + x2)
        # Second original objective function
        f[:, 1] = (4 * P * L * L * L) / (E * x4 * x3 * x3 * x3)
        # Constraint functions
        M = P * (L + (x2 / 2))
        tmpVar = ((x2 * x2) / 4.0) + np.power((x1 + x3) / 2.0, 2)
        R = np.sqrt(tmpVar)
        tmpVar = ((x2 * x2) / 12.0) + np.power((x1 + x3) / 2.0, 2)
        J = 2 * np.sqrt(2) * x1 * x2 * tmpVar

        tauDashDash = (M * R) / J
        tauDash = P / (np.sqrt(2) * x1 * x2)
        tmpVar = tauDash * tauDash + ((2 * tauDash * tauDashDash * x2) / (2 * R)) + (tauDashDash * tauDashDash)
        tau = np.sqrt(tmpVar)
        sigma = (6 * P * L) / (x4 * x3 * x3)
        tmpVar = 4.013 * E * np.sqrt((x3 * x3 * x4 * x4 * x4 * x4 * x4 * x4) / 36.0) / (L * L)
        tmpVar2 = (x3 / (2 * L)) * np.sqrt(E / (4 * G))
        PC = tmpVar * (1 - tmpVar2)

        g[:, 0] = tauMax - tau
        g[:, 1] = sigmaMax - sigma
        g[:, 2] = x4 - x1
        g[:, 3] = PC - P
        g = np.where(g < 0, -g, 0)
        f[:, 2] = g[:, 0] + g[:, 1] + g[:, 2] + g[:, 3]

        f_norm = (f - self.ideal) / (self.nadir - self.ideal)
        out["F"] = f_norm

class RE33(Problem):
    def __init__(self):
        self.ideal = np.array([-0.721525, 1.13907203907, 0.0])
        self.nadir = np.array([5.3067, 3.12833430979, 25.0])

        self.n_dim = 4
        self.n_obj = 3
        self.lbound = np.array([55, 75, 1000, 11], dtype=np.float64)
        self.ubound = np.array([80, 110, 3000, 20], dtype=np.float64)
        xl = np.zeros(self.n_dim)
        xu = np.ones(self.n_dim)

        super().__init__(n_var=4, n_obj=3, xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        X = X * (self.ubound - self.lbound) + self.lbound
        n_sub = len(X)
        f = np.zeros((n_sub, self.n_obj))
        g = np.zeros((n_sub, 4))
        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]
        x4 = X[:, 3]

        # First original objective function
        f[:, 0] = 4.9 * 1e-5 * (x2 * x2 - x1 * x1) * (x4 - 1.0)
        # Second original objective function
        f[:, 1] = ((9.82 * 1e6) * (x2 * x2 - x1 * x1)) / (x3 * x4 * (x2 * x2 * x2 - x1 * x1 * x1))

        # Reformulated objective functions
        g[:, 0] = (x2 - x1) - 20.0
        g[:, 1] = 0.4 - (x3 / (3.14 * (x2 * x2 - x1 * x1)))
        g[:, 2] = 1.0 - (2.22 * 1e-3 * x3 * (x2 * x2 * x2 - x1 * x1 * x1)) / np.power((x2 * x2 - x1 * x1), 2)
        g[:, 3] = (2.66 * 1e-2 * x3 * x4 * (x2 * x2 * x2 - x1 * x1 * x1)) / (x2 * x2 - x1 * x1) - 900.0
        g = np.where(g < 0, -g, 0)
        f[:, 2] = g[:, 0] + g[:, 1] + g[:, 2] + g[:, 3]

        f_norm = (f - self.ideal) / (self.nadir - self.ideal)
        out["F"] = f_norm

class RE34(Problem):
    def __init__(self):
        self.ideal = np.array([1661.7078225, 6.14280000608, 0.0394])
        self.nadir = np.array([1695.2002035, 10.7454, 0.26399999965])

        self.n_dim = 5
        self.n_obj = 3
        self.lbound = np.array([1, 1, 1, 1, 1], dtype=np.float64)
        self.ubound = np.array([3, 3, 3, 3, 3], dtype=np.float64)
        xl = np.zeros(self.n_dim)
        xu = np.ones(self.n_dim)

        super().__init__(n_var=5, n_obj=3, xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        X = X * (self.ubound - self.lbound) + self.lbound
        n_sub = len(X)
        f = np.zeros((n_sub, self.n_obj))
        g = np.zeros((n_sub, 0))
        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]
        x4 = X[:, 3]
        x5 = X[:, 4]

        f[:, 0] = 1640.2823 + (2.3573285 * x1) + (2.3220035 * x2) + (4.5688768 * x3) + (7.7213633 * x4) + (
                    4.4559504 * x5)
        f[:, 1] = 6.5856 + (1.15 * x1) - (1.0427 * x2) + (0.9738 * x3) + (0.8364 * x4) - (0.3695 * x1 * x4) + (
                0.0861 * x1 * x5) + (0.3628 * x2 * x4) - (0.1106 * x1 * x1) - (0.3437 * x3 * x3) + (
                          0.1764 * x4 * x4)
        f[:, 2] = -0.0551 + (0.0181 * x1) + (0.1024 * x2) + (0.0421 * x3) - (0.0073 * x1 * x2) + (0.024 * x2 * x3) - (
                0.0118 * x2 * x4) - (0.0204 * x3 * x4) - (0.008 * x3 * x5) - (0.0241 * x2 * x2) + (0.0109 * x4 * x4)

        f_norm = (f - self.ideal) / (self.nadir - self.ideal)
        out["F"] = f_norm

class RE35(Problem):
    def __init__(self):
        self.ideal = np.array([2352.34611145, 694.233587469, 0.0])
        self.nadir = np.array([6634.56208, 1695.96387746, 397.358927317])

        self.n_dim = 7
        self.n_obj = 3
        self.lbound = np.array([2.6, 0.7, 17, 7.3, 7.3, 2.9, 5.0], dtype=np.float64)
        self.ubound = np.array([3.6, 0.8, 28, 8.3, 8.3, 3.9, 5.5], dtype=np.float64)
        xl = np.zeros(self.n_dim)
        xu = np.ones(self.n_dim)

        super().__init__(n_var=7, n_obj=3, xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        X = X * (self.ubound - self.lbound) + self.lbound
        n_sub = len(X)
        f = np.zeros((n_sub, self.n_obj))
        g = np.zeros((n_sub, 11))
        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = np.round(X[:, 2])
        x4 = X[:, 3]
        x5 = X[:, 4]
        x6 = X[:, 5]
        x7 = X[:, 6]

        # First original objective function (weight)
        f[:, 0] = 0.7854 * x1 * (x2 * x2) * (((10.0 * x3 * x3) / 3.0) + (14.933 * x3) - 43.0934) - 1.508 * x1 * (
                x6 * x6 + x7 * x7) + 7.477 * (x6 * x6 * x6 + x7 * x7 * x7) + 0.7854 * (x4 * x6 * x6 + x5 * x7 * x7)

        # Second original objective function (stress)
        tmpVar = np.power((745.0 * x4) / (x2 * x3), 2.0) + 1.69 * 1e7
        f[:, 1] = np.sqrt(tmpVar) / (0.1 * x6 * x6 * x6)

        # Constraint functions
        g[:, 0] = -(1.0 / (x1 * x2 * x2 * x3)) + 1.0 / 27.0
        g[:, 1] = -(1.0 / (x1 * x2 * x2 * x3 * x3)) + 1.0 / 397.5
        g[:, 2] = -(x4 * x4 * x4) / (x2 * x3 * x6 * x6 * x6 * x6) + 1.0 / 1.93
        g[:, 3] = -(x5 * x5 * x5) / (x2 * x3 * x7 * x7 * x7 * x7) + 1.0 / 1.93
        g[:, 4] = -(x2 * x3) + 40.0
        g[:, 5] = -(x1 / x2) + 12.0
        g[:, 6] = -5.0 + (x1 / x2)
        g[:, 7] = -1.9 + x4 - 1.5 * x6
        g[:, 8] = -1.9 + x5 - 1.1 * x7
        g[:, 9] = -f[:, 1] + 1300.0
        tmpVar = np.power((745.0 * x5) / (x2 * x3), 2.0) + 1.575 * 1e8
        g[:, 10] = -np.sqrt(tmpVar) / (0.1 * x7 * x7 * x7) + 1100.0
        g = np.where(g < 0, -g, 0)
        f[:, 2] = g[:, 0] + g[:, 1] + g[:, 2] + g[:, 3] + g[:, 4] + g[:, 5] + g[:, 6] + g[:, 7] + g[:, 8] + g[:, 9] + g[:, 10]

        f_norm = (f - self.ideal) / (self.nadir - self.ideal)
        out["F"] = f_norm

class RE36(Problem):
    def __init__(self):
        self.ideal = np.array([7.89473684213e-05, 12.0, 0.0])
        self.nadir = np.array([5.931, 56.0, 0.355720675227])

        self.n_dim = 4
        self.n_obj = 3
        self.lbound = np.array([12, 12, 12, 12], dtype=np.float64)
        self.ubound = np.array([60, 60, 60, 60], dtype=np.float64)
        xl = np.zeros(self.n_dim)
        xu = np.ones(self.n_dim)

        super().__init__(n_var=4, n_obj=3, xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        X= X * (self.ubound - self.lbound) + self.lbound
        n_sub = len(X)
        f = np.zeros((n_sub, self.n_obj))
        g = np.zeros((n_sub, 1))
        # all the four variables must be inverger values
        x1 = np.round(X[:, 0]).astype(np.int32)
        x2 = np.round(X[:, 1]).astype(np.int32)
        x3 = np.round(X[:, 2]).astype(np.int32)
        x4 = np.round(X[:, 3]).astype(np.int32)

        f[:, 0] = np.abs(6.931 - ((x3 / x1) * (x4 / x2)))
        f[:, 1] = np.max([x1,x2,x3,x4], axis=0)

        g[:, 0] = 0.5 - (f[:, 0] / 6.931)
        g = np.where(g < 0, -g, 0)
        f[:, 2] = g[:, 0]

        f_norm = (f - self.ideal) / (self.nadir - self.ideal)
        out["F"] = f_norm

class RE37(Problem):
    def __init__(self):
        self.ideal = np.array([0.00889341391106, 0.00488, -0.431499999825])
        self.nadir = np.array([0.98949120096, 0.956587924661, 0.987530948586])

        self.n_dim = 4
        self.n_obj = 3
        self.lbound = np.array([0, 0, 0, 0], dtype=np.float64)
        self.ubound = np.array([1, 1, 1, 1], dtype=np.float64)
        xl = np.zeros(self.n_dim)
        xu = np.ones(self.n_dim)

        super().__init__(n_var=4, n_obj=3, n_ieq_constr=0, xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        X = X * (self.ubound - self.lbound) + self.lbound
        xAlpha = X[:, 0]
        xHA = X[:, 1]
        xOA = X[:, 2]
        xOPTT = X[:, 3]

        f1 = (
            0.692 + 0.477 * xAlpha - 0.687 * xHA - 0.080 * xOA - 0.0650 * xOPTT
            - 0.167 * xAlpha**2 - 0.0129 * xHA * xAlpha + 0.0796 * xHA**2
            - 0.0634 * xOA * xAlpha - 0.0257 * xOA * xHA + 0.0877 * xOA**2
            - 0.0521 * xOPTT * xAlpha + 0.00156 * xOPTT * xHA
            + 0.00198 * xOPTT * xOA + 0.0184 * xOPTT**2
        )

        f2 = (
            0.153 - 0.322 * xAlpha + 0.396 * xHA + 0.424 * xOA + 0.0226 * xOPTT
            + 0.175 * xAlpha**2 + 0.0185 * xHA * xAlpha - 0.0701 * xHA**2
            - 0.251 * xOA * xAlpha + 0.179 * xOA * xHA + 0.0150 * xOA**2
            + 0.0134 * xOPTT * xAlpha + 0.0296 * xOPTT * xHA
            + 0.0752 * xOPTT * xOA + 0.0192 * xOPTT**2
        )

        f3 = (
            0.370 - 0.205 * xAlpha + 0.0307 * xHA + 0.108 * xOA + 1.019 * xOPTT
            - 0.135 * xAlpha**2 + 0.0141 * xHA * xAlpha + 0.0998 * xHA**2
            + 0.208 * xOA * xAlpha - 0.0301 * xOA * xHA - 0.226 * xOA**2
            + 0.353 * xOPTT * xAlpha - 0.0497 * xOPTT * xOA - 0.423 * xOPTT**2
            + 0.202 * xHA * xAlpha**2 - 0.281 * xOA * xAlpha**2
            - 0.342 * xHA**2 * xAlpha - 0.245 * xHA**2 * xOA
            + 0.281 * xOA**2 * xHA - 0.184 * xOPTT**2 * xAlpha
            - 0.281 * xHA * xAlpha * xOA
        )

        f = np.column_stack([f1, f2, f3])

        f_norm = (f - self.ideal) / (self.nadir - self.ideal)
        out["F"] = f_norm

class RE41(Problem):
    def __init__(self):
        self.ideal = np.array([15.576004, 3.58525, 10.61064375, 0.0])
        self.nadir = np.array([39.2905121788, 4.42725, 13.09138125, 9.49401929991])

        self.n_dim = 7
        self.n_obj = 4
        self.lbound = np.array([0.5, 0.45, 0.5, 0.5, 0.875, 0.4, 0.4], dtype=np.float64)
        self.ubound = np.array([1.5, 1.35, 1.5, 1.5, 2.625, 1.2, 1.2], dtype=np.float64)

        xl = np.zeros(self.n_dim)
        xu = np.ones(self.n_dim)

        super().__init__(n_var=7, n_obj=4, xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        X = X * (self.ubound - self.lbound) + self.lbound
        n_sub = len(X)

        f = np.zeros((n_sub, self.n_obj))
        g = np.zeros((n_sub, 10))

        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]
        x4 = X[:, 3]
        x5 = X[:, 4]
        x6 = X[:, 5]
        x7 = X[:, 6]

        # First original objective function
        f[:, 0] = 1.98 + 4.9 * x1 + 6.67 * x2 + 6.98 * x3 + 4.01 * x4 + 1.78 * x5 + 0.00001 * x6 + 2.73 * x7
        # Second original objective function
        f[:, 1] = 4.72 - 0.5 * x4 - 0.19 * x2 * x3
        # Third original objective function
        Vmbp = 10.58 - 0.674 * x1 * x2 - 0.67275 * x2
        Vfd = 16.45 - 0.489 * x3 * x7 - 0.843 * x5 * x6
        f[:, 2] = 0.5 * (Vmbp + Vfd)

        # Constraint functions
        g[:, 0] = 1 - (1.16 - 0.3717 * x2 * x4 - 0.0092928 * x3)
        g[:, 1] = 0.32 - (0.261 - 0.0159 * x1 * x2 - 0.06486 * x1 - 0.019 * x2 * x7 + 0.0144 * x3 * x5 + 0.0154464 * x6)
        g[:, 2] = 0.32 - (
                0.214 + 0.00817 * x5 - 0.045195 * x1 - 0.0135168 * x1 + 0.03099 * x2 * x6 - 0.018 * x2 * x7 + 0.007176 * x3 + 0.023232 * x3 - 0.00364 * x5 * x6 - 0.018 * x2 * x2)
        g[:, 3] = 0.32 - (0.74 - 0.61 * x2 - 0.031296 * x3 - 0.031872 * x7 + 0.227 * x2 * x2)
        g[:, 4] = 32 - (28.98 + 3.818 * x3 - 4.2 * x1 * x2 + 1.27296 * x6 - 2.68065 * x7)
        g[:, 5] = 32 - (33.86 + 2.95 * x3 - 5.057 * x1 * x2 - 3.795 * x2 - 3.4431 * x7 + 1.45728)
        g[:, 6] = 32 - (46.36 - 9.9 * x2 - 4.4505 * x1)
        g[:, 7] = 4 - f[:, 1]
        g[:, 8] = 9.9 - Vmbp
        g[:, 9] = 15.7 - Vfd
        g = np.where(g < 0, -g, 0)
        f[:, 3] = g[:, 0] + g[:, 1] + g[:, 2] + g[:, 3] + g[:, 4] + g[:, 5] + g[:, 6] + g[:, 7] + g[:, 8] + g[:, 9]

        f_norm = (f - self.ideal) / (self.nadir - self.ideal)

        out["F"] = f_norm

class RE42(Problem):
    def __init__(self):
        self.ideal = np.array([-2756.2590400638524, 3962.557843228888, 1947.880856925791, 0.0])
        self.nadir = np.array([-663.57644, 15690.493, 5179.22898, 3207.04559])

        self.n_dim = 6
        self.n_obj = 4
        self.lbound = np.array([150.0, 20.0, 13.0, 10.0, 14.0, 0.63], dtype=np.float64)
        self.ubound = np.array([274.32, 32.31, 25.0, 11.71, 18.0, 0.75], dtype=np.float64)

        xl = np.zeros(self.n_dim)
        xu = np.ones(self.n_dim)

        super().__init__(n_var=6, n_obj=4, xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        X = X * (self.ubound - self.lbound) + self.lbound
        n_sub = len(X)

        f = np.zeros((n_sub, self.n_obj))
        # NOT g
        constraintFuncs = np.zeros((n_sub, 9))

        x_L = X[:, 0]
        x_B = X[:, 1]
        x_D = X[:, 2]
        x_T = X[:, 3]
        x_Vk = X[:, 4]
        x_CB = X[:, 5]

        displacement = 1.025 * x_L * x_B * x_T * x_CB
        V = 0.5144 * x_Vk
        g = 9.8065
        Fn = V / np.power(g * x_L, 0.5)
        a = (4977.06 * x_CB * x_CB) - (8105.61 * x_CB) + 4456.51
        b = (-10847.2 * x_CB * x_CB) + (12817.0 * x_CB) - 6960.32

        power = (np.power(displacement, 2.0 / 3.0) * np.power(x_Vk, 3.0)) / (a + (b * Fn))
        outfit_weight = 1.0 * np.power(x_L, 0.8) * np.power(x_B, 0.6) * np.power(x_D, 0.3) * np.power(x_CB, 0.1)
        steel_weight = 0.034 * np.power(x_L, 1.7) * np.power(x_B, 0.7) * np.power(x_D, 0.4) * np.power(x_CB, 0.5)
        machinery_weight = 0.17 * np.power(power, 0.9)
        light_ship_weight = steel_weight + outfit_weight + machinery_weight

        ship_cost = 1.3 * ((2000.0 * np.power(steel_weight, 0.85)) + (3500.0 * outfit_weight) + (
                2400.0 * np.power(power, 0.8)))
        capital_costs = 0.2 * ship_cost

        DWT = displacement - light_ship_weight

        running_costs = 40000.0 * np.power(DWT, 0.3)

        round_trip_miles = 5000.0
        sea_days = (round_trip_miles / 24.0) * x_Vk
        handling_rate = 8000.0

        daily_consumption = ((0.19 * power * 24.0) / 1000.0) + 0.2
        fuel_price = 100.0
        fuel_cost = 1.05 * daily_consumption * sea_days * fuel_price
        port_cost = 6.3 * np.power(DWT, 0.8)

        fuel_carried = daily_consumption * (sea_days + 5.0)
        miscellaneous_DWT = 2.0 * np.power(DWT, 0.5)

        cargo_DWT = DWT - fuel_carried - miscellaneous_DWT
        port_days = 2.0 * ((cargo_DWT / handling_rate) + 0.5)
        RTPA = 350.0 / (sea_days + port_days)

        voyage_costs = (fuel_cost + port_cost) * RTPA
        annual_costs = capital_costs + running_costs + voyage_costs
        annual_cargo = cargo_DWT * RTPA

        f[:, 0] = annual_costs / annual_cargo
        f[:, 1] = light_ship_weight
        # f_2 is dealt as a minimization problem
        f[:, 2] = -annual_cargo

        # Reformulated objective functions
        constraintFuncs[:, 0] = (x_L / x_B) - 6.0
        constraintFuncs[:, 1] = -(x_L / x_D) + 15.0
        constraintFuncs[:, 2] = -(x_L / x_T) + 19.0
        constraintFuncs[:, 3] = 0.45 * np.power(DWT, 0.31) - x_T
        constraintFuncs[:, 4] = 0.7 * x_D + 0.7 - x_T
        constraintFuncs[:, 5] = 500000.0 - DWT
        constraintFuncs[:, 6] = DWT - 3000.0
        constraintFuncs[:, 7] = 0.32 - Fn

        KB = 0.53 * x_T
        BMT = ((0.085 * x_CB - 0.002) * x_B * x_B) / (x_T * x_CB)
        KG = 1.0 + 0.52 * x_D
        constraintFuncs[:, 8] = (KB + BMT - KG) - (0.07 * x_B)

        constraintFuncs = np.where(constraintFuncs < 0, -constraintFuncs, 0)
        f[:, 3] = constraintFuncs[:, 0] + constraintFuncs[:, 1] + constraintFuncs[:, 2] + constraintFuncs[:,3] + constraintFuncs[:, 4] + constraintFuncs[:, 5] + constraintFuncs[:, 6] + constraintFuncs[:, 7] + constraintFuncs[:, 8]

        f_norm = (f - self.ideal) / (self.nadir - self.ideal)
        out["F"] = f_norm

class RE61(Problem):
    def __init__(self):
        self.ideal = np.array([63840.2774, 30.0, 285346.896494, 183749.967061, 7.22222222222, 0.0])
        self.nadir = np.array([80896.9128355, 1350.0, 2853468.96494, 7076861.67064, 87748.6339553, 2.50994535821])

        self.n_dim = 3
        self.n_obj = 6
        self.lbound = np.array([0.01, 0.01, 0.01], dtype=np.float64)
        self.ubound = np.array([0.45, 0.1, 0.1], dtype=np.float64)

        xl = np.zeros(self.n_dim)
        xu = np.ones(self.n_dim)

        super().__init__(n_var=3, n_obj=6, xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        X = X * (self.ubound - self.lbound) + self.lbound
        n_sub = len(X)
        f = np.zeros((n_sub, self.n_obj))
        g = np.zeros((n_sub, 7))
        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]

        # First original objective function
        f[:, 0] = 106780.37 * (x1 + x3) + 61704.67
        # Second original objective function
        f[:, 1] = 3000 * x1
        # Third original objective function
        f[:, 2] = 305700 * 2289 * x2 / np.power(0.06 * 2289, 0.65)
        # Fourth original objective function
        f[:, 3] = 250 * 2289 * np.exp(-39.75 * x2 + 9.9 * x3 + 2.74)
        # Fifth original objective function
        f[:, 4] = 25 * (1.39 / (x1 * x2) + 4940 * x3 - 80)

        # Constraint functions
        g[:, 0] = 1 - (0.00139 / (x1 * x2) + 4.94 * x3 - 0.08)
        g[:, 1] = 1 - (0.000306 / (x1 * x2) + 1.082 * x3 - 0.0986)
        g[:, 2] = 50000 - (12.307 / (x1 * x2) + 49408.24 * x3 + 4051.02)
        g[:, 3] = 16000 - (2.098 / (x1 * x2) + 8046.33 * x3 - 696.71)
        g[:, 4] = 10000 - (2.138 / (x1 * x2) + 7883.39 * x3 - 705.04)
        g[:, 5] = 2000 - (0.417 * x1 * x2 + 1721.26 * x3 - 136.54)
        g[:, 6] = 550 - (0.164 / (x1 * x2) + 631.13 * x3 - 54.48)

        g = np.where(g < 0, -g, 0)
        f[:, 5] = g[:, 0] + g[:, 1] + g[:, 2] + g[:, 3] + g[:, 4] + g[:, 5] + g[:, 6]

        f_norm = (f - self.ideal) / (self.nadir - self.ideal)
        out["F"] = f_norm

class RE91(Problem):
    def __init__(self):
        self.ideal = np.array([15.549754, 0.0, 0.0, 0.0907242296262, 0.367459287472, 0.527364946723, 0.735415465187, 0.618676033791, 0.660886967497])
        self.nadir = np.array([39.6023250742, 1.00188125422, 112.487885728, 0.79017024474, 1.42304666576, 1.08576907833, 1.1215181762, 0.993535488298, 1.01068662645])

        self.n_dim = 7
        self.n_obj = 9
        self.lbound = np.array([0.5, 0.45, 0.5, 0.5, 0.875, 0.4, 0.4], dtype=np.float64)
        self.ubound = np.array([1.5, 1.35, 1.5, 1.5, 2.625, 1.2, 1.2], dtype=np.float64)
        xl = np.zeros(self.n_dim)
        xu = np.ones(self.n_dim)

        super().__init__(n_var=7, n_obj=9, xl=xl, xu=xu)

    def _evaluate(self, X, out, *args, **kwargs):
        X = X * (self.ubound - self.lbound) + self.lbound
        n_sub = len(X)
        f = np.zeros((n_sub, self.n_obj))
        g = np.zeros((n_sub, 0))
        x1 = X[:, 0]
        x2 = X[:, 1]
        x3 = X[:, 2]
        x4 = X[:, 3]
        x5 = X[:, 4]
        x6 = X[:, 5]
        x7 = X[:, 6]
        # stochastic variables
        x8 = 0.006 * (np.random.normal(0, 1)) + 0.345
        x9 = 0.006 * (np.random.normal(0, 1)) + 0.192
        x10 = 10 * (np.random.normal(0, 1)) + 0.0
        x11 = 10 * (np.random.normal(0, 1)) + 0.0

        # First function
        f[:, 0] = 1.98 + 4.9 * x1 + 6.67 * x2 + 6.98 * x3 + 4.01 * x4 + 1.75 * x5 + 0.00001 * x6 + 2.73 * x7
        # Second function
        f[:, 1] = np.maximum(0.0,
                         (1.16 - 0.3717 * x2 * x4 - 0.00931 * x2 * x10 - 0.484 * x3 * x9 + 0.01343 * x6 * x10) / 1.0)
        # Third function
        f[:, 2] = np.maximum(0.0, (
                0.261 - 0.0159 * x1 * x2 - 0.188 * x1 * x8 - 0.019 * x2 * x7 + 0.0144 * x3 * x5 + 0.87570001 * x5 * x10 + 0.08045 * x6 * x9 + 0.00139 * x8 * x11 + 0.00001575 * x10 * x11) / 0.32)
        # Fourth function
        f[:, 3] = np.maximum(0.0, (
                0.214 + 0.00817 * x5 - 0.131 * x1 * x8 - 0.0704 * x1 * x9 + 0.03099 * x2 * x6 - 0.018 * x2 * x7 + 0.0208 * x3 * x8 + 0.121 * x3 * x9 - 0.00364 * x5 * x6 + 0.0007715 * x5 * x10 - 0.0005354 * x6 * x10 + 0.00121 * x8 * x11 + 0.00184 * x9 * x10 - 0.018 * x2 * x2) / 0.32)
        # Fifth function
        f[:, 4] = np.maximum(0.0, (
                0.74 - 0.61 * x2 - 0.163 * x3 * x8 + 0.001232 * x3 * x10 - 0.166 * x7 * x9 + 0.227 * x2 * x2) / 0.32)
        # Sixth function
        tmp = ((
                       28.98 + 3.818 * x3 - 4.2 * x1 * x2 + 0.0207 * x5 * x10 + 6.63 * x6 * x9 - 7.77 * x7 * x8 + 0.32 * x9 * x10) + (
                       33.86 + 2.95 * x3 + 0.1792 * x10 - 5.057 * x1 * x2 - 11 * x2 * x8 - 0.0215 * x5 * x10 - 9.98 * x7 * x8 + 22 * x8 * x9) + (
                       46.36 - 9.9 * x2 - 12.9 * x1 * x8 + 0.1107 * x3 * x10)) / 3
        f[:, 5] = np.maximum(0.0, tmp / 32)
        # Seventh function
        f[:, 6] = np.maximum(0.0, (
                4.72 - 0.5 * x4 - 0.19 * x2 * x3 - 0.0122 * x4 * x10 + 0.009325 * x6 * x10 + 0.000191 * x11 * x11) / 4.0)
        # EighthEighth function
        f[:, 7] = np.maximum(0.0, (
                10.58 - 0.674 * x1 * x2 - 1.95 * x2 * x8 + 0.02054 * x3 * x10 - 0.0198 * x4 * x10 + 0.028 * x6 * x10) / 9.9)
        # Ninth function
        f[:, 8] = np.maximum(0.0, (
                16.45 - 0.489 * x3 * x7 - 0.843 * x5 * x6 + 0.0432 * x9 * x10 - 0.0556 * x9 * x11 - 0.000786 * x11 * x11) / 15.7)

        f_norm = (f - self.ideal) / (self.nadir - self.ideal)
        out["F"] = f_norm
