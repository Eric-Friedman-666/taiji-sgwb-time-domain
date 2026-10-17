import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import InterpolatedUnivariateSpline, interp1d







PI = 3.141592653589793
TWOPI = 6.283185307179586
C = 299792458.0
G = 6.67e-11
MPC = 3.0857e22
MSUN = 1.989e30
AU = 1.496e11
DAY = 24.0 * 3600.0
YEAR = 31558149.763545603
OmegaM = 0.3111
F_LASER = 2.816e14
L_nominal = 3e9  # [m], nominal arm-length
SACC_nominal = 3e-15  # [m/s2], nominal amplitude of the test-mass acceleration noise
SOMS_nominal = 8e-12  # [m], nominal amplitude of the optical metrology noise
SRO_SCI_C_nominal = SOMS_nominal  # [m], sci_c readout noise
SRO_SCI_SB_nominal = 8.00e-12  # [m], sci_sb readout noise
SRO_TM_C_nominal = 0  # [m], tm_c readout noise
SRO_REF_C_nominal = 0  # [m], ref_c readout noise
SRO_REF_SB_nominal = 0  # [m], ref_sb readout noise
SOP_TM_LOCAL_nominal = 0  # [m]
SOP_REF_LOCAL_nominal = 0  # [m]
SOP_OTHER_nominal = 0  # [m]
SLASER_nominal = 30  # [Hz], laser frequency noise
SOB_nominal = 1e-2 
SCLOCK_nominal = 6.32e-14  # [s/s], clock noise
SBL_nominal = 3e-12  # [m], fibre back link noise
SR_nominal = 0.9  # [m], pseudo raninging noise
SM_LEFT_nominal = 5.2e-14  # [s/s]
SM_RIGHT_nominal = 5.2e-13  # [s/s]
MOSA_labels = ["12", "13", "23", "21", "31", "32"]
left_MOSA_labels = {"12", "23", "31"}
right_MOSA_labels = {"13", "21", "32"}
adjacent_MOSA_labels = {
    "12": "13",
    "13": "12",
    "23": "21",
    "21": "23",
    "31": "32",
    "32": "31",
}
SC_labels = ["1", "2", "3"]
modulation_freqs = {
    "12": 2.4e9,
    "13": 2.401e9,
    "23": 2.4e9,
    "21": 2.401e9,
    "31": 2.4e9,
    "32": 2.401e9,
}
lock_topology = {
    "12": "primary",
    "13": "adjacent",
    "23": "adjacent",
    "21": "distant",
    "31": "distant",
    "32": "adjacent",
}
lock_order = ["12", "13", "31", "32", "21", "23"]
fplan = {
    "12": -5510000.0,
    "13": -17130000.0,
    "23": 16440000.0,
    "21": -11080000.0,
    "31": 14740000.0,
    "32": -9020000.0,
}
default_clock_t0 = {"1": 0, "2": 0, "3": 0}
default_clock_y0 = {"1": 5e-8, "2": 6.25e-7, "3": -3.75e-7}
default_clock_y1 = {"1": 1.6e-15, "2": 2e-14, "3": -1.2e-14}
default_clock_y2 = {"1": 9e-24, "2": 6.75e-23, "3": -1.125e-22}
default_laser_O0 = {"12": 0.0, "13": 0.0, "23": 0.0, "21": 0.0, "31": 0.0, "32": 0.0}
MOSA_colors = {
    "12": "#E04F38",
    "13": "#7D8F4D",
    "23": "#3877E0",
    "21": "#662D24",
    "31": "#B3E038",
    "32": "#293E61",
}
SC_colors = {"1": "#E04F38", "2": "#3877E0", "3": "#B3E038"}
ORANGE = "#ED8D5A"
BLUE = "#68BED9"
GREEN1 = "#257D8B"
GREEN2 = "#BFDFB2"
YELLOW = "#EFCE87"
RED = "#B54764"

class Orbit:
    """
    return dictionaries of functions.
    all the functions take time array (N) as input and return (N, dim) array
    the start time of TCB is set to t = 0, and TCB/TPS coincide at t = 0
    """

    def __init__(self, OrbitDir, max_rows=None, tstart=0, dt=DAY, pn_order=2):
        # read in orbit data
        # each item is a N * 3 array
        if max_rows is None:
            self.rdata = {key: np.loadtxt(OrbitDir + "/SCP" + key + ".dat") * AU for key in SC_labels}
            self.vdata = {key: np.loadtxt(OrbitDir + "/SCV" + key + ".dat") * AU / DAY for key in SC_labels}
        else:
            self.rdata = {key: np.loadtxt(OrbitDir + "/SCP" + key + ".dat", max_rows=max_rows) * AU for key in SC_labels}
            self.vdata = {key: np.loadtxt(OrbitDir + "/SCV" + key + ".dat", max_rows=max_rows) * AU / DAY for key in SC_labels}

        self.tstart = tstart
        self.N = len(self.rdata[SC_labels[0]])
        self.tdata = np.arange(self.N) * dt - self.tstart
        self.dt = dt

        LTTdata = {}
        self._LTTfunctions = {}
        self._Dopplerfunctions = {}
        self.LTT_time_int = dict()  # used in MBHB response calculation
        self.LTT_data_int = dict()  # used in MBHB response calculation

        ARMdata = {}
        self._ArmVectorfunctions = {}
        self.ARM_time_int = dict()  # used in MBHB response calculation
        self.ARM_data_int = dict()  # used in MBHB response calculation

        self._Positionfunctions = {}
        self._Velocityfunctions = {}
        self.POS_time_int = dict()  # used in MBHB response calculation
        self.POS_data_int = dict()  # used in MBHB response calculation

        for label in MOSA_labels:
            send_SC = label[1]
            receive_SC = label[0]

            L = self.rdata[receive_SC] - self.rdata[send_SC]
            # 0-th order
            LTT0 = np.sqrt(np.sum(L * L, axis=1)) / C
            ARM0 = L / LTT0[:, np.newaxis] / C
            # 1/2-th order
            nv_recv = np.sum(ARM0 * self.vdata[receive_SC], axis=1)
            LTT1 = np.sum(L * self.vdata[receive_SC], axis=1) / C**2
            ARM1 = self.vdata[receive_SC] / C - ARM0 * nv_recv[:, np.newaxis] / C
            # 1-st order
            LTT2 = 0.5 * (np.sum((self.vdata[receive_SC]) ** 2, axis=1) / C**2 + (nv_recv / C) ** 2) * LTT0
            nx_send = np.sum(ARM0 * self.rdata[send_SC], axis=1)
            nx_recv = np.sum(ARM0 * self.rdata[receive_SC], axis=1)
            K2 = np.sum((self.rdata[send_SC]) ** 2, axis=1) - nx_send**2
            P = (self.rdata[send_SC] - ARM0 * nx_send[:, np.newaxis]) / K2[:, np.newaxis]
            r_recv = np.sqrt(np.sum((self.rdata[receive_SC]) ** 2, axis=1))
            r_send = np.sqrt(np.sum((self.rdata[send_SC]) ** 2, axis=1))
            chi = P * (r_recv - r_send)[:, np.newaxis] + ARM0 * (np.log((nx_recv + r_recv) / (nx_send + r_send)))[:, np.newaxis]
            LTT2 += np.sum(
                (2.0 * G * MSUN / C**3 * chi - (G * MSUN / 2.0 / C / r_recv**3 * LTT0**2)[:, np.newaxis] * self.rdata[receive_SC]) * ARM0,
                axis=1,
            )

            if pn_order == 2:
                LTTdata[label] = LTT0 + LTT1 + LTT2
            elif pn_order == 1:
                LTTdata[label] = LTT0 + LTT1
            elif pn_order == 0:
                LTTdata[label] = LTT0
            else:
                raise NotImplementedError("PN order not implemented.")

            # LTTs are calculated at the emission times, while in the simulation we use LTTs at the reception times
            self._LTTfunctions[label] = InterpolatedUnivariateSpline(self.tdata + LTTdata[label], LTTdata[label], k=5, ext="extrapolate")
            self._Dopplerfunctions[label] = self._LTTfunctions[label].derivative()
            self.LTT_time_int[label] = self.tdata + LTTdata[label]  # (Ntime) used in MBHB response calculation
            self.LTT_data_int[label] = LTTdata[label].copy()  # (Ntime) in [s] used in MBHB response calculation

            ARMdata[label] = ARM0 + ARM1
            # arm vectors are calculated at the emission times, while in the simulation we use arm vectors at the reception times
            self._ArmVectorfunctions[label] = interp1d(
                self.tdata + LTTdata[label],
                ARMdata[label],
                axis=0,
                kind="cubic",
                bounds_error=False,
                fill_value="extrapolate",
            )
            self.ARM_time_int[label] = self.tdata + LTTdata[label]  # (Ntime) used in MBHB response calculation
            self.ARM_data_int[label] = ARMdata[label].copy()  # (Ntime, 3) used in MBHB response calculation

        self.TCBinTPSfunctions = {}
        self.TPSinTCBfunctions = {}
        self.TPSwrtTCBfunctions = {}  # TPS - TCB in TCB
        TPSdata = {}
        for label in SC_labels:
            vSC = np.sqrt(np.sum((self.vdata[label]) ** 2, axis=1))
            rSC = np.sqrt(np.sum((self.rdata[label]) ** 2, axis=1))
            rel_diff = -G * MSUN / rSC / C**2 - vSC**2 / 2.0 / C**2
            rel_diff = cumulative_trapezoid(np.insert(rel_diff, 0, 0), dx=self.dt)  # proper time = tcb at the start time
            TPSdata[label] = self.tdata + rel_diff
            self.TCBinTPSfunctions[label] = InterpolatedUnivariateSpline(TPSdata[label], self.tdata, k=5, ext="extrapolate")
            self.TPSinTCBfunctions[label] = InterpolatedUnivariateSpline(self.tdata, TPSdata[label], k=5, ext="extrapolate")
            self.TPSwrtTCBfunctions[label] = InterpolatedUnivariateSpline(self.tdata, TPSdata[label] - self.tdata, k=5, ext="extrapolate")

        # calculate ppr data, recv proper time: TPSdata, recv tcb: tdata, send tcb: tdata - LTT(tdata), send proper time: TPSinTCB(send tcb)
        self._PPRfunctions = {}
        self._DPPRfunctions = {}
        for label in MOSA_labels:
            send_SC = label[1]
            receive_SC = label[0]
            recv_tps = TPSdata[receive_SC]
            recv_tcb = self.tdata
            send_tcb = recv_tcb - self._LTTfunctions[label](recv_tcb)
            send_tps = self.TPSinTCBfunctions[send_SC](send_tcb)
            ppr = recv_tps - send_tps
            self._PPRfunctions[label] = InterpolatedUnivariateSpline(recv_tps, ppr, k=5, ext="extrapolate")
            self._DPPRfunctions[label] = self._PPRfunctions[label].derivative()

        for label in SC_labels:
            self._Positionfunctions[label] = interp1d(
                self.tdata,
                self.rdata[label],
                axis=0,
                kind="cubic",
                bounds_error=False,
                fill_value="extrapolate",
            )
            self._Velocityfunctions[label] = interp1d(
                self.tdata,
                self.vdata[label],
                axis=0,
                kind="cubic",
                bounds_error=False,
                fill_value="extrapolate",
            )
            self.POS_time_int[label] = self.tdata.copy()  # (Ntime) used in MBHB response calculation
            self.POS_data_int[label] = self.rdata[label] / C 

    def ListMembers(self):
        for name, value in vars(self).items():
            print("%s=%s" % (name, value))

    def LTTfunctions(self):
        return self._LTTfunctions

    def Dopplerfunctions(self):
        return self._Dopplerfunctions

    def PPRfunctions(self):
        return self._PPRfunctions

    def DPPRfunctions(self):
        return self._DPPRfunctions

    def ArmVectorfunctions(self):
        return self._ArmVectorfunctions

    def Positionfunctions(self):
        return self._Positionfunctions

    def Velocityfunctions(self):
        return self._Velocityfunctions

class MOSADict:
    """
    NOTE: be careful when manipulating array or dict elements
    NOTE: use B = A.copy() instead of B = A
    the keys of MOSADict must be MOSA_labels
    the values of MOSADict are numpy arrays of the same dim or scalars
    """

    dict_keys = MOSA_labels

    def __init__(self, data_dict):
        if isinstance(data_dict, dict):
            if set(data_dict.keys()) == set(self.dict_keys):
                self.data = data_dict
            else:
                raise ValueError("keys of dictionaries mismatch.")
        else:
            raise TypeError("data_dict must be a dict.")

    def __repr__(self):
        return str(self.data)

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        if key in self.dict_keys:
            self.data[key] = value
        else:
            raise ValueError(str(key) + " is not a surpported key.")

    def __add__(self, other):
        if isinstance(other, MOSADict):
            new_data = {}
            for key in self.dict_keys:
                new_data[key] = self.data[key] + other.data[key]
            return MOSADict(new_data)
        elif np.isscalar(other):
            new_data = {}
            for key in self.dict_keys:
                new_data[key] = self.data[key] + other
            return MOSADict(new_data)
        else:
            raise TypeError("Unsupported operation between instances of 'MOSADict' and '{}'".format(type(other)))

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, MOSADict):
            new_data = {}
            for key in self.dict_keys:
                new_data[key] = self.data[key] * other.data[key]
            return MOSADict(new_data)
        elif np.isscalar(other):
            new_data = {}
            for key in self.dict_keys:
                new_data[key] = self.data[key] * other
            return MOSADict(new_data)
        else:
            raise TypeError("Unsupported operation between instances of 'MOSADict' and '{}'".format(type(other)))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        if isinstance(other, MOSADict):
            new_data = {}
            for key in self.dict_keys:
                new_data[key] = self.data[key] - other.data[key]
            return MOSADict(new_data)
        elif np.isscalar(other):
            new_data = {}
            for key in self.dict_keys:
                new_data[key] = self.data[key] - other
            return MOSADict(new_data)
        else:
            raise TypeError("Unsupported operation between instances of 'MOSADict' and '{}'".format(type(other)))

    def __rsub__(self, other):
        if isinstance(other, MOSADict):
            new_data = {}
            for key in self.dict_keys:
                new_data[key] = other.data[key] - self.data[key]
            return MOSADict(new_data)
        elif np.isscalar(other):
            new_data = {}
            for key in self.dict_keys:
                new_data[key] = other - self.data[key]
            return MOSADict(new_data)
        else:
            raise TypeError("Unsupported operation between instances of 'MOSADict' and '{}'".format(type(other)))

    def __neg__(self):
        return MOSADict({key: -self.data[key] for key in self.dict_keys})

    def __truediv__(self, other):
        if isinstance(other, MOSADict):
            new_data = {}
            for key in self.dict_keys:
                new_data[key] = self.data[key] / other.data[key]
            return MOSADict(new_data)
        elif np.isscalar(other):
            new_data = {}
            for key in self.dict_keys:
                new_data[key] = self.data[key] / other
            return MOSADict(new_data)
        else:
            raise TypeError("Unsupported operation between instances of 'MOSADict' and '{}'".format(type(other)))

    def __rtruediv__(self, other):
        if isinstance(other, MOSADict):
            new_data = {}
            for key in self.dict_keys:
                new_data[key] = other.data[key] / self.data[key]
            return MOSADict(new_data)
        elif np.isscalar(other):
            new_data = {}
            for key in self.dict_keys:
                new_data[key] = other / self.data[key]
            return MOSADict(new_data)
        else:
            raise TypeError("Unsupported operation between instances of 'MOSADict' and '{}'".format(type(other)))

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def copy(self):
        if type(self.data[MOSA_labels[0]]) is np.ndarray:
            d = {}
            for key in MOSA_labels:
                d[key] = np.copy(self.data[key])
            return MOSADict(d)
        else:
            return MOSADict(self.data.copy())

    def reverse(self):
        d = {}
        for k in MOSA_labels:
            k_re = k[1] + k[0]
            d[k] = self.data[k_re]
        return MOSADict(d)

    def adjacent(self):
        d = {}
        for k in MOSA_labels:
            k_ad = adjacent_MOSA_labels[k]
            d[k] = self.data[k_ad]
        return MOSADict(d)

    def timedelay(self, fsample, delay, doppler=None, order=31, pool=None):
        """
        delay and doppler (if not None) must be MOSADicts
        """
        if isinstance(delay, MOSADict):
            d = {}
            if isinstance(doppler, MOSADict):
                if pool is None:
                    for k in MOSA_labels:
                        d[k] = timeshift(self.data[k], -delay[k] * fsample, order=order)
                        # d[k] = timeshift(self.data[k], -delay[k] * fsample, order=order) * (1. - doppler[k])
                else:
                    param_arr = [(self.data[k], -delay[k] * fsample, order) for k in MOSA_labels]
                    d_arr = pool.starmap(timeshift, param_arr)
                    for j, k in enumerate(MOSA_labels):
                        d[k] = d_arr[j]
                return MOSADict(d) * (1.0 - doppler)
            elif doppler is None:
                if pool is None:
                    for k in MOSA_labels:
                        d[k] = timeshift(self.data[k], -delay[k] * fsample, order=order)
                else:
                    param_arr = [(self.data[k], -delay[k] * fsample, order) for k in MOSA_labels]
                    d_arr = pool.starmap(timeshift, param_arr)
                    for j, k in enumerate(MOSA_labels):
                        d[k] = d_arr[j]
                return MOSADict(d)
            else:
                raise TypeError("unsurpported doppler type.")
        else:
            raise TypeError("unsurpported delay type.")

    def integrate(self, fsample):
        d = {}
        for k in MOSA_labels:
            d[k] = cumulative_trapezoid(np.insert(self.data[k], 0, 0), dx=1 / fsample)
        return MOSADict(d)

    def derivative(self, fsample):
        d = {}
        for k in MOSA_labels:
            d[k] = np.gradient(self.data[k], 1 / fsample)
        return MOSADict(d)

    def downsampled(self, fsample, downsample, kaiser_filter_coef):
        d = {}
        for k in MOSA_labels:
            d[k] = downsampling(self.data[k], fsample, downsample, kaiser_filter_coef)
        return MOSADict(d)

    def detrended(self, order=2):
        d = {}
        for k in MOSA_labels:
            d[k] = detrending(self.data[k], order=order)
        return MOSADict(d)

    def drop_edge_points(self, points1=0, points2=0):
        d = {}
        if points2 == 0:
            for k in MOSA_labels:
                d[k] = self.data[k][points1:]
        else:
            for k in MOSA_labels:
                d[k] = self.data[k][points1:-points2]
        return MOSADict(d)

    def get_data_by_idx(self, idx):
        d = {}
        for key in MOSA_labels:
            d[key] = self.data[key][idx]
        return MOSADict(d)


class SCDict:
    """
    NOTE: be careful when manipulating array or dict elements
    NOTE: use B = A.copy() instead of B = A
    the keys of SCDict must be SC_labels
    the values of SCDict are numpy arrays of the same dim or scalars
    """

    dict_keys = SC_labels

    def __init__(self, data_dict):
        if isinstance(data_dict, dict):
            if set(data_dict.keys()) == set(self.dict_keys):
                self.data = data_dict
            else:
                raise ValueError("keys of dictionaries mismatch.")
        else:
            raise TypeError("data_dict must be a dict.")

    def __repr__(self):
        return str(self.data)

    def __getitem__(self, key):
        return self.data[key]

    def __setitem__(self, key, value):
        if key in self.dict_keys:
            self.data[key] = value
        else:
            raise ValueError(str(key) + " is not a surpported key.")

    def __add__(self, other):
        if isinstance(other, SCDict):
            new_data = {}
            for key in self.dict_keys:
                new_data[key] = self.data[key] + other.data[key]
            return SCDict(new_data)
        elif np.isscalar(other):
            new_data = {}
            for key in self.dict_keys:
                new_data[key] = self.data[key] + other
            return SCDict(new_data)
        else:
            raise TypeError("Unsupported operation between instances of 'SCDict' and '{}'".format(type(other)))

    def __radd__(self, other):
        return self.__add__(other)

    def __mul__(self, other):
        if isinstance(other, SCDict):
            new_data = {}
            for key in self.dict_keys:
                new_data[key] = self.data[key] * other.data[key]
            return SCDict(new_data)
        elif np.isscalar(other):
            new_data = {}
            for key in self.dict_keys:
                new_data[key] = self.data[key] * other
            return SCDict(new_data)
        else:
            raise TypeError("Unsupported operation between instances of 'SCDict' and '{}'".format(type(other)))

    def __rmul__(self, other):
        return self.__mul__(other)

    def __sub__(self, other):
        if isinstance(other, SCDict):
            new_data = {}
            for key in self.dict_keys:
                new_data[key] = self.data[key] - other.data[key]
            return SCDict(new_data)
        elif np.isscalar(other):
            new_data = {}
            for key in self.dict_keys:
                new_data[key] = self.data[key] - other
            return SCDict(new_data)
        else:
            raise TypeError("Unsupported operation between instances of 'SCDict' and '{}'".format(type(other)))

    def __rsub__(self, other):
        if isinstance(other, SCDict):
            new_data = {}
            for key in self.dict_keys:
                new_data[key] = other.data[key] - self.data[key]
            return SCDict(new_data)
        elif np.isscalar(other):
            new_data = {}
            for key in self.dict_keys:
                new_data[key] = other - self.data[key]
            return SCDict(new_data)
        else:
            raise TypeError("Unsupported operation between instances of 'SCDict' and '{}'".format(type(other)))

    def __neg__(self):
        return SCDict({key: -self.data[key] for key in self.dict_keys})

    def __truediv__(self, other):
        if isinstance(other, SCDict):
            new_data = {}
            for key in self.dict_keys:
                new_data[key] = self.data[key] / other.data[key]
            return SCDict(new_data)
        elif np.isscalar(other):
            new_data = {}
            for key in self.dict_keys:
                new_data[key] = self.data[key] / other
            return SCDict(new_data)
        else:
            raise TypeError("Unsupported operation between instances of 'SCDict' and '{}'".format(type(other)))

    def __rtruediv__(self, other):
        if isinstance(other, SCDict):
            new_data = {}
            for key in self.dict_keys:
                new_data[key] = other.data[key] / self.data[key]
            return SCDict(new_data)
        elif np.isscalar(other):
            new_data = {}
            for key in self.dict_keys:
                new_data[key] = other / self.data[key]
            return SCDict(new_data)
        else:
            raise TypeError("Unsupported operation between instances of 'SCDict' and '{}'".format(type(other)))

    def keys(self):
        return self.data.keys()

    def values(self):
        return self.data.values()

    def items(self):
        return self.data.items()

    def copy(self):
        if type(self.data[SC_labels[0]]) is np.ndarray:
            d = {}
            for key in SC_labels:
                d[key] = np.copy(self.data[key])
            return SCDict(d)
        else:
            return SCDict(self.data.copy())

    def toMOSA(self):
        d = {}
        for key in MOSA_labels:
            d[key] = self.data[key[0]]
        return MOSADict(d)

    def integrate(self, fsample):
        d = {}
        for k in SC_labels:
            d[k] = cumulative_trapezoid(np.insert(self.data[k], 0, 0), dx=1 / fsample)
        return SCDict(d)

    def derivative(self, fsample):
        d = {}
        for k in SC_labels:
            d[k] = np.gradient(self.data[k], 1 / fsample)
        return SCDict(d)

    def timedelay(self, fsample, delay, order=31, pool=None):
        """
        delay must be SCDicts
        """
        if isinstance(delay, SCDict):
            d = {}
            if pool is None:
                for k in SC_labels:
                    d[k] = timeshift(self.data[k], -delay[k] * fsample, order=order)
            else:
                param_arr = [(self.data[k], -delay[k] * fsample, order) for k in SC_labels]
                d_arr = pool.starmap(timeshift, param_arr)
                for j, k in enumerate(SC_labels):
                    d[k] = d_arr[j]
            return SCDict(d)
        else:
            raise TypeError("unsurpported delay type.")

    def downsampled(self, fsample, downsample, kaiser_filter_coef):
        d = {}
        for k in SC_labels:
            d[k] = downsampling(self.data[k], fsample, downsample, kaiser_filter_coef)
        return SCDict(d)

    def detrended(self, order=2):
        d = {}
        for k in SC_labels:
            d[k] = detrending(self.data[k], order=order)
        return SCDict(d)

    def drop_edge_points(self, points1=0, points2=0):
        d = {}
        if points2 == 0:
            for k in SC_labels:
                d[k] = self.data[k][points1:]
        else:
            for k in SC_labels:
                d[k] = self.data[k][points1:-points2]
        return SCDict(d)


class TDI:
    """
    NOTE: garbage points should be dropped after the calculation of TDI.
    """

    # examples for the path strings
    X1_plus = "12131"
    X1_minus = "13121"
    X2_plus = "121313121"
    X2_minus = "131212131"
    alpha1_plus = "1231"
    alpha1_minus = "1321"
    alpha2_plus = "1321231"
    alpha2_minus = "1231321"

    # examples for the P strings (NOTE that the sign of XYZ should be inverted to be consistent with the path strings and fast michelson calculations)
    X2_string = {
        "12": [(1.0, []), (-1.0, ["13", "31"]), (-1.0, ["13", "31", "12", "21"]), (1.0, ["12", "21", "13", "31", "13", "31"])],
        "23": [],
        "31": [(-1.0, ["13"]), (1.0, ["12", "21", "13"]), (1.0, ["12", "21", "13", "31", "13"]), (-1.0, ["13", "31", "12", "21", "12", "21", "13"])],
        "21": [(1.0, ["12"]), (-1.0, ["13", "31", "12"]), (-1.0, ["13", "31", "12", "21", "12"]), (1.0, ["12", "21", "13", "31", "13", "31", "12"])],
        "32": [],
        "13": [(-1.0, []), (1.0, ["12", "21"]), (1.0, ["12", "21", "13", "31"]), (-1.0, ["13", "31", "12", "21", "12", "21"])],
    }
    X1_string = {
        "12": [(1.0, []), (-1.0, ["13", "31"])],
        "23": [],
        "31": [(-1.0, ["13"]), (1.0, ["12", "21", "13"])],
        "21": [(1.0, ["12"]), (-1.0, ["13", "31", "12"])],
        "32": [],
        "13": [(-1.0, []), (1.0, ["12", "21"])],
    }
    
    C_12_3_strings = {
        "13": [(1.0, ["32", "23", "-31"]), (-1.0, ["31", "-12", "23", "-31"])],
        "21": [(-1.0, ["31", "-12"]), (1.0, ["31", "-12", "23", "-31", "12"])],
        "12": [(1.0, ["31", "-12", "23", "-31"]), (-1.0, ["32", "23", "-31"])],
        "23": [(-1.0, ["32"]), (1.0, ["31", "-12"])],
        "31": [(1.0, []), (-1.0, ["32", "23", "-31", "12", "-23"])],
        "32": [(-1.0, []), (1.0, ["32", "23", "-31", "12", "-23"])],
    }
    C_14_3_strings = {
        "12": [
            (1.0, ["23", "-31"]),
            (-1.0, ["21", "12", "23", "-31"]),
            (1.0, ["23", "-31", "12", "-23", "31"]),
            (-1.0, ["21"]),
        ],
        "13": [(-1.0, ["23", "-31"]), (1.0, ["21", "12", "23", "-31"])],
        "32": [
            (-1.0, ["23", "-31", "12", "-23"]),
            (1.0, ["21", "12", "23", "-31", "12", "-23"]),
        ],
        "21": [(-1.0, []), (1.0, ["23", "-31", "12", "-23", "31", "12"])],
        "31": [
            (1.0, ["23", "-31", "12", "-23"]),
            (-1.0, ["21", "12", "23", "-31", "12", "-23"]),
        ],
        "23": [(1.0, []), (-1.0, ["21", "12"])],
    }
    C_16_28_strings = {
        "12": [
            (1.0, ["-12", "23", "-31"]),
            (-1.0, ["-13", "-31", "12", "23", "-31"]),
            (1.0, ["-12", "23", "-31", "12", "-23", "-31"]),
            (-1.0, ["-13", "-31"]),
        ],
        "32": [
            (-1.0, ["-12", "23", "-31", "12", "-23"]),
            (1.0, ["-13", "-31", "12", "23", "-31", "12", "-23"]),
        ],
        "23": [(1.0, ["-12"]), (-1.0, ["-13", "-31", "12"])],
        "13": [
            (1.0, ["-13", "-31", "12", "23", "-31"]),
            (-1.0, ["-12", "23", "-31"]),
            (-1.0, ["-12", "23", "-31", "12", "-23", "-31"]),
            (1.0, ["-13", "-31"]),
        ],
        "31": [(1.0, ["-13"]), (-1.0, ["-13", "-31", "12", "23", "-31", "12", "-23"])],
        "21": [(-1.0, ["-12"]), (1.0, ["-12", "23", "-31", "12", "-23", "-31", "12"])],
    }

    def __init__(self, measurements, delays, fsample, order=31, delay_order=5):
        """
        the measurements dict contains following keys:
        "sci_c",
        "sci_sb",
        "ref_c",
        "ref_sb",
        "tm_c",
        "mpr",
        "a",
        "b",
        each is a MOSADict object
        delays is also a MOSADict object
        """
        self.measurements = measurements.copy()
        self.delays = delays.copy()
        self.fsample = fsample
        if "sci_c" in self.measurements.keys():
            self.size = len(measurements["sci_c"]["12"])
        elif "eta" in self.measurements.keys():
            self.size = len(measurements["eta"]["12"])
        else:
            raise ValueError("can not decide data size.")
        self.order = order
        self.delay_order = delay_order
        if "dpl" not in self.measurements.keys():
            self.measurements["dpl"] = self.delays.derivative(fsample)
        self.dpl = self.measurements["dpl"].copy()

        # construct other channels
        self.Y1_plus = self.index_cycle(self.X1_plus)
        self.Y1_minus = self.index_cycle(self.X1_minus)
        self.Z1_plus = self.index_cycle(self.Y1_plus)
        self.Z1_minus = self.index_cycle(self.Y1_minus)
        self.Y2_plus = self.index_cycle(self.X2_plus)
        self.Y2_minus = self.index_cycle(self.X2_minus)
        self.Z2_plus = self.index_cycle(self.Y2_plus)
        self.Z2_minus = self.index_cycle(self.Y2_minus)

        self.beta1_plus = self.index_cycle(self.alpha1_plus)
        self.beta1_minus = self.index_cycle(self.alpha1_minus)
        self.gamma1_plus = self.index_cycle(self.beta1_plus)
        self.gamma1_minus = self.index_cycle(self.beta1_minus)
        self.beta2_plus = self.index_cycle(self.alpha2_plus)
        self.beta2_minus = self.index_cycle(self.alpha2_minus)
        self.gamma2_plus = self.index_cycle(self.beta2_plus)
        self.gamma2_minus = self.index_cycle(self.beta2_minus)

    def CalculateXi(self, doppler=True, pool=None):
        """
        calculate intermediate variable xi
        """
        if doppler:
            d_args = dict(
                fsample=self.fsample,
                delay=self.delays,
                doppler=self.measurements["dpl"],
                order=self.order,
            )
        else:
            d_args = dict(fsample=self.fsample, delay=self.delays, order=self.order)
        ob_term = self.measurements["ref_c"] - self.measurements["tm_c"]
        delayed_reverse_ob_term = (ob_term.reverse()).timedelay(**d_args, pool=pool)
        xi = self.measurements["sci_c"]
        xi += 0.5 * ob_term
        xi += 0.5 * delayed_reverse_ob_term
        self.measurements["xi"] = xi
        # logger.info('Intervar Xi calculated.')

    def CalculateEta(self, doppler=True):
        """
        calculate intermediate variable eta
        """
        d = {}
        for k in left_MOSA_labels:
            k_re = k[1] + k[0]
            k_ad = adjacent_MOSA_labels[k_re]  # adjacent of reverse
            delayed_ref = timeshift(
                self.measurements["ref_c"][k_re] - self.measurements["ref_c"][k_ad],
                -self.fsample * self.delays[k],
                order=self.order,
            )
            if doppler:
                delayed_ref *= 1.0 - self.measurements["dpl"][k]
            d[k] = self.measurements["xi"][k] + 0.5 * delayed_ref
        for k in right_MOSA_labels:
            k_ad = adjacent_MOSA_labels[k]
            ref = self.measurements["ref_c"][k_ad] - self.measurements["ref_c"][k]
            d[k] = self.measurements["xi"][k] + 0.5 * ref
        self.measurements["eta"] = MOSADict(d)
        # logger.info('intervar Eta calculated.')

    def Path_string_interpret(self, Path_string):
        """
        interpret a virtual path as the combination of data
        """
        N_plus = len(Path_string)
        Path_string_inv = Path_string[::-1]
        delay_array = []
        for i in range(N_plus - 1):
            delay_array.append({})
            delay_array[i]["measurement"] = Path_string_inv[i] + Path_string_inv[i + 1]
            delay_array[i]["delay"] = []
            for j in range(i):
                delay_array[i]["delay"].append(Path_string_inv[j] + Path_string_inv[j + 1])
        return delay_array

    def index_cycle(self, Path_string):
        """
        get the path string of Y from X, Z from Y
        """
        cycle_pair = {1: 2, 2: 3, 3: 1}
        Path_string_cycled = ""
        for char in Path_string:
            idx = int(char)
            Path_string_cycled = Path_string_cycled + str(cycle_pair[idx])
        return Path_string_cycled

    def CalculatePath(self, Path_string, eta_dict, doppler=True):
        """
        calculate the result of a virtual path using the (eta) data and delays
        the path ends at current time
        """
        delay_array = self.Path_string_interpret(Path_string)
        N_measurement = len(delay_array)
        tdi = np.zeros(self.size)
        for i in range(N_measurement):
            N_delay = len(delay_array[i]["delay"])
            total_delay_time = np.zeros(self.size)
            for j in range(N_delay):
                delay_idx = delay_array[i]["delay"][j]
                total_delay_time = total_delay_time - timeshift(
                    self.delays[delay_idx],
                    total_delay_time * self.fsample,
                    order=self.delay_order,
                )  # delays can be calculated with low-order interpolation
            if doppler is True:
                total_delay_doppler = np.gradient(total_delay_time, 1.0 / self.fsample)
                tdi += (1.0 + total_delay_doppler) * timeshift(
                    eta_dict[delay_array[i]["measurement"]],
                    total_delay_time * self.fsample,
                    order=self.order,
                )
            else:
                tdi += timeshift(
                    eta_dict[delay_array[i]["measurement"]],
                    total_delay_time * self.fsample,
                    order=self.order,
                )
        return tdi

    def CalculatePathCombination(self, plus_paths, minus_paths, eta_dict, doppler=True):
        """
        plus path and minus path must be lists: e.g. plus path = [string1, string2, ...]
        """
        N_plus = len(plus_paths)
        N_minus = len(minus_paths)
        tdi_plus = np.zeros(self.size)
        tdi_minus = np.zeros(self.size)

        for i in range(N_plus):
            logger.debug("Calculating plus path " + plus_paths[i])
            tdi_plus += self.CalculatePath(Path_string=plus_paths[i], eta_dict=eta_dict, doppler=doppler)
        for i in range(N_minus):
            logger.debug("Calculating minus path " + minus_paths[i])
            tdi_minus += self.CalculatePath(Path_string=minus_paths[i], eta_dict=eta_dict, doppler=doppler)
        return tdi_plus - tdi_minus

    def CalculateNestedDelay(self, Delay_string, measurement, doppler=True):
        """
        calculate nested delay of any measurement
        """
        if Delay_string == "":
            return measurement
        N_delay = len(Delay_string) - 1
        delays = []
        for i in range(N_delay):
            delays.append(Delay_string[i] + Delay_string[i + 1])
        total_delay_time = np.zeros(self.size)
        for i in range(N_delay):
            delay_idx = delays[i]
            total_delay_time = total_delay_time - timeshift(
                self.delays[delay_idx],
                total_delay_time * self.fsample,
                order=self.delay_order,
            )
        if doppler:
            total_delay_doppler = np.gradient(total_delay_time, 1.0 / self.fsample)
            return (1.0 + total_delay_doppler) * timeshift(measurement, total_delay_time * self.fsample, order=self.order)
        else:
            return timeshift(measurement, total_delay_time * self.fsample, order=self.order)

    def CalculateNestedDelayTime(self, Delay_string):
        """
        calculate the total delay time of a multiple delay operator
        """
        N_delay = len(Delay_string) - 1
        delays = []
        for i in range(N_delay):
            delays.append(Delay_string[i] + Delay_string[i + 1])
        total_delay_time = np.zeros(self.size)
        for i in range(N_delay):
            delay_idx = delays[i]
            total_delay_time = total_delay_time - timeshift(
                self.delays[delay_idx],
                total_delay_time * self.fsample,
                order=self.delay_order,
            )
        return total_delay_time

    def CalculateNestedDelayCombination(self, Delay_string_plus, Delay_string_minus, measurement, doppler=True):
        N_plus = len(Delay_string_plus)
        N_minus = len(Delay_string_minus)
        plus_delay = np.zeros(self.size)
        minus_delay = np.zeros(self.size)
        for i in range(N_plus):
            plus_delay += self.CalculateNestedDelay(
                Delay_string=Delay_string_plus[i],
                measurement=measurement,
                doppler=doppler,
            )
        for i in range(N_minus):
            minus_delay += self.CalculateNestedDelay(
                Delay_string=Delay_string_minus[i],
                measurement=measurement,
                doppler=doppler,
            )
        return plus_delay - minus_delay

    def CalculateBasicTDI(self, channel="X2", doppler=True, channel_name="tdi_channel"):
        """
        channel can be either a string or a list
        1. string: 'X1', 'X2',...
        2. list: [plus_path, minus_path]
        3. to calculate a single path, set channel = [path, '']
        NOTE: calculatexi and calculateeta should be called before
        """
        if isinstance(channel, str):
            channel_name = channel
            if channel == "X1":
                plus_paths = [
                    self.X1_plus,
                ]
                minus_paths = [
                    self.X1_minus,
                ]
            elif channel == "X2":
                plus_paths = [
                    self.X2_plus,
                ]
                minus_paths = [
                    self.X2_minus,
                ]
            elif channel == "alpha1":
                plus_paths = [
                    self.alpha1_plus,
                ]
                minus_paths = [
                    self.alpha1_minus,
                ]
            elif channel == "alpha2":
                plus_paths = [
                    self.alpha2_plus,
                ]
                minus_paths = [
                    self.alpha2_minus,
                ]
            else:
                raise NotImplementedError("channel " + channel + " not implemented.")
        elif isinstance(channel, list):
            plus_paths = [
                channel[0],
            ]
            minus_paths = [
                channel[1],
            ]
        else:
            raise ValueError("channel should be string or list.")

        self.measurements[channel_name] = self.CalculatePathCombination(
            plus_paths=plus_paths,
            minus_paths=minus_paths,
            eta_dict=self.measurements["eta"],
            doppler=doppler,
        )
        logger.info("TDI channel " + channel_name + " calculated.")

    def CalculateClockTDI(
        self,
        channel="X2",
        doppler=True,
        modulation_correction=False,
        channel_name="tdi_channel",
    ):
        """
        channel should be string or list:
        1. string: 'X1', 'X2', ...
        2. list: [plus_path, minus_path]
        3. to calculate a single path, set channel = [path, '']
        the clock correction term will be named as channel_name + '_q'
        """
        if isinstance(channel, str):
            channel_name = channel
            if channel == "X1":
                plus_array = self.Path_string_interpret(self.X1_plus)
                minus_array = self.Path_string_interpret(self.X1_minus)
            elif channel == "X2":
                plus_array = self.Path_string_interpret(self.X2_plus)
                minus_array = self.Path_string_interpret(self.X2_minus)
            elif channel == "alpha1":
                plus_array = self.Path_string_interpret(self.alpha1_plus)
                minus_array = self.Path_string_interpret(self.alpha1_minus)
            elif channel == "alpha2":
                plus_array = self.Path_string_interpret(self.alpha2_plus)
                minus_array = self.Path_string_interpret(self.alpha2_minus)
            else:
                raise NotImplementedError("TDI channel " + channel + " not implemented.")
        elif isinstance(channel, list):
            plus_array = self.Path_string_interpret(channel[0])
            minus_array = self.Path_string_interpret(channel[1])
        else:
            raise ValueError("channel should be string or list.")

        # calculate r
        nu_m = MOSADict(modulation_freqs)
        r = (self.measurements["sci_sb"] - self.measurements["sci_c"]) / (nu_m.reverse())
        if modulation_correction:
            dM = {}
            for i, j, k in ["123", "231", "312"]:
                dM[i] = (self.measurements["ref_sb"][i + k] - self.measurements["ref_c"][i + k] - self.measurements["ref_sb"][i + j] + self.measurements["ref_c"][i + j]) / 2.0
            for i, j, k in ["123", "231", "312"]:
                delayed_dM = timeshift(dM[j], -self.fsample * self.delays[i + j], order=self.order)
                if doppler:
                    delayed_dM *= 1.0 - self.measurements["dpl"][i + j]
                r[i + j] += delayed_dM / nu_m[j + i]
                r[i + k] += -dM[i] / nu_m[k + i]
            logger.debug("modulation correction built.")
        logger.debug("intervar r calculated.")

        # construct TDI P arrays, each term will be interpreted as the path of q
        plus_array_summed = {key: [] for key in MOSA_labels}
        minus_array_summed = {key: [] for key in MOSA_labels}

        for i in range(len(plus_array)):
            tmp = plus_array[i]["delay"]
            tmp1 = ""
            for j in range(len(tmp)):
                if j == 0:
                    tmp1 += tmp[j]
                else:
                    tmp1 += tmp[j][1]
            plus_array_summed[plus_array[i]["measurement"]].append(tmp1)

        for i in range(len(minus_array)):
            tmp = minus_array[i]["delay"]
            tmp1 = ""
            for j in range(len(tmp)):
                if j == 0:
                    tmp1 += tmp[j]
                else:
                    tmp1 += tmp[j][1]
            minus_array_summed[minus_array[i]["measurement"]].append(tmp1)

        # calculate Pr
        Pr = {key: np.zeros(self.size) for key in MOSA_labels}
        for key in MOSA_labels:
            for delay in plus_array_summed[key]:
                if delay == "":
                    Pr[key] += r[key]
                else:
                    Pr[key] += self.CalculateNestedDelay(Delay_string=delay, measurement=r[key], doppler=doppler)
            for delay in minus_array_summed[key]:
                if delay == "":
                    Pr[key] += -r[key]
                else:
                    Pr[key] += -self.CalculateNestedDelay(Delay_string=delay, measurement=r[key], doppler=doppler)
        Pr = MOSADict(Pr)
        logger.debug("intervar P * r calculated.")

        # calculate R
        R = {key: np.zeros(self.size) for key in MOSA_labels}
        for key in MOSA_labels:
            paths = plus_array_summed[key]
            for path in paths:
                if path != "":
                    R[key] += self.CalculatePath(Path_string=path[::-1], eta_dict=r, doppler=doppler)
            paths = minus_array_summed[key]
            for path in paths:
                if path != "":
                    R[key] += -self.CalculatePath(Path_string=path[::-1], eta_dict=r, doppler=doppler)
        R = MOSADict(R)
        logger.debug("intervar R calculated.")

        # construct clock correction
        clock_correction = np.zeros(self.size)
        for i, j, k in ["123", "231", "312"]:
            clock_correction += (self.measurements["b"][j + k] - self.measurements["a"][i + j]) * R[i + j] - (self.measurements["b"][i + j] + self.measurements["a"][i + k]) * R[i + k] + self.measurements["b"][j + k] * Pr[i + j]
        self.measurements[channel_name + "_q"] = clock_correction
        logger.info("clock correction for channel " + channel_name + " calculated.")
        # return clock_correction

    def CalculateFullTDI(
        self,
        channel="X2",
        channel_name="tdi_channel",
        doppler=True,
        clock_correction=True,
        garbage_time=None,
    ):
        if isinstance(channel, str):
            channel_name = channel
        self.CalculateXi(doppler=doppler)
        self.CalculateEta(doppler=doppler)
        self.CalculateBasicTDI(channel=channel, doppler=doppler, channel_name=channel_name)
        if clock_correction:
            self.CalculateClockTDI(channel=channel, doppler=doppler, channel_name=channel_name)
            self.measurements[channel_name + "_c"] = self.measurements[channel_name] - self.measurements[channel_name + "_q"]
        if garbage_time is not None:
            drop_points = int(garbage_time * self.fsample)
            self.measurements[channel_name] = self.measurements[channel_name][drop_points:]
            if clock_correction:
                self.measurements[channel_name + "_c"] = self.measurements[channel_name + "_c"][drop_points:]

    def FastMichelson(self, doppler=True, channel="X"):
        """
        calculate the 2nd-generation Michelson TDI channels with an optimized algorithm
        """
        if channel in ["X", "XYZ", "AET"]:
            X0plus = (
                self.CalculateNestedDelay(
                    Delay_string="12",
                    measurement=self.measurements["eta"]["21"],
                    doppler=doppler,
                )
                + self.measurements["eta"]["12"]
            )
            X0minus = (
                self.CalculateNestedDelay(
                    Delay_string="13",
                    measurement=self.measurements["eta"]["31"],
                    doppler=doppler,
                )
                + self.measurements["eta"]["13"]
            )
            X0 = X0plus - X0minus
            X1plus = self.CalculateNestedDelay(Delay_string="131", measurement=X0plus, doppler=doppler) + X0minus
            X1minus = self.CalculateNestedDelay(Delay_string="121", measurement=X0minus, doppler=doppler) + X0plus
            X1 = X1plus - X1minus
            X2plus = self.CalculateNestedDelay(Delay_string="12131", measurement=X1plus, doppler=doppler) + X1minus
            X2minus = self.CalculateNestedDelay(Delay_string="13121", measurement=X1minus, doppler=doppler) + X1plus
            X2 = X2plus - X2minus
            self.measurements["X0"] = X0
            self.measurements["X1"] = X1
            self.measurements["X2"] = X2
            if channel in ["XYZ", "AET"]:
                X0plus = (
                    self.CalculateNestedDelay(
                        Delay_string="23",
                        measurement=self.measurements["eta"]["32"],
                        doppler=doppler,
                    )
                    + self.measurements["eta"]["23"]
                )
                X0minus = (
                    self.CalculateNestedDelay(
                        Delay_string="21",
                        measurement=self.measurements["eta"]["12"],
                        doppler=doppler,
                    )
                    + self.measurements["eta"]["21"]
                )
                Y0 = X0plus - X0minus
                X1plus = self.CalculateNestedDelay(Delay_string="212", measurement=X0plus, doppler=doppler) + X0minus
                X1minus = self.CalculateNestedDelay(Delay_string="232", measurement=X0minus, doppler=doppler) + X0plus
                Y1 = X1plus - X1minus
                X2plus = self.CalculateNestedDelay(Delay_string="23212", measurement=X1plus, doppler=doppler) + X1minus
                X2minus = self.CalculateNestedDelay(Delay_string="21232", measurement=X1minus, doppler=doppler) + X1plus
                Y2 = X2plus - X2minus
                self.measurements["Y0"] = Y0
                self.measurements["Y1"] = Y1
                self.measurements["Y2"] = Y2
                X0plus = (
                    self.CalculateNestedDelay(
                        Delay_string="31",
                        measurement=self.measurements["eta"]["13"],
                        doppler=doppler,
                    )
                    + self.measurements["eta"]["31"]
                )
                X0minus = (
                    self.CalculateNestedDelay(
                        Delay_string="32",
                        measurement=self.measurements["eta"]["23"],
                        doppler=doppler,
                    )
                    + self.measurements["eta"]["32"]
                )
                Z0 = X0plus - X0minus
                X1plus = self.CalculateNestedDelay(Delay_string="323", measurement=X0plus, doppler=doppler) + X0minus
                X1minus = self.CalculateNestedDelay(Delay_string="313", measurement=X0minus, doppler=doppler) + X0plus
                Z1 = X1plus - X1minus
                X2plus = self.CalculateNestedDelay(Delay_string="31323", measurement=X1plus, doppler=doppler) + X1minus
                X2minus = self.CalculateNestedDelay(Delay_string="32313", measurement=X1minus, doppler=doppler) + X1plus
                Z2 = X2plus - X2minus
                self.measurements["Z0"] = Z0
                self.measurements["Z1"] = Z1
                self.measurements["Z2"] = Z2
                if channel == "AET":
                    A0, E0, T0 = AETfromXYZ(X0, Y0, Z0)
                    A1, E1, T1 = AETfromXYZ(X1, Y1, Z1)
                    A2, E2, T2 = AETfromXYZ(X2, Y2, Z2)
                    self.measurements["A0"] = A0
                    self.measurements["E0"] = E0
                    self.measurements["T0"] = T0
                    self.measurements["A1"] = A1
                    self.measurements["E1"] = E1
                    self.measurements["T1"] = T1
                    self.measurements["A2"] = A2
                    self.measurements["E2"] = E2
                    self.measurements["T2"] = T2
        else:
            raise ValueError("Only X, XYZ or AET supported.")

    def CalculateTDIFromPStrings(self, PStrings, doppler=True):
        """
        args:
            eta, delays, PStrings should be MOSADicts
            shapes of each item:
            - eta: (Nt)
            - delays: (Nt)
            - PStrings: [(1, ["ij", "-jk"]), (-1, []), ...]
        """
        eta = self.measurements["eta"]  # MOSADict, each item (Nt)
        delays = self.delays  # MOSADict, each item (Nt)
        Nt = len(eta["12"])
        TDICombination = np.zeros(Nt)
        for key in MOSA_labels:
            TDIContributer_ij = np.zeros(Nt)
            for prefactor, DStrings in PStrings[key]:
                delayed_eta_ij = eta[key]
                for DString in DStrings[::-1]:
                    if DString[0] == "-":
                        delay_sign = 1.0
                    else:
                        delay_sign = -1.0
                    if doppler is True:
                        tmp_delay = delays[DString[-2] + DString[-1]]
                        tmp_doppler = np.gradient(tmp_delay) * self.fsample
                        delayed_eta_ij = timeshift(
                            data=delayed_eta_ij,
                            shifts=delay_sign * self.fsample * tmp_delay,
                            order=self.order,
                        ) * (1.0 - tmp_doppler)
                    else:
                        delayed_eta_ij = timeshift(
                            data=delayed_eta_ij,
                            shifts=delay_sign * self.fsample * delays[DString[-2] + DString[-1]],
                            order=self.order,
                        )
                TDIContributer_ij += prefactor * delayed_eta_ij
            TDICombination += TDIContributer_ij
        return TDICombination



class TDISensitivity:
    """
    calculate sensitivity for given orbit configuration and TDI scheme
    """
    X2_string = {
        "12": [(1.0, []), (-1.0, ["13", "31"]), (-1.0, ["13", "31", "12", "21"]), (1.0, ["12", "21", "13", "31", "13", "31"])],
        "23": [],
        "31": [(-1.0, ["13"]), (1.0, ["12", "21", "13"]), (1.0, ["12", "21", "13", "31", "13"]), (-1.0, ["13", "31", "12", "21", "12", "21", "13"])],
        "21": [(1.0, ["12"]), (-1.0, ["13", "31", "12"]), (-1.0, ["13", "31", "12", "21", "12"]), (1.0, ["12", "21", "13", "31", "13", "31", "12"])],
        "32": [],
        "13": [(-1.0, []), (1.0, ["12", "21"]), (1.0, ["12", "21", "13", "31"]), (-1.0, ["13", "31", "12", "21", "12", "21"])],
    }
    
    X1_string = {
        "12": [(-1.0, ["21"])],
        "23": [],
        "31": [(1.0, [])],
        "21": [(-1.0, [])],
        "32": [],
        "13": [(1.0, ["31"])],
    }

    def __init__(self, Ri, nij, dij, L=L_nominal, S_OMS=SOMS_nominal, S_ACC=SACC_nominal):
        """
        initialize with the orbit configuration.
        Args:
            Ri: SCDict, each item has shape (3,)
            nij: MosaDict, each item has shape (3,)
            dij: MosaDict, each item is a scalar
            L: nominal arm-length of the detector
            S_OMS: amplitude of optical metrology noise
            S_ACC: amplitude of test-mass acceleration noise
        """
        self.Ri, self.nij, self.dij = Ri, nij, dij
        R0 = (self.Ri["1"] + self.Ri["2"] + self.Ri["3"]) / 3.0  # (3,)
        self.qi = dict()
        for key in SC_labels:
            self.qi[key] = self.Ri[key] - R0
        self.qi = SCDict(self.qi)
        self.L = L
        self.S_OMS = S_OMS
        self.S_ACC = S_ACC

    def wave_vector(self, lam, beta):
        """
        calculate the wavevector k of indicent GW.
        """
        return np.array([-np.cos(beta) * np.cos(lam), -np.cos(beta) * np.sin(lam), -np.sin(beta)])  # (3,)

    def polar_basis(self, lam, beta):
        """
        calculate the polarization basis e+, ex.
        """
        u = np.array([np.sin(lam), -np.cos(lam), 0.0])
        v = np.array([-np.sin(beta) * np.cos(lam), -np.sin(beta) * np.sin(lam), np.cos(beta)])
        ep = np.outer(u, u) - np.outer(v, v)
        ec = np.outer(u, v) + np.outer(v, u)
        return ep, ec  # (3, 3)

    def Prefactor_ij(self, f):
        """
        Args:
            f: array of frequencies, (Nf)
        Returns:
            prefactor_ij: MosaDict, each item (Nf)
        """
        tmp = -1.0j * np.pi * f
        prefactor_ij = dict()
        for key in MOSA_labels:
            prefactor_ij[key] = tmp * self.dij[key]  # (Nf,)
        return MOSADict(prefactor_ij)

    def Exp_ij(self, k, f):
        tmp = -1.0j * np.pi * f
        exp_ij = dict()
        for key in MOSA_labels:
            exp_ij[key] = np.exp(tmp * (self.dij[key] + np.dot(k, (self.qi[key[0]] + self.qi[key[1]]) / C)))
            # exp_ij[key] = np.exp(tmp * (self.dij[key] + self.dij[key]*np.dot(k, self.nij[key] / C)))
        return MOSADict(exp_ij)

    def Sinc_ij(self, k, f):
        sinc_ij = dict()
        for key in MOSA_labels:
            sinc_ij[key] = np.sinc(f * self.dij[key] * (1.0 - np.dot(k, self.nij[key])))  # (Nf,)
        return MOSADict(sinc_ij)

    def PatternFunction_ij(self, ep, ec):
        Fp_ij = dict()
        Fc_ij = dict()
        for key in MOSA_labels:
            Fp_ij[key] = np.dot(self.nij[key], np.matmul(ep, self.nij[key]))  # scalar
            Fc_ij[key] = np.dot(self.nij[key], np.matmul(ec, self.nij[key]))  # scalar
        return MOSADict(Fp_ij), MOSADict(Fc_ij)

    def TDI_P_ij(self, P_ij_strings, f):
        """
        calculate the frequency-domain P_ij opetators.
        Args:
            P_ij_strings is a dictionary, and each item should be like: "12": [(1., ["12", "21"]), (-1., ["-13"]), ...], repersenting [D_12 D_21 - A_13 ...], representing the P_ij opetator.
        Returns:
            P_ij: MosaDict, each item (Nf)
        """
        # calculate single delay operators
        tmp = -1.0j * TWOPI * f  # -2pif
        D_ij = dict()
        for key in MOSA_labels:
            D_ij[key] = np.exp(tmp * self.dij[key])  # (Nf)
        D_ij = MOSADict(D_ij)

        P_ij = dict()
        for key in MOSA_labels:
            P_ij[key] = np.zeros_like(f, dtype=np.complex128)  # (Nf)
            for D_strings in P_ij_strings[key]:  # D_strings = (sign, ["ij", "jk", ...])
                # print('D_strings:', D_strings)
                prefactor = D_strings[0] + 0.0j
                D_operators = np.ones_like(f, dtype=np.complex128)
                for D_string in D_strings[1]:  # D_string = "ij" or "-ij"
                    # print('D_strings_content:',D_strings[1])
                    # print('D_string:', D_string)
                    # print('D_string_last2:', D_string[-2] + D_string[-1])
                    D_operator = D_ij[D_string[-2] + D_string[-1]]  # (Nf)
                    if D_string[0] == "-":
                        D_operator = np.conjugate(D_operator)
                    D_operators *= D_operator
                P_ij[key] += D_operators * prefactor  # (Nf)
        return MOSADict(P_ij)

    def TDI_response_function(self, lam, beta, f, P_ij=None, P_ij_strings=None):
        """
        calculate R from P_ij or the strings representing P_ij.
        """
        if P_ij is None:
            P_ij = self.TDI_P_ij(P_ij_strings=P_ij_strings, f=f)  # mosa, (Nf,)

        k = self.wave_vector(lam=lam, beta=beta)  # (3,)
        ep, ec = self.polar_basis(lam=lam, beta=beta)  # (3, 3)
        PreFactor = self.Prefactor_ij(f=f)  # mosa, (Nf,)
        ExpFactor = self.Exp_ij(k=k, f=f)  # mosa, (Nf,)
        SincFactor = self.Sinc_ij(k=k, f=f)  # mosa, (Nf,)
        Fp, Fc = self.PatternFunction_ij(ep=ep, ec=ec)  # mosa, scalar
        # print(Fp)
        # print(Fc)

        R2p = np.zeros_like(f, dtype=np.complex128)
        R2c = np.zeros_like(f, dtype=np.complex128)
        for key in MOSA_labels:
            tmp = P_ij[key] * PreFactor[key] * ExpFactor[key] * SincFactor[key]
            R2p += tmp * Fp[key]
            R2c += tmp * Fc[key]
        R2p = np.abs(R2p) ** 2  # (Nf)
        R2c = np.abs(R2c) ** 2
        return (R2p + R2c) / 2.0

    def TDI_noise_contributor(self, f, P_ij=None, P_ij_strings=None):
        """
        calculate oms and acc noise contributor for each MOSA from Pij or the strings representing Pij.
        """
        if P_ij is None:
            P_ij = self.TDI_P_ij(P_ij_strings=P_ij_strings, f=f)

        tmp = -1.0j * TWOPI * f  # -2pif
        D_ij = dict()
        for key in MOSA_labels:
            D_ij[key] = np.exp(tmp * self.dij[key])  # (Nf)
        D_ij = MOSADict(D_ij)

        noise_oms_ij = dict()
        noise_acc_ij = dict()
        for key in MOSA_labels:
            noise_oms_ij[key] = np.abs(P_ij[key]) ** 2  # (Nf)
            key_inverse = key[1] + key[0]
            noise_acc_ij[key] = np.abs(P_ij[key] + D_ij[key_inverse] * P_ij[key_inverse]) ** 2  # (Nf)
        return noise_oms_ij, noise_acc_ij

    def TDI_noise(self, f, P_ij=None, P_ij_strings=None):
        """
        calculate total noise PSD from Pij or the Pij strings in the fractional frequency difference unit
        """
        PSDFunction = InstrumentalPSDs(L=self.L)
        noise_oms_ij, noise_acc_ij = self.TDI_noise_contributor(f=f, P_ij=P_ij, P_ij_strings=P_ij_strings)
        PSDOMS = PSDFunction.PSD_RO(f=f, sro=self.S_OMS)
        PSDACC = PSDFunction.PSD_ACC(f=f, sacc=self.S_ACC)
        total_noise = np.zeros_like(f)
        for key in MOSA_labels:
            total_noise += noise_oms_ij[key] * PSDOMS + noise_acc_ij[key] * PSDACC  # (Nf)
        return total_noise

    def TDI_sensitivity(self, f, P_ij=None, P_ij_strings=None, Nsource=1024):
        """
        calculate sensitivity from Pij or the Pij strings. the response funciton is averaged over N sources in random directions.
        """
        # calculate Pij if absent
        if P_ij is None:
            P_ij = self.TDI_P_ij(P_ij_strings=P_ij_strings, f=f)  # mosa, (Nf,)

        # calculate instrumental noise PSD
        PSD = self.TDI_noise(f=f, P_ij=P_ij)  # (Nf)

        # calculate the average response function
        Response_arr = []
        for _ in tqdm(range(Nsource)):
            longitude = np.random.uniform(0, TWOPI)
            latitude = np.arcsin(np.random.uniform(-1, 1))
            Response_arr.append(self.TDI_response_function(lam=longitude, beta=latitude, f=f, P_ij=P_ij))
        Response_arr = np.array(Response_arr)  # (Nsource, Nf)
        Response_avg = np.mean(Response_arr, axis=0)  # (Nf)

        # return sensitivity
        return PSD / Response_avg

    def TDI_noise_CSD(self, f, P_ij=None, Q_ij=None, P_ij_strings=None, Q_ij_strings=None, return_PSD=False):
        """calculate the noise CSD of channel P and Q, i.e. 2/T <P^* Q>, reduce to PSD if P = Q"""
        if P_ij is None:
            P_ij = self.TDI_P_ij(P_ij_strings=P_ij_strings, f=f)
        if Q_ij is None:
            Q_ij = self.TDI_P_ij(P_ij_strings=Q_ij_strings, f=f)

        Dij = dict()
        for key in MOSA_labels:
            Dij[key] = np.exp(-1.0j * TWOPI * f * self.dij[key])

        PSDFunction = InstrumentalPSDs(L=self.L)
        PSDOMS = PSDFunction.PSD_RO(f=f, sro=self.S_OMS)
        PSDACC = PSDFunction.PSD_ACC(f=f, sacc=self.S_ACC)

        CSD = np.zeros_like(f, dtype=np.complex128)
        for key in MOSA_labels:
            invk = key[1] + key[0]
            CSD += np.conjugate(P_ij[key]) * Q_ij[key] * PSDOMS + np.conjugate(P_ij[key] + P_ij[invk] * Dij[invk]) * (Q_ij[key] + Q_ij[invk] * Dij[invk]) * PSDACC

        if return_PSD:
            return np.real(CSD)
        else:
            return CSD


class TDIStringManipulation:
    replacement_rule = {"1": "2", "2": "3", "3": "1"}

    @classmethod
    def replace_string(cls, s):
        return "".join(cls.replacement_rule[char] for char in s)

    @classmethod
    def replace_in_list(cls, lst):
        return [cls.replace_string(item) for item in lst]

    @classmethod
    def TDIStringCyc(cls, data):
        new_dict = {}
        for key, value in data.items():
            new_key = cls.replace_string(key)

            new_value = []
            for v in value:
                coefficient, str_list = v
                new_value.append((coefficient, cls.replace_in_list(str_list)))

            new_dict[new_key] = new_value
        return new_dict

    @classmethod
    def TDIStringScalarProduct(cls, X2_strings, a):
        new_X2_strings = dict()
        for key, value in X2_strings.items():
            new_X2_strings[key] = [(coef * a, seq) for coef, seq in value]
        return new_X2_strings

    @classmethod
    def AETStringsfromXString(cls, X2_strings):
        Y2_strings = cls.TDIStringCyc(X2_strings)
        Z2_strings = cls.TDIStringCyc(Y2_strings)
        A2_strings = dict()
        E2_strings = dict()
        T2_strings = dict()
        for key in MOSA_labels:
            A2_strings[key] = cls.TDIStringScalarProduct(X2_strings, -1.0 / np.sqrt(2.0))[key] + cls.TDIStringScalarProduct(Z2_strings, 1.0 / np.sqrt(2.0))[key]
            E2_strings[key] = cls.TDIStringScalarProduct(X2_strings, 1.0 / np.sqrt(6.0))[key] + cls.TDIStringScalarProduct(Y2_strings, -2.0 / np.sqrt(6.0))[key] + cls.TDIStringScalarProduct(Z2_strings, 1.0 / np.sqrt(6.0))[key]
            T2_strings[key] = cls.TDIStringScalarProduct(X2_strings, 1.0 / np.sqrt(3.0))[key] + cls.TDIStringScalarProduct(Y2_strings, 1.0 / np.sqrt(3.0))[key] + cls.TDIStringScalarProduct(Z2_strings, 1.0 / np.sqrt(3.0))[key]
        return A2_strings, E2_strings, T2_strings



class InstrumentalPSDs:
    """
    the default unit is ffd (fractional frequency difference)
    ffd is also the default unit used by the inteferometer class and the TDI noise class
    f can be either a float number or a numpy array
    """

    def __init__(
        self,
        L=L_nominal,
        unit="ffd",
    ):
        self.L = L
        self.unit = unit

    def PSD_ACC(self, f, sacc=SACC_nominal):
        if self.unit == "ffd":
            u = TWOPI * f * self.L / C
            return (sacc * self.L / u / C**2) ** 2 * (1.0 + (0.4e-3 / f) ** 2) * (1.0 + (f / 8e-3) ** 4)
        elif self.unit == "frequency":
            u = TWOPI * f * self.L / C
            return (sacc * self.L / u / C**2) ** 2 * (1.0 + (0.4e-3 / f) ** 2) * (1.0 + (f / 8e-3) ** 4) * F_LASER**2

    def PSD_RO(self, f, sro=SOMS_nominal):
        if self.unit == "ffd":
            u = TWOPI * f * self.L / C
            return (u * sro / self.L) ** 2 * (1.0 + (2e-3 / f) ** 4)
        elif self.unit == "frequency":
            u = TWOPI * f * self.L / C
            return (u * sro / self.L) ** 2 * (1.0 + (2e-3 / f) ** 4) * F_LASER**2

    def PSD_LASER(self, f, slaser=SLASER_nominal):  # white in frequency
        if self.unit == "ffd":
            if isinstance(f, np.ndarray):
                return np.ones_like(f) * (slaser / F_LASER) ** 2
            else:
                return (slaser / F_LASER) ** 2
        elif self.unit == "frequency":
            if isinstance(f, np.ndarray):
                return np.ones_like(f) * (slaser) ** 2
            else:
                return (slaser) ** 2

    def PSD_OB(self, f, sob=SOB_nominal):  # white in phase
        if self.unit == "ffd":
            return (sob * f / F_LASER) ** 2
        elif self.unit == "frequency":
            return (sob * f) ** 2

    def PSD_CLOCK(self, f, sclock=SCLOCK_nominal):
        if self.unit == "ffd":
            result = sclock**2 / f
        elif self.unit == "frequency":
            result = sclock**2 / f * F_LASER**2
        return result

    def PSD_BL(self, f, sbl=SBL_nominal):
        if self.unit == "ffd":
            return (TWOPI * f * sbl / C) ** 2 * (1.0 + (2e-3 / f) ** 4)
        elif self.unit == "frequency":
            return (TWOPI * f * sbl / C) ** 2 * (1.0 + (2e-3 / f) ** 4) * F_LASER**2

    def PSD_R(self, f, sr=SR_nominal):
        if self.unit == "ffd":
            return (TWOPI * f * sr / C) ** 2
        elif self.unit == "frequency":
            return (TWOPI * f * sr / C) ** 2 * F_LASER**2

    def PSD_OP(self, f, sop=SOP_OTHER_nominal):
        if self.unit == "ffd":
            return (TWOPI * f * sop / C) ** 2
        elif self.unit == "frequency":
            return (TWOPI * f * sop / C) ** 2 * F_LASER**2

    def PSD_M(self, f, sm=SM_RIGHT_nominal):
        if self.unit == "ffd":
            return sm**2 * f ** (2.0 / 3.0)
        elif self.unit == "frequency":
            return sm**2 * f ** (2.0 / 3.0) * F_LASER**2


class TDIPSDs:
    """
    TDI PSDs in [fractional frequency difference], considering only the optical metrology system noise and test-mass acceleration noise
    """

    def __init__(self, sacc=SACC_nominal, sro=SOMS_nominal, L=L_nominal):  # default Taiji [sacc] = acceleration, [sro] = distance
        self.sa = sacc
        self.so = sro
        self.L = L
        self.BasicNoisePSDs = InstrumentalPSDs(L=L, unit="ffd")

    def PSD_Sa(self, f):
        return self.BasicNoisePSDs.PSD_ACC(f=f, sacc=self.sa)

    def PSD_So(self, f):
        return self.BasicNoisePSDs.PSD_RO(f=f, sro=self.so)

    def PSD_X(self, f):  # self.sa, self.so are the asd of acc and opt noise, WTN and wang and Vallisneri
        u = TWOPI * f * self.L / C
        Sa = self.PSD_Sa(f)
        So = self.PSD_So(f)
        return Sa * (8.0 * (np.sin(2.0 * u)) ** 2 + 32.0 * (np.sin(u)) ** 2) + 16.0 * So * (np.sin(u)) ** 2

    def PSD_X2(self, f):
        u = TWOPI * f * self.L / C
        Sa = self.PSD_Sa(f)
        So = self.PSD_So(f)
        return 64.0 * (np.sin(2.0 * u)) ** 2 * (np.sin(u)) ** 2 * (So + (3.0 + np.cos(2.0 * u)) * Sa)

    def PSD_X2_unequal(self, f, arms):
        """
        arms should be given by a MOSADict object
        arms[ij] = L_ij / C
        """
        So = self.PSD_So(f)
        Sa = self.PSD_Sa(f)
        u121 = TWOPI * f * (arms["12"] + arms["21"])
        u131 = TWOPI * f * (arms["13"] + arms["31"])
        u12131 = u121 + u131
        s121 = np.sin(u121 / 2.0) ** 2
        s131 = np.sin(u131 / 2.0) ** 2
        c121 = np.cos(u121 / 2.0) ** 2
        c131 = np.cos(u131 / 2.0) ** 2
        s12131 = np.sin(u12131 / 2.0) ** 2
        OMS_term = 32.0 * So * s12131 * (s121 + s131)
        ACC_term = 64.0 * Sa * s12131 * (s121 * (c131 + 1) + s131 * (c121 + 1))
        return OMS_term + ACC_term

    def PSD_Y2_unequal(self, f, arms):
        """
        arms should be given by a MOSADict object
        arms[ij] = L_ij / C
        """
        So = self.PSD_So(f)
        Sa = self.PSD_Sa(f)
        u232 = TWOPI * f * (arms["23"] + arms["32"])
        u212 = TWOPI * f * (arms["21"] + arms["12"])
        u23212 = u232 + u212
        s232 = np.sin(u232 / 2.0) ** 2
        s212 = np.sin(u212 / 2.0) ** 2
        c232 = np.cos(u232 / 2.0) ** 2
        c212 = np.cos(u212 / 2.0) ** 2
        s23212 = np.sin(u23212 / 2.0) ** 2
        OMS_term = 32.0 * So * s23212 * (s232 + s212)
        ACC_term = 64.0 * Sa * s23212 * (s232 * (c212 + 1) + s212 * (c232 + 1))
        return OMS_term + ACC_term

    def PSD_Z2_unequal(self, f, arms):
        """
        arms should be given by a MOSADict object
        arms[ij] = L_ij / C
        """
        So = self.PSD_So(f)
        Sa = self.PSD_Sa(f)
        u313 = TWOPI * f * (arms["31"] + arms["13"])
        u323 = TWOPI * f * (arms["32"] + arms["23"])
        u31323 = u313 + u323
        s313 = np.sin(u313 / 2.0) ** 2
        s323 = np.sin(u323 / 2.0) ** 2
        c313 = np.cos(u313 / 2.0) ** 2
        c323 = np.cos(u323 / 2.0) ** 2
        s31323 = np.sin(u31323 / 2.0) ** 2
        OMS_term = 32.0 * So * s31323 * (s313 + s323)
        ACC_term = 64.0 * Sa * s31323 * (s313 * (c323 + 1) + s323 * (c313 + 1))
        return OMS_term + ACC_term

    def PSD_X2Y2star_unequal(self, f, arms):
        """
        arms should be given by a MOSADict object
        arms[ij] = L_ij / C
        """
        So = self.PSD_So(f)
        Sa = self.PSD_Sa(f)
        u121 = TWOPI * f * (arms["12"] + arms["21"])
        u131 = TWOPI * f * (arms["13"] + arms["31"])
        u12131 = u121 + u131
        u232 = TWOPI * f * (arms["23"] + arms["32"])
        u212 = TWOPI * f * (arms["21"] + arms["12"])
        u23212 = u232 + u212
        D21 = np.exp(-1.0j * TWOPI * f * arms["21"])
        D12 = np.exp(-1.0j * TWOPI * f * arms["12"])
        D131 = np.exp(-1.0j * u131)
        D12131 = np.exp(-1.0j * u12131)
        D232 = np.exp(-1.0j * u232)
        D23212 = np.exp(-1.0j * u23212)
        delay_factor = (1.0 - D131) * (1.0 - D12131) * (1.0 - D232.conjugate()) * (1.0 - D23212.conjugate()) * (D21.conjugate() + D12)
        OMS_term = -So
        ACC_term = -4.0 * Sa
        return (OMS_term + ACC_term) * delay_factor

    def PSD_Y2Z2star_unequal(self, f, arms):
        """
        arms should be given by a MOSADict object
        arms[ij] = L_ij / C
        """
        So = self.PSD_So(f)
        Sa = self.PSD_Sa(f)
        u232 = TWOPI * f * (arms["23"] + arms["32"])
        u212 = TWOPI * f * (arms["21"] + arms["12"])
        u23212 = u232 + u212
        u313 = TWOPI * f * (arms["31"] + arms["13"])
        u323 = TWOPI * f * (arms["32"] + arms["23"])
        u31323 = u313 + u323
        D32 = np.exp(-1.0j * TWOPI * f * arms["32"])
        D23 = np.exp(-1.0j * TWOPI * f * arms["23"])
        D212 = np.exp(-1.0j * u212)
        D23212 = np.exp(-1.0j * u23212)
        D313 = np.exp(-1.0j * u313)
        D31323 = np.exp(-1.0j * u31323)
        delay_factor = (1.0 - D212) * (1.0 - D23212) * (1.0 - D313.conjugate()) * (1.0 - D31323.conjugate()) * (D32.conjugate() + D23)
        OMS_term = -So
        ACC_term = -4.0 * Sa
        return (OMS_term + ACC_term) * delay_factor

    def PSD_Z2X2star_unequal(self, f, arms):
        """
        arms should be given by a MOSADict object
        arms[ij] = L_ij / C
        """
        So = self.PSD_So(f)
        Sa = self.PSD_Sa(f)
        u313 = TWOPI * f * (arms["31"] + arms["13"])
        u323 = TWOPI * f * (arms["32"] + arms["23"])
        u31323 = u313 + u323
        u121 = TWOPI * f * (arms["12"] + arms["21"])
        u131 = TWOPI * f * (arms["13"] + arms["31"])
        u12131 = u121 + u131
        D13 = np.exp(-1.0j * TWOPI * f * arms["13"])
        D31 = np.exp(-1.0j * TWOPI * f * arms["31"])
        D323 = np.exp(-1.0j * u323)
        D31323 = np.exp(-1.0j * u31323)
        D121 = np.exp(-1.0j * u121)
        D12131 = np.exp(-1.0j * u12131)
        delay_factor = (1.0 - D323) * (1.0 - D31323) * (1.0 - D121.conjugate()) * (1.0 - D12131.conjugate()) * (D13.conjugate() + D31)
        OMS_term = -So
        ACC_term = -4.0 * Sa
        return (OMS_term + ACC_term) * delay_factor

    def PSD_A(self, f):  # WTN
        u = TWOPI * f * self.L / C
        Sa = self.PSD_Sa(f)
        So = self.PSD_So(f)
        return 8.0 * So * (2.0 + np.cos(u)) * (np.sin(u)) ** 2 + 16.0 * Sa * (3.0 + 2.0 * np.cos(u) + np.cos(2.0 * u)) * (np.sin(u)) ** 2

    def PSD_A2(self, f):
        u = TWOPI * f * self.L / C
        return np.maximum(4.0 * (np.sin(2.0 * u)) ** 2 * self.PSD_A(f), 1e-50)

    def PSD_A2_unequal(self, f, arms):
        return 0.5 * self.PSD_X2_unequal(f, arms) + 0.5 * self.PSD_Z2_unequal(f, arms) - np.real(self.PSD_Z2X2star_unequal(f, arms))

    def PSD_E2_unequal(self, f, arms):
        return (
            self.PSD_X2_unequal(f, arms)
            + 4.0 * self.PSD_Y2_unequal(f, arms)
            + self.PSD_Z2_unequal(f, arms)
            - 4.0 * np.real(self.PSD_X2Y2star_unequal(f, arms))
            + 2.0 * np.real(self.PSD_Z2X2star_unequal(f, arms))
            - 4.0 * np.real(self.PSD_Y2Z2star_unequal(f, arms))
        ) / 6.0

    def PSD_T2_unequal(self, f, arms):
        return (
            self.PSD_X2_unequal(f, arms)
            + self.PSD_Y2_unequal(f, arms)
            + self.PSD_Z2_unequal(f, arms)
            + 2.0 * np.real(self.PSD_X2Y2star_unequal(f, arms))
            + 2.0 * np.real(self.PSD_Z2X2star_unequal(f, arms))
            + 2.0 * np.real(self.PSD_Y2Z2star_unequal(f, arms))
        ) / 3.0

    def PSD_X2_unequal_different_amps(self, f, arms, A_OMS, A_ACC):
        """
        arms, amplitudes should be given by MOSADict objects
        arms[ij] = L_ij / C
        """
        So = self.PSD_So(f) / self.so**2  # oms noise with unit amplitude
        Sa = self.PSD_Sa(f) / self.sa**2  # acc noise with unit amplitude
        u121 = TWOPI * f * (arms["12"] + arms["21"])
        u131 = TWOPI * f * (arms["13"] + arms["31"])
        u12131 = u121 + u131
        s121 = np.sin(u121 / 2.0) ** 2
        s131 = np.sin(u131 / 2.0) ** 2
        c121 = np.cos(u121 / 2.0) ** 2
        c131 = np.cos(u131 / 2.0) ** 2
        s12131 = np.sin(u12131 / 2.0) ** 2
        OMS_term = 16.0 * s12131 * (s131 * (A_OMS["12"] ** 2 + A_OMS["21"] ** 2) + s121 * (A_OMS["13"] ** 2 + A_OMS["31"] ** 2))
        ACC_term = 64.0 * s12131 * (s131 * (c121 * A_ACC["12"] ** 2 + A_ACC["21"] ** 2) + s121 * (c131 * A_ACC["13"] ** 2 + A_ACC["31"] ** 2))
        return OMS_term * So + ACC_term * Sa

    def PSD_Y2_unequal_different_amps(self, f, arms, A_OMS, A_ACC):
        """
        arms should be given by a MOSADict object
        arms[ij] = L_ij / C
        """
        So = self.PSD_So(f) / self.so**2  # oms noise with unit amplitude
        Sa = self.PSD_Sa(f) / self.sa**2  # acc noise with unit amplitude
        u232 = TWOPI * f * (arms["23"] + arms["32"])
        u212 = TWOPI * f * (arms["21"] + arms["12"])
        u23212 = u232 + u212
        s232 = np.sin(u232 / 2.0) ** 2
        s212 = np.sin(u212 / 2.0) ** 2
        c232 = np.cos(u232 / 2.0) ** 2
        c212 = np.cos(u212 / 2.0) ** 2
        s23212 = np.sin(u23212 / 2.0) ** 2
        OMS_term = 16.0 * s23212 * (s212 * (A_OMS["23"] ** 2 + A_OMS["32"] ** 2) + s232 * (A_OMS["21"] ** 2 + A_OMS["12"] ** 2))
        ACC_term = 64.0 * s23212 * (s212 * (c232 * A_ACC["23"] ** 2 + A_ACC["32"] ** 2) + s232 * (c212 * A_ACC["21"] ** 2 + A_ACC["12"] ** 2))
        return OMS_term * So + ACC_term * Sa

    def PSD_Z2_unequal_different_amps(self, f, arms, A_OMS, A_ACC):
        """
        arms should be given by a MOSADict object
        arms[ij] = L_ij / C
        """
        So = self.PSD_So(f) / self.so**2  # oms noise with unit amplitude
        Sa = self.PSD_Sa(f) / self.sa**2  # acc noise with unit amplitude
        u313 = TWOPI * f * (arms["31"] + arms["13"])
        u323 = TWOPI * f * (arms["32"] + arms["23"])
        u31323 = u313 + u323
        s313 = np.sin(u313 / 2.0) ** 2
        s323 = np.sin(u323 / 2.0) ** 2
        c313 = np.cos(u313 / 2.0) ** 2
        c323 = np.cos(u323 / 2.0) ** 2
        s31323 = np.sin(u31323 / 2.0) ** 2
        OMS_term = 16.0 * s31323 * (s323 * (A_OMS["31"] ** 2 + A_OMS["13"] ** 2) + s313 * (A_OMS["32"] ** 2 + A_OMS["23"] ** 2))
        ACC_term = 64.0 * s31323 * (s323 * (c313 * A_ACC["31"] ** 2 + A_ACC["13"] ** 2) + s313 * (c323 * A_ACC["32"] ** 2 + A_ACC["23"] ** 2))
        return OMS_term * So + ACC_term * Sa

    def PSD_X2Y2star_unequal_different_amps(self, f, arms, A_OMS, A_ACC):
        """
        arms should be given by a MOSADict object
        arms[ij] = L_ij / C
        """
        So = self.PSD_So(f) / self.so**2  # oms noise with unit amplitude
        Sa = self.PSD_Sa(f) / self.sa**2  # acc noise with unit amplitude
        u121 = TWOPI * f * (arms["12"] + arms["21"])
        u131 = TWOPI * f * (arms["13"] + arms["31"])
        u12131 = u121 + u131
        u232 = TWOPI * f * (arms["23"] + arms["32"])
        u212 = TWOPI * f * (arms["21"] + arms["12"])
        u23212 = u232 + u212
        D21 = np.exp(-1.0j * TWOPI * f * arms["21"])
        D12 = np.exp(-1.0j * TWOPI * f * arms["12"])
        D131 = np.exp(-1.0j * u131)
        D12131 = np.exp(-1.0j * u12131)
        D232 = np.exp(-1.0j * u232)
        D23212 = np.exp(-1.0j * u23212)
        OMS_term = -1.0 * (1.0 - D131) * (1.0 - D12131) * np.conjugate((1.0 - D232) * (1.0 - D23212)) * (np.conjugate(D21) * A_OMS["12"] ** 2 + D12 * A_OMS["21"] ** 2)
        ACC_term = -2.0 * (1.0 - D131) * (1.0 - D12131) * np.conjugate((1.0 - D232) * (1.0 - D23212)) * ((np.conjugate(D21) + D12) * (A_ACC["12"] ** 2 + A_ACC["21"] ** 2))
        return OMS_term * So + ACC_term * Sa

    def PSD_Y2Z2star_unequal_different_amps(self, f, arms, A_OMS, A_ACC):
        """
        arms should be given by a MOSADict object
        arms[ij] = L_ij / C
        """
        So = self.PSD_So(f) / self.so**2  # oms noise with unit amplitude
        Sa = self.PSD_Sa(f) / self.sa**2  # acc noise with unit amplitude
        u232 = TWOPI * f * (arms["23"] + arms["32"])
        u212 = TWOPI * f * (arms["21"] + arms["12"])
        u23212 = u232 + u212
        u313 = TWOPI * f * (arms["31"] + arms["13"])
        u323 = TWOPI * f * (arms["32"] + arms["23"])
        u31323 = u313 + u323
        D32 = np.exp(-1.0j * TWOPI * f * arms["32"])
        D23 = np.exp(-1.0j * TWOPI * f * arms["23"])
        D212 = np.exp(-1.0j * u212)
        D23212 = np.exp(-1.0j * u23212)
        D313 = np.exp(-1.0j * u313)
        D31323 = np.exp(-1.0j * u31323)
        OMS_term = -1.0 * (1.0 - D212) * (1.0 - D23212) * np.conjugate((1.0 - D313) * (1.0 - D31323)) * (np.conjugate(D32) * A_OMS["23"] ** 2 + D23 * A_OMS["32"] ** 2)
        ACC_term = -2.0 * (1.0 - D212) * (1.0 - D23212) * np.conjugate((1.0 - D313) * (1.0 - D31323)) * ((np.conjugate(D32) + D23) * (A_ACC["23"] ** 2 + A_ACC["32"] ** 2))
        return OMS_term * So + ACC_term * Sa

    def PSD_Z2X2star_unequal_different_amps(self, f, arms, A_OMS, A_ACC):
        """
        arms should be given by a MOSADict object
        arms[ij] = L_ij / C
        """
        So = self.PSD_So(f) / self.so**2  # oms noise with unit amplitude
        Sa = self.PSD_Sa(f) / self.sa**2  # acc noise with unit amplitude
        u313 = TWOPI * f * (arms["31"] + arms["13"])
        u323 = TWOPI * f * (arms["32"] + arms["23"])
        u31323 = u313 + u323
        u121 = TWOPI * f * (arms["12"] + arms["21"])
        u131 = TWOPI * f * (arms["13"] + arms["31"])
        u12131 = u121 + u131
        D13 = np.exp(-1.0j * TWOPI * f * arms["13"])
        D31 = np.exp(-1.0j * TWOPI * f * arms["31"])
        D323 = np.exp(-1.0j * u323)
        D31323 = np.exp(-1.0j * u31323)
        D121 = np.exp(-1.0j * u121)
        D12131 = np.exp(-1.0j * u12131)
        OMS_term = -1.0 * (1.0 - D323) * (1.0 - D31323) * np.conjugate((1.0 - D121) * (1.0 - D12131)) * (np.conjugate(D13) * A_OMS["31"] ** 2 + D31 * A_OMS["13"] ** 2)
        ACC_term = -2.0 * (1.0 - D323) * (1.0 - D31323) * np.conjugate((1.0 - D121) * (1.0 - D12131)) * ((np.conjugate(D13) + D31) * (A_ACC["31"] ** 2 + A_ACC["13"] ** 2))
        return OMS_term * So + ACC_term * Sa

    def PSD_A2_unequal_different_amps(self, f, arms, A_OMS, A_ACC):
        return 0.5 * self.PSD_X2_unequal_different_amps(f, arms, A_OMS, A_ACC) + 0.5 * self.PSD_Z2_unequal_different_amps(f, arms, A_OMS, A_ACC) - np.real(self.PSD_Z2X2star_unequal_different_amps(f, arms, A_OMS, A_ACC))

    def PSD_E2_unequal_different_amps(self, f, arms, A_OMS, A_ACC):
        return (
            self.PSD_X2_unequal_different_amps(f, arms, A_OMS, A_ACC)
            + 4.0 * self.PSD_Y2_unequal_different_amps(f, arms, A_OMS, A_ACC)
            + self.PSD_Z2_unequal_different_amps(f, arms, A_OMS, A_ACC)
            - 4.0 * np.real(self.PSD_X2Y2star_unequal_different_amps(f, arms, A_OMS, A_ACC))
            + 2.0 * np.real(self.PSD_Z2X2star_unequal_different_amps(f, arms, A_OMS, A_ACC))
            - 4.0 * np.real(self.PSD_Y2Z2star_unequal_different_amps(f, arms, A_OMS, A_ACC))
        ) / 6.0

    def PSD_T2_unequal_different_amps(self, f, arms, A_OMS, A_ACC):
        return (
            self.PSD_X2_unequal_different_amps(f, arms, A_OMS, A_ACC)
            + self.PSD_Y2_unequal_different_amps(f, arms, A_OMS, A_ACC)
            + self.PSD_Z2_unequal_different_amps(f, arms, A_OMS, A_ACC)
            + 2.0 * np.real(self.PSD_X2Y2star_unequal_different_amps(f, arms, A_OMS, A_ACC))
            + 2.0 * np.real(self.PSD_Z2X2star_unequal_different_amps(f, arms, A_OMS, A_ACC))
            + 2.0 * np.real(self.PSD_Y2Z2star_unequal_different_amps(f, arms, A_OMS, A_ACC))
        ) / 3.0

    def PSD_T(self, f):  # WTN
        u = TWOPI * f * self.L / C
        Sa = self.PSD_Sa(f)
        So = self.PSD_So(f)
        return 16.0 * So * (1.0 - np.cos(u)) * (np.sin(u)) ** 2 + 128.0 * Sa * (np.sin(u)) ** 2 * (np.sin(u / 2.0)) ** 4

    def PSD_T2(self, f):
        u = TWOPI * f * self.L / C
        return 4.0 * (np.sin(2.0 * u)) ** 2 * self.PSD_T(f)

    def PSD_alpha(self, f):  # wang, Vallis
        u = TWOPI * f * self.L / C
        Sa = self.PSD_Sa(f)
        So = self.PSD_So(f)
        return 8.0 * Sa * ((np.sin(1.5 * u)) ** 2 + 2.0 * (np.sin(0.5 * u)) ** 2) + 6.0 * So

    def PSD_alpha2(self, f):
        u = TWOPI * f * self.L / C
        return self.PSD_alpha(f) * 4.0 * np.sin(1.5 * u) ** 2

    def PSD_alpha2_unequal(self, f, arms):
        Sa = self.PSD_Sa(f)
        So = self.PSD_So(f)
        u12 = TWOPI * f * arms["12"]
        u13 = TWOPI * f * arms["13"]
        u23 = TWOPI * f * arms["23"]
        u123 = u12 + u23
        u132 = u13 + u23
        u1231 = u123 + u13
        OMS_term = 24.0 * So * np.sin(u1231 / 2.0) ** 2
        ACC_term = 32.0 * Sa * np.sin(u1231 / 2.0) ** 2 * (np.sin(u1231 / 2.0) ** 2 + np.sin((u132 - u12) / 2.0) ** 2 + np.sin((u123 - u13) / 2.0) ** 2)
        return OMS_term + ACC_term

    def PSD_beta2_unequal(self, f, arms):
        Sa = self.PSD_Sa(f)
        So = self.PSD_So(f)
        u23 = TWOPI * f * arms["23"]
        u21 = TWOPI * f * arms["21"]
        u31 = TWOPI * f * arms["31"]
        u231 = u23 + u31
        u213 = u21 + u31
        u2312 = u231 + u21
        OMS_term = 24.0 * So * np.sin(u2312 / 2.0) ** 2
        ACC_term = 32.0 * Sa * np.sin(u2312 / 2.0) ** 2 * (np.sin(u2312 / 2.0) ** 2 + np.sin((u213 - u23) / 2.0) ** 2 + np.sin((u231 - u21) / 2.0) ** 2)
        return OMS_term + ACC_term

    def PSD_gamma2_unequal(self, f, arms):
        Sa = self.PSD_Sa(f)
        So = self.PSD_So(f)
        u31 = TWOPI * f * arms["31"]
        u32 = TWOPI * f * arms["32"]
        u12 = TWOPI * f * arms["12"]
        u312 = u31 + u12
        u321 = u32 + u12
        u3123 = u312 + u32
        OMS_term = 24.0 * So * np.sin(u3123 / 2.0) ** 2
        ACC_term = 32.0 * Sa * np.sin(u3123 / 2.0) ** 2 * (np.sin(u3123 / 2.0) ** 2 + np.sin((u321 - u31) / 2.0) ** 2 + np.sin((u312 - u32) / 2.0) ** 2)
        return OMS_term + ACC_term

    def PSD_alpha2gamma2star_unequal(self, f, arms):
        Sa = self.PSD_Sa(f)
        So = self.PSD_So(f)
        u12 = TWOPI * f * arms["12"]
        u13 = TWOPI * f * arms["13"]
        u23 = TWOPI * f * arms["23"]
        u123 = u12 + u23
        u132 = u13 + u23
        u1231 = u123 + u13
        prefactor = 4.0 * np.sin(u1231 / 2.0) ** 2
        OMS_term = 2.0 * So * (2.0 * np.cos(u13) + np.cos(u123))
        ACC_term = 4.0 * Sa * (np.cos(u13) - np.cos(u23 - u12))
        return prefactor * (OMS_term + ACC_term)

    def PSD_P22_unequal(self, f, arms):
        return 0.5 * (self.PSD_alpha2_unequal(f, arms) + self.PSD_gamma2_unequal(f, arms) + 2.0 * self.PSD_alpha2gamma2star_unequal(f, arms))

