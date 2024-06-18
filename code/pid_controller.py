from dataclasses import dataclass, field
from typing import List, Dict, Union
import numpy as np
from scipy.integrate import trapezoid

@dataclass
class PIDController:
    """
    Proportional Integral Derivative controller. Tracks a reference signal and outputs
    a control signal to minimize output. The equation is:

        err = reference - measurement

        u = k_p * err + k_d * d(err)/dt + k_i * int(err . dt)
    
    Can control a single or an array of signals, given float or array PID constants.
    """

    k_p: np.ndarray
    "Proportional constant"
    k_i: np.ndarray
    "Integral constant"
    k_d: np.ndarray
    "Derivative constant"
    max_err_i: np.ndarray
    "Maximum accumulated error"

    def __post_init__(self):
        self.err_p = np.atleast_1d(np.zeros_like(self.k_p))
        self.err_i = np.atleast_1d(np.zeros_like(self.k_i))
        self.err_d = np.atleast_1d(np.zeros_like(self.k_d))
        self.err = np.atleast_1d(np.zeros_like(self.k_p))
        self.dtype = self.err_p.dtype
        if self.max_err_i is None:
            self.max_err_i = np.atleast_1d(np.inf, self.dtype)
        else:
            self.max_err_i = np.asarray(self.max_err_i, dtype=self.err.dtype)
        self.reference = np.zeros_like(self.err)
        self.action = np.zeros_like(self.err)
        self._params = ('k_p', 'k_i', 'k_d', 'max_err_i')


    def reset(self):
        self.action = np.zeros_like(self.err, self.dtype)
        self.reference *= 0
        self.err *= 0
        self.err_p *= 0
        self.err_i *= 0
        self.err_d *= 0


    def set_params(self, **params: Dict[str, Union[np.ndarray, bool, float, int]]):
        for name, param in params.items():
            if hasattr(self, name):
                attr = getattr(self, name)
                if isinstance(attr, np.ndarray):
                    param = np.asarray(param, dtype=attr.dtype)
                    # cast to an axis & assign in-place
                    # for cases where float param is assigned to array attr
                    if attr.ndim > 0:
                        attr[:] = param
                    else:
                        attr = param
                else:
                    attr = param
                setattr(self, name, attr)
            else:
                raise AttributeError('Attribute %s not part of class.' % name)


    def get_params(self) -> Dict[str, np.ndarray]:
        return {name: getattr(self, name) for name in self._params}


    @property
    def state(self) -> np.ndarray:
        return np.concatenate((
            np.atleast_1d(self.err_p),
            np.atleast_1d(self.err_i),
            np.atleast_1d(self.err_d)
        ))


    def step(
        self, reference: np.ndarray, measurement: np.ndarray, dt: float=1.,
        ref_is_error: bool=False, persist: bool=True
    ) -> np.ndarray:
        """
        Calculate the output, based on the current measurement and the reference
        signal.

        Parameters
        ----------
        reference : np.ndarray
            The reference signal(s) to track. Can be a number or an array.
        measurement : np.ndarray
            The actual measurement(s).
        ref_is_error: bool
            Whether to interpret the reference input as the error.
        persist: bool
            Whether to store the current state for the next step.

        Returns
        -------
        np.ndarray
            The action signal.
        """
        if ref_is_error:
            err = reference
        else:
            err = reference - measurement
        err_p = self.k_p * err
        err_i = self.k_i * np.clip(
            self.err_i + trapezoid((self.err, err), dx=dt, axis=0),
            a_min=-self.max_err_i, a_max=self.max_err_i
        )
        err_d = self.k_d * (err - self.err) / dt
        action = err_p + err_i + err_d
        
        if persist:
            self.err_p, self.err_i, self.err_d, self.action = \
                err_p, err_i, err_d, action
            self.reference = reference
            self.err = err
        return action

@dataclass
class PID_Params:
    pos_p: List[float] = field(default_factory=lambda: [0.3, 0.3, 0.2])
    pos_i: List[int] = field(default_factory=lambda: [0, 0, 0])
    pos_d: List[int] = field(default_factory=lambda: [0, 0, 0])
    
    vel_p: List[int] = field(default_factory=lambda: [1, 1, 100])
    vel_i: List[float] = field(default_factory=lambda: [0.1, 0.1, 0])
    vel_d: List[int] = field(default_factory=lambda: [0, 0, 0])
    
    att_p: float = 5
    
    rate_p: float = 8
