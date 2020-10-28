import mrcfile
import numpy as np
from scipy import fft, interpolate, ndimage, optimize, signal


def wavelength_from_voltage(voltage):
    """Given an accelerating voltage in kV, return the electron wavelength in picometers."""
    return 1226.39 / np.sqrt(1000.0 * voltage + 0.97845e-6 * pow(1000.0 * voltage, 2))


def ctf(amplitude_contrast, chi_args):
    """Contrast Transfer Function

    amplitude_contrast - fraction between 0 and 1
    chi_args - tuple of arguments for chi
    """
    return np.sin(2 * np.pi * chi(*chi_args) + np.pi - np.arcsin(amplitude_contrast))


def chi(defocus_matrix, kx, ky, electron_wavelength, spherical_abberation, phase_shift):
    """Phase perturbation term in cycles rather than radians.

    Units for arguments:
     defocus_matrix: 2x2 matrix whose eigenvalues z1, z2 are in microns
        z1 > z2
        Orthonormal eigenvectors associated with z1 & z2 are minor and major semiaxes
     kx, ky: angstrom^-1
     electron_wavelength: picometers
     spherical_abberation: millimeters
     phase_shift: degrees (0 to 180)
    """
    return (
        -(100 * (1 / 2) * electron_wavelength * quadratic_form(defocus_matrix, kx, ky))
        + 10
        * (1 / 4)
        * spherical_abberation
        * electron_wavelength ** 3
        * (kx ** 2 + ky ** 2) ** 2
        + phase_shift / 360
    )


def quadratic_form(A, x1, x2):
    """Quadratic form of a 2x2 matrix.

    x1 and x2 can be vectorized inputs, e.g. from np.meshgrid
    """
    assert A.shape[0] == A.shape[1] == 2, "A = %r is not 2x2" % A
    assert np.isclose(A[0, 1], A[1, 0]), "A = %r is not symmetric" % A
    return A[0][0] * x1 ** 2 + 2 * A[0][1] * x1 * x2 + A[1][1] * x2 ** 2


class CTFModel:
    """Forward model for simulating CTF

    mrc - path to mrc file, or a numpy array; used for calculating spectral properties
    voltage - in kV
    spherical_abberation - in picometers
    amplitude_contrast - fraction between 0 and 1
    low_cutoff_res - in angstroms
    high_cutoff_res - in angstroms
    """

    def __init__(
        self,
        mrc,
        pixelsize,
        voltage,
        spherical_abberation,
        amplitude_contrast,
        low_cutoff_res=30,
        high_cutoff_res=5,
    ):
        self.pixelsize = pixelsize
        self.voltage = voltage
        self.spherical_abberation = spherical_abberation
        self.amplitude_contrast = amplitude_contrast
        self.electron_wavelength = wavelength_from_voltage(voltage)
        if type(mrc) == str:
            with mrcfile.open(mrc, permissive=True) as f:
                img = f.data
        elif type(mrc) == np.ndarray:
            img = mrc
        self.spectrum = np.log(
            np.abs(fft.rfft2(img, s=(max(img.shape), max(img.shape))))
        )
        # keep a copy of the full spectrum
        self.full_spectrum = self.spectrum
        # reduce the spectrum to a desired range and subtract background
        self.spectrum = self._preprocess_spectrum(
            self.spectrum, low_cutoff_res, high_cutoff_res
        )

    def _preprocess_spectrum(self, spectrum, low_cutoff_res, high_cutoff_res):
        """Cuts off low and high frequencies and then subtracts the background by
        calculating a lower envelope to the min peaks (ctf zero crossings).
        Also finds an upper envelope to be used as a damping function for computing
        forward models.
        """
        low_cutoff_freq = 1 / low_cutoff_res
        high_cutoff_freq = 1 / high_cutoff_res
        freqencies = fft.rfftfreq(n=self.full_spectrum.shape[0], d=self.pixelsize)
        # dft indices of low and high cutoffs
        self._low_idx = np.argmax(freqencies > low_cutoff_freq)
        self._high_idx = np.argmax(freqencies > high_cutoff_freq)
        # get envelopes
        bottom_envelope, self._upper_envelope = self._get_envelopes(
            spectrum, self._low_idx, self._high_idx
        )
        # cut off high frequencies
        spectrum = spectrum[:, : self._high_idx]
        spectrum = np.vstack(
            (spectrum[: self._high_idx, :], spectrum[-self._high_idx :, :])
        )
        spectrum = fft.fftshift(spectrum, axes=0)
        # subtract bottom envelope
        x = range(spectrum.shape[1])
        y = range(-spectrum.shape[0] // 2, spectrum.shape[0] // 2)
        X, Y = np.meshgrid(x, y)
        r = np.sqrt(X ** 2 + Y ** 2)
        spectrum -= bottom_envelope(r - self._low_idx)
        # zero out distances not between low_idx and high_idx
        w = np.where((r < self._low_idx) | (r > self._high_idx))
        spectrum[w[0], w[1]] = 0
        return spectrum

    def get_1d_average(self, spectrum, horizontal_res=50, sigma_ratio=1 / 60):
        """Returns a 1D signal of the vertical spectrum inside the cutoff range"""
        freqencies = fft.rfftfreq(n=self.full_spectrum.shape[0], d=self.pixelsize)
        width = np.argmax(freqencies > 1 / horizontal_res)
        vertical_strip = spectrum[self._low_idx : self._high_idx, :width]
        vertical_1d = np.average(vertical_strip, axis=1)
        sigma = sigma_ratio * len(vertical_1d)
        vertical_1d = ndimage.gaussian_filter1d(
            vertical_1d, sigma=sigma, mode="nearest"
        )
        return vertical_1d

    def _get_envelopes(self, spectrum, low_idx, high_idx):
        vertical_1d = self.get_1d_average(spectrum)
        # get bottom envelope
        zero_crossings, _ = signal.find_peaks(-1 * vertical_1d)
        assert len(zero_crossings) > 0, "could not find any min peaks"
        bottom_envelope = interpolate.interp1d(
            zero_crossings, vertical_1d[zero_crossings], fill_value="extrapolate"
        )
        # subtract off bottom envelope
        xvals = np.array(list(range(len(vertical_1d))))
        vertical_1d -= bottom_envelope(xvals)
        # get upper envelope
        peaks, _ = signal.find_peaks(vertical_1d)
        peaks = np.array([0] + list(peaks))
        upper_envelope = interpolate.interp1d(
            peaks, vertical_1d[peaks], fill_value="extrapolate"
        )
        # side effect, calculate approxiate defocus for initial value to optimization
        self._approx_defocus = self._get_approx_defocus(zero_crossings[0])
        return bottom_envelope, upper_envelope

    def _get_approx_defocus(self, first_zero_crossing_idx):
        # get approximate defocus from first zero crossing
        freqencies = fft.rfftfreq(n=self.full_spectrum.shape[0], d=self.pixelsize)
        k_first_zero_crossing = freqencies[self._low_idx + first_zero_crossing_idx]
        chi_first_zero_crossing = -0.5
        approximate_defocus = (
            chi_first_zero_crossing
            - 10
            * (1 / 4)
            * self.spherical_abberation
            * self.electron_wavelength ** 3
            * k_first_zero_crossing ** 4
        ) * (-2 / (100 * self.electron_wavelength * k_first_zero_crossing ** 2))
        return approximate_defocus

    def _ctf_array(self, z1, z2, major_semiaxis_angle, phase_shift):
        """Returns a 2D array of CTF values in the same shape as the reduced spectrum"""
        # calculate defocus matrix
        minor_semiaxis_angle = 90 + major_semiaxis_angle
        minor_semiaxis = np.array(
            [
                np.cos(minor_semiaxis_angle * np.pi / 180),
                np.sin(minor_semiaxis_angle * np.pi / 180),
            ]
        ).reshape(2, 1)
        semiaxes, _ = np.linalg.qr(np.hstack((minor_semiaxis, np.identity(2))))
        defocus_matrix = semiaxes @ [[z1, 0], [0, z2]] @ semiaxes.transpose()
        # vectorize ctf calculation
        kx = fft.rfftfreq(n=self.full_spectrum.shape[0], d=self.pixelsize)
        kx = kx[: self._high_idx]
        ky = fft.fftfreq(n=self.full_spectrum.shape[0], d=self.pixelsize)
        ky = np.hstack((ky[: self._high_idx], ky[-self._high_idx :]))
        ky = fft.fftshift(ky)
        KX, KY = np.meshgrid(kx, ky)
        chi_args = (
            defocus_matrix,
            KX,
            KY,
            self.electron_wavelength,
            self.spherical_abberation,
            phase_shift,
        )
        return ctf(self.amplitude_contrast, chi_args)

    def forward_model(self, z1, z2, major_semiaxis_angle, phase_shift):
        """Returns the ctf magnitude, damped by an upper envelope"""
        absctf = np.abs(self._ctf_array(z1, z2, major_semiaxis_angle, phase_shift))
        x = range(absctf.shape[1])
        y = range(-absctf.shape[0] // 2, absctf.shape[0] // 2)
        X, Y = np.meshgrid(x, y)
        r = np.sqrt(X ** 2 + Y ** 2)
        # damping
        absctf *= self._upper_envelope(r - self._low_idx)
        # zero out distances not between low_idx and high_idx
        w = np.where((r < self._low_idx) | (r > self._high_idx))
        absctf[w[0], w[1]] = 0
        return absctf

    def score(self, z1, z2, major_semiaxis_angle, phase_shift):
        """Returns cross correlation between forward model and background subtracted spectrum"""
        simulated_ctf = self.forward_model(z1, z2, major_semiaxis_angle, phase_shift)
        self.cross_correlation_score = signal.correlate2d(
            self.spectrum, simulated_ctf, mode="valid"
        )[0][0] / (np.linalg.norm(self.spectrum) * np.linalg.norm(simulated_ctf))
        return self.cross_correlation_score


def loss(x, ctf_model, search_phase=True):
    if search_phase:
        z1, z2, angle_astig, phase_shift = x
    else:
        z1, z2, angle_astig = x
        phase_shift = 0
    return -1 * ctf_model.score(z1, z2, angle_astig, phase_shift)


def find_ctf(ctf_model, search_phase=False):
    if search_phase:
        x0 = [ctf_model._approx_defocus, ctf_model._approx_defocus, 0, 0]
        result = optimize.minimize(
            loss,
            x0,
            args=(ctf_model, search_phase),
            method="SLSQP",
            bounds=((0, 5), (0, 5), (-90, 90), (0, 180)),
        )
        return result.x
    else:
        x0 = [ctf_model._approx_defocus, ctf_model._approx_defocus, 0]
        result = optimize.minimize(
            loss,
            x0,
            args=(ctf_model, search_phase),
            method="SLSQP",
            bounds=((0, 5), (0, 5), (-90, 90)),
        )
        return list(result.x) + [0]
