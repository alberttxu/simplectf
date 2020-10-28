import numpy as np

from simplectf import wavelength_from_voltage, chi, ctf, quadratic_form


def test_wavelength_from_voltge():
    # verify with JEOL table: https://www.jeol.co.jp/en/words/emterms/search_result.html?keyword=wavelength%20of%20electron
    assert np.isclose(wavelength_from_voltage(200), 2.5079, rtol=1e-4)
    assert np.isclose(wavelength_from_voltage(300), 1.9687, rtol=1e-4)


def test_chi():
    defocus_matrix = np.array([[1, 0], [0, 1]])
    kx = 1 / 10
    ky = 1 / 10
    electron_wavelength = 2.5079
    spherical_abberation = 2.7
    phase_shift = 0
    assert np.isclose(
        chi(defocus_matrix, kx, ky, electron_wavelength, spherical_abberation, phase_shift),
        -2.4653,
        rtol=1e-4,
    )


def test_ctf():
    defocus_matrix = np.array([[1, 0], [0, 1]])
    kx = 1 / 10
    ky = 1 / 10
    electron_wavelength = 2.5079
    spherical_abberation = 2.7
    phase_shift = 0
    chi_args = defocus_matrix, kx, ky, electron_wavelength, spherical_abberation, phase_shift
    amplitude_contrast = 0.07
    assert np.isclose(ctf(amplitude_contrast, chi_args), 0.14735, rtol=1e-4)


def test_quadratic_form():
    np.random.seed(0)
    x = np.random.rand(2)
    A = np.array([[1, 2], [2, 3]])
    print(quadratic_form(A, x[0], x[1]))
    print(x @ A @ x)
    assert np.isclose(quadratic_form(A, x[0], x[1]), x @ A @ x)
