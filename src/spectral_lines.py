import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy.ndimage import percentile_filter
from scipy.ndimage import median_filter
from scipy.optimize import curve_fit
from scipy.special import voigt_profile

from math import isnan



def find_absorption_lines(wavelength, flux, smoothing_width, min_prominence):
    
    flux_smooth = median_filter(flux, size=smoothing_width)

    prominence = min_prominence * np.nanmax(flux_smooth)

    peaks, properties = find_peaks(-flux_smooth, prominence=prominence)

    peak_wavelengths = wavelength[peaks]
    peak_fluxes = flux_smooth[peaks]

    return peak_wavelengths, peak_fluxes
    
    
    
def identify_absorption_lines(peak_wavelengths, peak_fluxes, spectral_lines, z = 0.0, line_window = 2.0):
    identified_lines = []
    for name, model, lam_rest in spectral_lines:
        lam_obs = lam_rest * (1 + z)
        
        # find detected peaks close to this known line
        distance = np.abs(peak_wavelengths - lam_obs)
        match = np.where(distance < line_window)[0]
        
        if len(match) > 0:
            best = match[np.argmin(distance[match])]
            identified_lines.append(
                (name, model, lam_obs, peak_wavelengths[best], peak_fluxes[best])
            )
    
    return identified_lines
            


def gaussian_model(x, amp, mu, sigma, offset):
    return amp * np.exp(-0.5 * ((x - mu) / sigma) ** 2) + offset
    
    

def voigt_model(x, amp, mean, sigma, gamma, offset):
    # voigt_profile(x - mean, sigma, gamma) è normalizzata a area = 1
    return amp * voigt_profile(x - mean, sigma, gamma) + offset



R_instr = 850

def instr_fwhm(lambda0, R):
    return lambda0 / R
    
    
    
def estimate_fwhm(wavelength, flux, peak_index):
    
    fwhm_instr = instr_fwhm(wavelength[peak_index], R_instr)
    
    peak_flux = flux[peak_index]
    half_max = peak_flux + (np.nanmax(flux) - peak_flux) / 2
    
    left = peak_index
    while left > 0 and flux[left] < half_max:
        left -= 1
        
    right = peak_index
    while right < len(flux) - 1 and flux[right] < half_max:
        right += 1
            
    if left == right:
        return fwhm_instr
    
    fwhm_est = wavelength[right] - wavelength[left]
    
    return max(fwhm_est, fwhm_instr)



def plot_gauss_fit(w, f, cont, popt, name):
    plt.figure(figsize=(6,4))
    plt.plot(w, f, "k", label="Data")
    plt.plot(w, cont, "g--", label="Continuum")
    plt.plot(w, gaussian_model(w, *popt), "r", label="Gaussian fit")
    plt.legend()
    plt.xlabel("Wavelength")
    plt.title(name)
    plt.show()



def plot_voigt_fit(w, f, cont, popt, name):
    plt.figure(figsize=(6,4))
    plt.plot(w, f, "k", label="Data")
    plt.plot(w, cont, "g--", label="Continuum")
    plt.plot(w, voigt_model(w, *popt), "r", label="Voigt fit")
    plt.legend()
    plt.xlabel("Wavelength")
    plt.title(name)
    plt.show()



def estimate_noise(flux, cont_mask, snr_cap=300):
    # Noise estimate from continuum regions
    # 1.4826 rescales MAD to gaussian sigma
    sigma = 1.4826 * np.median(
        np.abs(flux[cont_mask] - np.median(flux[cont_mask]))
    )
    
    # Cap unrealistcally small noise at high SNR
    return max(sigma, np.median(flux) / snr_cap)



def plot_with_continuum(wavelength, flux, continuum):
    plt.figure(figsize=(12,5))

    plt.plot(wavelength, flux, color="black", lw=0.8)
    plt.plot(wavelength, continuum, color="red", lw=2)

    plt.xlabel("Wavelength")
    plt.ylabel("Flux")
    #plt.title(spectrum_name)
    plt.xlim(3650,6750)
    plt.ylim(0.5,1.05)

    plt.grid(alpha=0.3)
    plt.show()
    
    
    
def estimate_continuum(wavelength, flux, lam = 5000, smooth_win_size = 5, env_win_size = 10, percentile = 96):
    smooth_width = smooth_win_size * int((lam / R_instr) / np.median(np.diff(wavelength)))
    smooth_width = smooth_width + 1 if smooth_width % 2 == 0 else smooth_width

    flux_smooth_env = savgol_filter(flux, smooth_width, polyorder=2)
    
    #plot_with_continuum(wavelength, flux, flux_smooth_env)

    env_window = int(env_win_size * (lam / R_instr) / np.median(np.diff(wavelength)))

    continuum_env = percentile_filter(
        flux_smooth_env,
        percentile=percentile,
        size=env_window,
        mode="nearest"
    )
    
    #plot_with_continuum(wavelength, flux, continuum_env)

    continuum_env = savgol_filter(
        continuum_env,
        2 * smooth_width + 1,
        polyorder=2
    )
    
    #plot_with_continuum(wavelength, flux, continuum_env)
    
    return continuum_env



def fit_line_gaussian(wavelength, flux, continuum, peak_idx, mask, fwhm_use, plot=False, name=""):
    lam_detected = wavelength[peak_idx]
    w = wavelength[mask]
    f = flux[mask]
    c = continuum[mask]
    amp0 = f.max() - flux[peak_idx]
    offset = np.median(c)
    #p0 = [-amp0, lam_detected, fwhm_use / 2.355, median_continuum]
    p0 = [-amp0, lam_detected, fwhm_use / 2.355]
    #if plot:
    #    print("initial guess", p0)
    try:
        # popt, _ = curve_fit(gaussian_model, w, f, p0=p0)
        popt, _ = curve_fit(
            lambda x, a, m, s: gaussian_model(x, a, m, s, offset),
            w,
            f,
            p0 = p0)
        #amp, mu, sigma, offset = popt
        amp, mu, sigma = popt
        if plot:
            print("amp =",amp,"\nmu =",mu,"\nsigma =",sigma,"\noffset =",offset)
        sigma_nom = abs(sigma)
        fwhm_fit = 2.355 * sigma_nom
        fwhm_nom = fwhm_fit #np.clip(fwhm_fit, 0.7 * fwhm_instr, 1.5 * fwhm_instr)
        ew_nom = np.sqrt(2 * np.pi) * abs(amp) * sigma_nom / offset
        if plot:
            plot_gauss_fit(w, f, c, (amp, mu, sigma, offset), name)
    except RuntimeError:
        fwhm_nom = np.nan
        ew_nom = np.nan
        mu = np.nan

    return (mu, ew_nom, fwhm_nom)
 
 
 
def fit_line_voigt(wavelength, flux, continuum, peak_idx, mask, fwhm_use, plot=False, name=""):
    lam_detected = wavelength[peak_idx]
    w = wavelength[mask]
    f = flux[mask]
    c = continuum[mask]
    amp0 = f.max() - flux[peak_idx]
    offset = np.median(c)
    p0 = [-amp0, lam_detected, fwhm_use / 2.355, fwhm_use / 2.355]
    #p0 = [-amp0, lam_detected, fwhm_use / 2.355, fwhm_use / 2.355, offset]
    #if plot:
    #    print("initial guess", p0)
    try:
        #popt, _ = curve_fit(voigt_model, w, f, p0=p0)
        popt, _ = curve_fit(
            lambda x, a, m, s, g: voigt_model(x, a, m, s, g, offset),
            w,
            f,
            p0 = p0)
        #amp, mu, sigma, gam, offset = popt
        amp, mu, sigma, gam = popt
        if plot:
            print("amp =",amp,"\nmu =",mu,"\nsigma =",sigma,"\ngamma =",gam,"\noffset =",offset)
        #sigma_nom = abs(sigma)
        
        f_g = 2 * sigma * np.sqrt(2 * np.log(2))
        f_l = 2 * gam
        fwhm_nom = 0.5346 * f_l + np.sqrt(0.2166 * f_l**2 + f_g**2)
        
        ew_nom = abs(amp)/offset
        
        if plot:
            plot_voigt_fit(w, f, c, (amp, mu, sigma, gam, offset), name)
    except RuntimeError:
        fwhm_nom = np.nan
        ew_nom = np.nan
        mu = np.nan

    return (mu, ew_nom, fwhm_nom)
 

n_mc = 80 # number of Monte Carlo realizations 
 
def mc_line_errors(wavelength, flux, peak_idx, mask, fwhm_use, model):
    # noise estimate
    noise_cont_mask = (wavelength > 5500) & (wavelength < 5600)
    # noise = estimate_noise(f, cont_mask)
    noise = estimate_noise(flux, noise_cont_mask)
    
    # Monte Carlo error propagation
    ew_mc = []
    fwhm_mc = []
    
    for _ in range(n_mc):
        #slope_perturb = np.random.normal(0, 0.02) # 2% tilt
        #continuum_mc = continuum * (1 + slope_perturb * (w - w.mean()) / w.ptp())
        flux_mc = (flux + np.random.normal(0, noise, size=len(flux)))
        continuum_mc = estimate_continuum(wavelength, flux_mc)
        
        if model == "V":
            mu_i, ew_i, fwhm_i = fit_line_voigt(wavelength, flux_mc, continuum_mc, peak_idx, mask, fwhm_use)
        else:
            mu_i, ew_i, fwhm_i = fit_line_gaussian(wavelength, flux_mc, continuum_mc, peak_idx, mask, fwhm_use)
        
        if not(isnan(mu_i)):
            fwhm_mc.append(fwhm_i)
            ew_mc.append(ew_i)
        
    ew_mc = np.array(ew_mc)
    fwhm_mc = np.array(fwhm_mc)

    return (np.nanstd(ew_mc), np.nanstd(fwhm_mc))
    
    
    
def line_region_overlapping(wavelength, flux, lam, left, right, other_lam, win_size):
    other_peak_idx = np.argmin(np.abs(wavelength - other_lam))
    other_fwhm = estimate_fwhm(wavelength, flux, other_peak_idx)
    other_left = other_lam - win_size * other_fwhm
    other_right = other_lam + win_size * other_fwhm
    if other_right < left or other_left > right:
        # no overlap
        return (left, right)
    elif left < other_right < lam:
        return ((left+other_right)/2.0, right)
    elif lam < other_left < right:
        return (left,(right+other_left)/2.0)
    else:
        if lam < other_lam:
            return (left, (lam+other_lam)/2.0)
        else:
            return ((lam+other_lam)/2.0, right)
 
 
 
def fit_identified_line(wavelength, flux, continuum_env, identified_lines, name, lam_detected, win_size, model, plot=True):
    # find closest detected peak index
    peak_idx = np.argmin(np.abs(wavelength - lam_detected))
    
    fwhm_use = estimate_fwhm(wavelength, flux, peak_idx)
    
    #print("fwhm_use = ", fwhm_use)
    
    # Extraction window
    
    win = win_size * fwhm_use
    left = lam_detected - win
    right = lam_detected + win
    for other_name, _, _, other_lam_detected, _ in identified_lines:
        if name != other_name:
            left, right = line_region_overlapping(wavelength, flux, lam_detected, left, right, other_lam_detected, win_size)
    mask = (wavelength > left) & (wavelength < right)
    
    w = wavelength[mask]
    f = flux[mask]
    
    #if len(w) < 12:
    #    continue

    continuum = continuum_env[mask]
    
    if model == "V":
        mu, ew_nom, fwhm_nom = fit_line_voigt(wavelength, flux, continuum_env, peak_idx, mask, fwhm_use, plot, name)
    else:
        mu, ew_nom, fwhm_nom = fit_line_gaussian(wavelength, flux, continuum_env, peak_idx, mask, fwhm_use, plot, name)
    
    delta_ew, delta_fwhm = mc_line_errors(wavelength, flux, peak_idx, mask, fwhm_use, model)
    
    return (mu, ew_nom, delta_ew, fwhm_nom, delta_fwhm)
    
    
    
def fit_all_identified_lines(wavelength, flux, continuum_env, identified_lines):
    results = []

    for name, model, lam_expected, lam_detected, _ in identified_lines:
        win_size = 2
        
        mu, ew_nom, delta_ew, fwhm_nom, delta_fwhm = fit_identified_line(wavelength, flux, continuum_env, identified_lines, name, lam_detected, win_size, model)
        
        results.append({
            "Line": name,
            "Lambda min": lam_detected,
            "Lambda fit": mu,
            "EW": ew_nom,
            "EW_Err": delta_ew,
            "FWHM": fwhm_nom,
            "FWHM_Err": delta_fwhm
        })
    
    return results