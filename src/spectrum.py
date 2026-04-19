import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from scipy.signal import find_peaks
from scipy.signal import savgol_filter
from scipy.ndimage import median_filter


def load(file_name):
    hdul = fits.open(file_name)
    header = hdul[0].header
    flux = hdul[0].data
    
    crval = header.get("CRVAL1") # starting wavelength
    cdelt = header.get("CDELT1") # wavelength increment
    crpix = header.get("CRPIX1", 1)

    npix = flux.size
    pixel = np.arange(npix)

    wavelength = crval + (pixel + 1 - crpix) * cdelt
    
    return wavelength, flux
    
    
    
def cleanup_flux(flux):
    flux64 = np.asarray(flux, dtype=np.float64)    # convert to 64bit float
    valid = np.isfinite(flux64)
    median_value = np.nanmedian(flux64)
    clean_flux = np.where(valid, flux64, median_value)   # use the median where the flux is not finite
    return clean_flux
    
    
    
def save_as(wavelength, flux, file_name):
    hdr = fits.Header()
    hdr["CRVAL1"] = wavelength[0]
    hdr["CDELT1"] = wavelength[1] - wavelength[0]
    hdr["CRPIX1"] = 1
    hdr["CTYPE1"] = "Wavelength"
    hdr["CUNIT1"] = "Angstrom"
    
    hdu = fits.PrimaryHDU(data=flux, header=hdr)
    hdu.writeto(file_name, overwrite=True)
    
    
    
def simple_plot(wavelength, flux, title):
    plt.figure(figsize=(12,5))
    plt.plot(wavelength, flux, color="black", lw=1)

    plt.ylim(0.4, 1.1)
    plt.xlim(3700,6800)

    plt.xlabel("Wavelength (Angstrom)")
    plt.ylabel("Flux")
    plt.title(title)

    plt.grid(alpha=0.3)
    #plt.tight_layout()
    plt.show()



def find_absorption_lines(wavelength, flux, smoothing_width, min_prominence):
    
    flux_smooth = median_filter(flux, size=smoothing_width)

    prominence = min_prominence * np.nanmax(flux_smooth)

    peaks, properties = find_peaks(-flux_smooth, prominence=prominence)

    peak_wavelengths = wavelength[peaks]
    peak_fluxes = flux_smooth[peaks]

    return peak_wavelengths, peak_fluxes
    
    
    
def identify_absorption_lines(peak_wavelengths, peak_fluxes, spectral_lines, z = 0.0, line_window = 2.0):
    identified_lines = []
    for name, lam_rest in spectral_lines.items():
        lam_obs = lam_rest * (1 + z)
        
        # find detected peaks close to this known line
        distance = np.abs(peak_wavelengths - lam_obs)
        match = np.where(distance < line_window)[0]
        
        if len(match) > 0:
            best = match[np.argmin(distance[match])]
            identified_lines.append(
                (name, lam_obs, peak_wavelengths[best], peak_fluxes[best])
            )
    
    return identified_lines
            


def calc_snr(wavelength, flux, wl_min, wl_max, window_length = 51, polyorder = 3, plot = False):
    # 1. Filtra lo spettro per ottenere solo il "segnale pulito"
    # window_length deve essere superiore alla FWHM delle tue righe (es. 21 o 31)
    clean_signal = savgol_filter(flux, window_length=51, polyorder=3)

    if plot:
        simple_plot(wavelength, flux, "Iota Her 06/06/2025")
        simple_plot(wavelength, clean_signal, "Iota Her 06/06/2025")

    # 2. Calcola i residui (questo è il tuo vero rumore)
    noise_array = flux - clean_signal


    if plot:
        plt.figure(figsize=(12,6))
        plt.plot(wavelength, noise_array, color="black", lw=1)

        plt.ylim(-0.1, 0.1)
        plt.xlim(3700,6800)

        plt.xlabel("Wavelength (Angstrom)")
        plt.ylabel("Noise")

        plt.grid(alpha=0.3)
        #plt.tight_layout()
        plt.show()



    # 3. Seleziona una zona senza righe di assorbimento forti
    # Esempio: zona tra 5400 e 5600 Å
    mask = (wavelength > wl_min) & (wavelength < wl_max)

    signal_level = np.median(flux[mask])
    noise_level = np.std(noise_array[mask])

    final_snr = signal_level / noise_level
    
    return final_snr