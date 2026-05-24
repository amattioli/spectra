import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from astropy.io import fits


line_labels_colors = {
        "A": "green",
        "B": "darkorange",
        "C": "red"
    }


class Spectral_Line:
    
    def __init__(self, name, wavelength, cl):
        self.name = name
        self.wavelength = wavelength
        self.ident_class = cl
        
        
        
class Spectral_Lines_Set:
    
    def __init__(self):
        self.lines = []
        
    def add_line(self, line):
        self.lines.append(line)

    def load_csv(self, file_name, cols):
        spectral_lines = pd.read_csv(file_name, usecols = cols, names=["name","class","wavelength"])
        for index, row in spectral_lines.iterrows():
            name = row["name"]
            lin_ident_class = row["class"]
            wl = row["wavelength"]
            self.add_line(Spectral_Line(name, wl, lin_ident_class))


class Spectrum:
    
    def __init__(self):
        line_sets = {}
    
    def load(self, file_name):
        hdul = fits.open(file_name)
        header = hdul[0].header
        flux = hdul[0].data

        crval = header.get("CRVAL1") # starting wavelength
        cdelt = header.get("CDELT1") # wavelength increment
        crpix = header.get("CRPIX1", 1)
        self.title = header.get("OBJECT")

        npix = flux.size
        pixel = np.arange(npix)

        self.wavelength = crval + (pixel + 1 - crpix) * cdelt
        
        flux64 = np.asarray(flux, dtype=np.float64)    # convert to 64bit float
        valid = np.isfinite(flux64)
        median_value = np.nanmedian(flux64)
        clean_flux = np.where(valid, flux64, median_value)   # use the median where the flux is not finite
        
        self.flux = clean_flux
        
        
        
    def simple_plot(self, wl_range = (3700,6800), flux_range=(0.0,1.1)):
        plt.figure(figsize=(12,6))
        plt.plot(self.wavelength, self.flux, color="blue", lw=0.5)

        wlmin, wlmax = wl_range
        ymin, ymax = flux_range
        plt.ylim(ymin, ymax)
        plt.xlim(wlmin,wlmax)

        plt.xlabel("Wavelength (Angstrom)")
        plt.ylabel("Flux")
        plt.title(self.title)

        plt.grid(alpha=0.3)
        #plt.tight_layout()
        plt.show()

        
        
    def evaluate_snr(self, wl_range, window_length = 51, polyorder = 3):
        
        wl_min, wl_max = wl_range
        
        # 1. Filtra lo spettro per ottenere solo il "segnale pulito"
        # window_length deve essere superiore alla FWHM delle tue righe (es. 21 o 31)
        clean_signal = savgol_filter(flux, window_length=51, polyorder=3)

        # 2. Calcola i residui (questo è il tuo vero rumore)
        noise_array = flux - clean_signal

        # 3. Seleziona una zona senza righe di assorbimento forti
        # Esempio: zona tra 5400 e 5600 Å
        mask = (self.wavelength > wl_min) & (self.wavelength < wl_max)

        signal_level = np.median(self.flux[mask])
        noise_level = np.std(noise_array[mask])

        final_snr = signal_level / noise_level

        return final_snr
    
    
    
    def plot_with_lines(self, spectral_lines, wl_range, flux_range=(0.45,1.2)):
        wlmin, wlmax = wl_range
        ymin, ymax = flux_range
        plt.figure(figsize=(12,6.5))
        plt.plot(self.wavelength, self.flux, color="blue", lw=0.5)

        fmin, fmax = np.min(self.flux), np.max(self.flux)

        plt.ylim(ymin, ymax)
        plt.xlim(wlmin,wlmax)

        for line in spectral_lines.lines:
            name = line.name
            lin_ident_class = line.ident_class
            lam_obs = line.wavelength

            if (wlmin < lam_obs < wlmax) & (isinstance(name, str)):
                #print(name, lam_rest, lin_ident_class)
                lin_col = line_labels_colors[lin_ident_class if isinstance(lin_ident_class, str) else "A"]
                plt.axvline(lam_obs, color=lin_col, alpha=0.4, lw=1, ls="--", ymax=0.73)
                plt.text(
                    lam_obs,
                    ymin + (ymax-ymin) * 0.75,
                    name.replace(r'\n', '\n'),
                    rotation=90,
                    verticalalignment="bottom",
                    horizontalalignment="center",
                    fontsize=9,
                    color=lin_col
                )

        plt.xlabel("Wavelength (Angstrom)")
        plt.title(self.title)

        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()