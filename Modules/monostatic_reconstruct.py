import numpy as np
import matplotlib.pyplot as plt
from pynufft import NUFFT
import scipy.constants
import types
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"
from scipy.ndimage import zoom

mm = 1E-3
C = scipy.constants.c

class MonostaticReconstruction():
    '''
    Class for reconstructing simulated or experimental monostatic imaging data.
    '''
    def __init__(self, f, Xa, Ya, z_offset, measurements):
        self.f = f
        self.lam = C/self.f
        self.k = 2*np.pi/self.lam
        self.array = types.SimpleNamespace()
        self.array.X = Xa
        self.array.Y = Ya
        self.z_offset = z_offset

        xa = np.unique(Xa)
        ya = np.unique(Ya)
        self.array.delta_x = xa[1] - xa[0]
        self.array.delta_y = ya[1] - ya[0]
        self.array.Lx = np.amax(xa) - np.amin(xa)
        self.array.Ly = np.amax(ya) - np.amin(ya)
        self.measurements = measurements
        self.volume = types.SimpleNamespace()

    def reconstruct(self, scene_lengths, scene_deltas, Lxa, Lya, fc, bw, scene_offsets=None, method='RMA', pad_amount=100):
        
        if scene_offsets is None:
            scene_offsets = (0, 0, self.z_offset)
        
        if method == 'RMA':
            ######################################
            ########    set up geometry   ########
            ######################################
            self.volume.Lx, self.volume.Ly, self.volume.Lz = scene_lengths
            self.volume.delta_x, self.volume.delta_y, self.volume.delta_z = scene_deltas
            self.volume.x_offset, self.volume.y_offset, self.volume.z_offset = scene_offsets
            f_indx = np.argwhere(np.abs(self.f - fc)<=bw/2)[:,0]
            f = self.f[f_indx]
            lam = C/f
            k = 2*np.pi/lam
            indx_center_x = self.array.X.shape[1]//2
            indx_center_y = self.array.Y.shape[0]//2
            measurements = self.measurements[int(indx_center_y-Lya//(2*self.array.delta_y)):int((indx_center_y+Lya//(2*self.array.delta_y))),
                            int(indx_center_x-Lxa//(2*self.array.delta_x)):int((indx_center_x+Lxa//(2*self.array.delta_x))),
                            f_indx]
            self.measurement_truncated = measurements
            self.f_indx = f_indx
            
            Nkx, Nky, Nkz = (int(4*np.amax(k) * self.volume.Lx / (2*np.pi)), 
                             int(4*np.amax(k) * self.volume.Ly / (2*np.pi)), 
                             int(4*np.amax(k) * self.volume.Lz / (2*np.pi)))
            Nx, Ny, Nz = (int(2*np.pi * Nkx / (4*np.amax(k) * self.volume.delta_x)), 
                          int(2*np.pi * Nky / (4*np.amax(k) * self.volume.delta_y)), 
                          int(2*np.pi * Nkz / (4*np.amax(k) * self.volume.delta_z)))
            if Nz == 0:
                Nz += 1
            if Nkz == 0:
                Nkz += 1

            self.volume.x = np.arange(-np.floor(Nx/2), np.ceil(Nx/2)) * self.volume.delta_x + self.volume.x_offset
            self.volume.y = np.arange(-np.floor(Ny/2), np.ceil(Ny/2)) * self.volume.delta_y + self.volume.y_offset
            self.volume.z = np.arange(-np.floor(Nz/2), np.ceil(Nz/2)) * self.volume.delta_z + self.volume.z_offset
            self.volume.X, self.volume.Y, self.volume.Z = np.meshgrid(self.volume.x, self.volume.y, self.volume.z, indexing='ij')

            ######################################
            ########      reconstruct     ########
            ######################################
            measurements_pad = np.pad(measurements, ((pad_amount, pad_amount), (pad_amount, pad_amount), (0, 0)), mode='constant')
            Ny_pad, Nx_pad = measurements_pad.shape[:-1]
            measurements_ft = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(
                        measurements_pad,
                        axes=(0,1)), axes=(0,1)), axes=(0,1))
            delta_kx = 2*np.pi/(self.array.delta_x * Nx_pad)
            delta_ky = 2*np.pi/(self.array.delta_y * Ny_pad)
            kx = np.arange(-np.floor(Nx_pad/2), np.ceil(Nx_pad/2)) * delta_kx
            ky = np.arange(-np.floor(Ny_pad/2), np.ceil(Ny_pad/2)) * delta_ky
            Kx, Ky, K = np.meshgrid(kx.astype(np.complex64), ky.astype(np.complex64), k.astype(np.complex64), indexing='xy')
            Kz = np.sqrt(4*K**2 - Kx**2 - Ky**2)
            Kz[np.imag(Kz)!=0] = 0
            measurements_ft[np.imag(Kz)!=0] = 0

            filt = np.nan_to_num(K / Kz**2)
            measurements_ft = measurements_ft * np.exp(1j * Kx * self.volume.x_offset) * np.exp(1j * Ky * self.volume.y_offset) * np.exp(1j * Kz * self.volume.z_offset) / filt

            NufftObj = NUFFT()
            Jd = (3, 3, 3)

            K_array = np.stack((np.real(Kx).flatten(), np.real(Ky).flatten(), np.real(Kz).flatten()), axis=1) * np.pi/(2*np.amax(k))   # normalizing to pi
            measurements_reshape = measurements_ft.flatten()

            NufftObj.plan(K_array, (Nx, Ny, Nz), (Nkx, Nky, Nkz), Jd)

            self.image = NufftObj.adjoint(measurements_reshape)

    def resample(self, scale):
        # self.image = self.fft_resample(self.image, int(scale*self.image.shape[0]), int(scale*self.image.shape[1]), int(scale*self.image.shape[2]))
        self.image = zoom(self.image, (scale, scale, scale))
        self.volume.x = np.linspace(np.amin(self.volume.x), np.amax(self.volume.x), self.image.shape[0])
        self.volume.y = np.linspace(np.amin(self.volume.y), np.amax(self.volume.y), self.image.shape[1])
        self.volume.z = np.linspace(np.amin(self.volume.z), np.amax(self.volume.z), self.image.shape[2])
        self.volume.X, self.volume.Y, self.volume.Z = np.meshgrid(self.volume.x, self.volume.y, self.volume.z, indexing='ij')

    def plot(self, plot_type='3D', **kwargs):

        cmap = kwargs.get('cmap', 'viridis')
        scale = kwargs.get('scale', 'dB')

        if scale == 'linear':
            colormin = kwargs.get('colormin', 0)
            colormax = kwargs.get('colormax', 1)
        else:
            colormin = kwargs.get('colormin', -10)
            colormax = kwargs.get('colormax', 0)
            
        if plot_type == '3D':

            im_plot = np.abs(self.image)**2
            im_plot = im_plot/np.amax(im_plot)
            if scale == 'dB':
                im_plot = 10*np.log10(im_plot)

            surface_count = kwargs.get('surface_count', 20)
            opacity = kwargs.get('opacity', 0.1)

            fig = go.Figure(data=go.Volume(
                            x=self.volume.Z.flatten(),
                            y=self.volume.X.flatten(),
                            z=self.volume.Y.flatten(),
                            value=im_plot.flatten(),
                            name='Reconstruction',
                            isomin=colormin,
                            isomax=colormax,
                            opacity=opacity, # needs to be small to see through all surfaces
                            surface_count=surface_count, # needs to be a large number for good volume rendering
                            colorscale=cmap,
                            caps= dict(x_show=False, y_show=False, z_show=False),
                            opacityscale='uniform'
                        ))
            fig.show()
        
        if plot_type == 'section':
            slice_axis = kwargs.get('slice_axis', 'z')
            scale = kwargs.get('scale', 'dB')
            cmap = kwargs.get('cmap', 'viridis')
            if scale == 'linear':
                colormin = kwargs.get('colormin', 0)
                colormax = kwargs.get('colormax', 1)
            else:
                colormin = kwargs.get('colormin', -10)
                colormax = kwargs.get('colormax', 0)

            if slice_axis == 'x':
                slice_index = kwargs.get('slice_index', self.image.shape[0]//2)
                im_plot = np.transpose(np.abs(self.image[slice_index,:,:])**2, (1,0))
                x = self.volume.y
                y = self.volume.z
                xlabel = '$y$ (m)'
                ylabel = '$z$ (m)'
            elif slice_axis == 'y':
                slice_index = kwargs.get('slice_index', self.image.shape[1]//2)
                im_plot = np.transpose(np.abs(self.image[:,slice_index,:])**2, (1,0))
                x = self.volume.x
                y = self.volume.z
                xlabel = '$x$ (m)'
                ylabel = '$z$ (m)'
            elif slice_axis == 'z':
                slice_index = kwargs.get('slice_index', self.image.shape[2]//2)
                im_plot = np.transpose(np.abs(self.image[:,:,slice_index])**2, (1,0))
                x = self.volume.x
                y = self.volume.y
                xlabel = '$x$ (m)'
                ylabel = '$y$ (m)'

            im_plot = im_plot/np.amax(np.abs(self.image)**2)
            if scale == 'dB':
                im_plot = 10*np.log10(im_plot)

            self.set_font(fontsize=15)
            plt.figure()
            plt.imshow(im_plot,
                    extent=(np.amin(x), np.amax(x), np.amin(y), np.amax(y)),
                    origin='lower',
                    cmap=cmap)
            plt.clim(colormin, colormax)
            plt.xlabel(xlabel)
            plt.ylabel(ylabel)
            plt.title('Reconstruction')
            if scale == 'linear':
                plt.colorbar()
            else:
                plt.colorbar(label='dB')
            plt.show()     

        if plot_type == 'xy':
            im_plot = np.transpose(np.mean(np.abs(self.image)**2, 2), (1,0))
            im_plot = im_plot/np.amax(im_plot)
            if scale == 'dB':
                im_plot = 10*np.log10(im_plot)

            self.set_font(fontsize=15)
            plt.figure()
            plt.imshow(im_plot,
                    extent=(np.amin(self.volume.x), np.amax(self.volume.x), np.amin(self.volume.y), np.amax(self.volume.y)),
                    origin='lower',
                    cmap=cmap)
            plt.clim(colormin, colormax)
            plt.xlabel('$x$ (m)')
            plt.ylabel('$y$ (m)')
            plt.title('Reconstruction')
            if scale == 'linear':
                plt.colorbar()
            else:
                plt.colorbar(label='dB')
            plt.show()               

    def fft_resample(self, img, Nx, Ny, Nz):
        '''
        Resamples image img by padding or truncating in the Fourier domain.

        '''
        img = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(img)))
        scal = img.size
        img = self.padortruncate(img, Nx, Ny, Nz)
        scal = img.size / scal
        img = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(img)))*scal

        return img

    @staticmethod
    def padortruncate(array, dx, dy, dz, val_bg=0):
        '''
        Pads (with value val_bg) or truncates array depending on whether array dimensions are great than or less than (dx, dy)

        '''
        dx = int(dx)
        dy = int(dy)
        dz = int(dz)
        nx = max((dx-array.shape[0])//2, 0)
        ny = max((dy-array.shape[1])//2, 0)
        nz = max((dz-array.shape[2])//2, 0)
        px = max((array.shape[0]-dx)//2, 0)
        py = max((array.shape[1]-dy)//2, 0)
        pz = max((array.shape[2]-dz)//2, 0)
        newarray = val_bg * np.ones((dx, dy, dz), dtype=array.dtype)
        cx = min(array.shape[0], dx)
        cy = min(array.shape[1], dy)
        cz = min(array.shape[2], dz)
        newarray[nx:nx+cx, ny:ny+cy, nz:nz+cz] = array[px:px+cx, py:py+cy, pz:pz+cz]

        return newarray

    @staticmethod
    def set_font(fontsize=18, font="Times New Roman"):
        rc = {"font.size" : fontsize,
        "font.family" : font,
        "mathtext.fontset" : "stix"}
        plt.rcParams.update(rc)










