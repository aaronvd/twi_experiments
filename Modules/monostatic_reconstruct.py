import numpy as np
import matplotlib.pyplot as plt
from pynufft import NUFFT
import scipy.interpolate
import scipy.constants
import types
import plotly.graph_objects as go
import plotly.io as pio
pio.renderers.default = "browser"
from scipy.ndimage import zoom
from tqdm import tqdm
from matplotlib.widgets import Slider

mm = 1E-3
C = scipy.constants.c

class MonostaticReconstruction():
    '''
    Class for reconstructing simulated or experimental monostatic imaging data.
    '''
    def __init__(self, f, Xa, Ya, z_offset):
        self.f = f
        self.lam = C/self.f
        self.k = 2*np.pi/self.lam
        self.array = types.SimpleNamespace()
        self.array.X = Xa
        self.array.Y = Ya
        self.indx_center_x = self.array.X.shape[1]//2
        self.indx_center_y = self.array.Y.shape[0]//2
        self.z_offset = z_offset

        xa = np.unique(Xa)
        ya = np.unique(Ya)
        self.array.delta_x = xa[1] - xa[0]
        self.array.delta_y = ya[1] - ya[0]
        self.array.Lx = np.amax(xa) - np.amin(xa)
        self.array.Ly = np.amax(ya) - np.amin(ya)
        self.volume = types.SimpleNamespace()

    def setup(self, scene_lengths, scene_deltas, Lxa, Lya, fc, bw, scene_offsets=None, delta_f_indx=1, method='RMA-NUFFT', pad_amount=100):

        self.method = method
        if scene_offsets is None:
            self.scene_offsets = (0, 0, self.z_offset)

        if self.method == 'RMA-NUFFT':
            self.volume.Lx, self.volume.Ly, self.volume.Lz = scene_lengths
            self.volume.delta_x, self.volume.delta_y, self.volume.delta_z = scene_deltas
            self.volume.x_offset, self.volume.y_offset, self.volume.z_offset = scene_offsets
            self.f_indx = np.argwhere(np.abs(self.f - fc)<=bw/2)[:,0]
            self.f_indx = np.arange(self.f_indx[0], self.f_indx[-1], delta_f_indx)
            f = self.f[self.f_indx]
            lam = C/f
            k = 2*np.pi/lam

            self.array.Lxa_trunc = Lxa
            self.array.Lya_trunc = Lya
            
            self.Nkx, self.Nky, self.Nkz = (int(4*np.amax(k) * self.volume.Lx / (2*np.pi)), 
                             int(4*np.amax(k) * self.volume.Ly / (2*np.pi)), 
                             int(4*np.amax(k) * self.volume.Lz / (2*np.pi)))
            self.Nx, self.Ny, self.Nz = (int(2*np.pi * self.Nkx / (4*np.amax(k) * self.volume.delta_x)), 
                          int(2*np.pi * self.Nky / (4*np.amax(k) * self.volume.delta_y)), 
                          int(2*np.pi * self.Nkz / (4*np.amax(k) * self.volume.delta_z)))
            if self.Nz == 0:
                self.Nz += 1
            if self.Nkz == 0:
                self.Nkz += 1

            self.volume.x = np.arange(-np.floor(self.Nx/2), np.ceil(self.Nx/2)) * self.volume.delta_x + self.volume.x_offset
            self.volume.y = np.arange(-np.floor(self.Ny/2), np.ceil(self.Ny/2)) * self.volume.delta_y + self.volume.y_offset
            self.volume.z = np.arange(-np.floor(self.Nz/2), np.ceil(self.Nz/2)) * self.volume.delta_z + self.volume.z_offset
            self.volume.X, self.volume.Y, self.volume.Z = np.meshgrid(self.volume.x, self.volume.y, self.volume.z, indexing='ij')

            dummy_array = np.ones((self.array.X.shape[0], self.array.X.shape[1]))
            dummy_array = dummy_array[int(self.indx_center_y-self.array.Lya_trunc//(2*self.array.delta_y)):int((self.indx_center_y+self.array.Lya_trunc//(2*self.array.delta_y))),
                            int(self.indx_center_x-self.array.Lxa_trunc//(2*self.array.delta_x)):int((self.indx_center_x+self.array.Lxa_trunc//(2*self.array.delta_x)))]

            self.pad_amount = pad_amount
            dummy_pad = np.pad(dummy_array, ((self.pad_amount, self.pad_amount), (self.pad_amount, self.pad_amount)), mode='constant')
            Ny_pad, Nx_pad = dummy_pad.shape
            delta_kx = 2*np.pi/(self.array.delta_x * Nx_pad)
            delta_ky = 2*np.pi/(self.array.delta_y * Ny_pad)
            kx = np.arange(-np.floor(Nx_pad/2), np.ceil(Nx_pad/2)) * delta_kx
            ky = np.arange(-np.floor(Ny_pad/2), np.ceil(Ny_pad/2)) * delta_ky
            self.Kx, self.Ky, K = np.meshgrid(kx.astype(np.complex64), ky.astype(np.complex64), k.astype(np.complex64), indexing='xy')
            self.Kz = np.sqrt(4*K**2 - self.Kx**2 - self.Ky**2)
            self.evanescent_indx = np.where(np.imag(self.Kz)!=0)
            self.Kz[self.evanescent_indx] = 0
            self.filt = np.nan_to_num(K / self.Kz**2)

            self.K_array = np.stack((np.real(self.Kx).flatten(), np.real(self.Ky).flatten(), np.real(self.Kz).flatten()), axis=1) * np.pi/(2*np.amax(k))   # normalizing to pi

            self.NufftObj = NUFFT()
            self.Jd = (3, 3, 3)
            self.NufftObj.plan(self.K_array, (self.Nx, self.Ny, self.Nz), (self.Nkx, self.Nky, self.Nkz), self.Jd)
        
        if self.method == 'RMA-interp':
            self.volume.Lx, self.volume.Ly, self.volume.Lz = scene_lengths
            self.volume.delta_x, self.volume.delta_y, self.volume.delta_z = scene_deltas
            self.volume.x_offset, self.volume.y_offset, self.volume.z_offset = scene_offsets
            self.f_indx = np.argwhere(np.abs(self.f - fc)<=bw/2)[:,0]
            self.f_indx = np.arange(self.f_indx[0], self.f_indx[-1], delta_f_indx)
            f = self.f[self.f_indx]
            lam = C/f
            k = 2*np.pi/lam

            self.array.Lxa_trunc = Lxa
            self.array.Lya_trunc = Lya

            self.pad_x = int(self.volume.Lx // self.array.delta_x)
            self.pad_y = int(self.volume.Ly // self.array.delta_y)

            dummy_array = np.ones((self.array.X.shape[0], self.array.X.shape[1], self.f.size))
            dummy_array = dummy_array[int(self.indx_center_y-self.array.Lya_trunc//(2*self.array.delta_y)):int((self.indx_center_y+self.array.Lya_trunc//(2*self.array.delta_y))),
                            int(self.indx_center_x-self.array.Lxa_trunc//(2*self.array.delta_x)):int((self.indx_center_x+self.array.Lxa_trunc//(2*self.array.delta_x))),
                            self.f_indx]

            dummy_pad = self.padortruncate(dummy_array, self.pad_y, self.pad_x, dummy_array.shape[2])
            Ny_pad, Nx_pad = dummy_pad.shape[:-1]
            delta_kx = 2*np.pi/(self.array.delta_x * Nx_pad)
            delta_ky = 2*np.pi/(self.array.delta_y * Ny_pad)
            kx = np.arange(-np.floor(Nx_pad/2), np.ceil(Nx_pad/2)) * delta_kx
            ky = np.arange(-np.floor(Ny_pad/2), np.ceil(Ny_pad/2)) * delta_ky
            self.Kx, self.Ky, K = np.meshgrid(kx.astype(np.complex64), ky.astype(np.complex64), k.astype(np.complex64), indexing='xy')
            self.Kz = np.sqrt(4*K**2 - self.Kx**2 - self.Ky**2)
            self.evanescent_indx = np.where(np.imag(self.Kz)!=0)
            self.Kz[self.evanescent_indx] = 0
            self.Kz = np.real(self.Kz)
            self.filt = np.nan_to_num(K / self.Kz**2)

    def reconstruct(self, measurements=None):
        
        if self.method == 'RMA-NUFFT':
            measurements = measurements[int(self.indx_center_y-self.array.Lya_trunc//(2*self.array.delta_y)):int((self.indx_center_y+self.array.Lya_trunc//(2*self.array.delta_y))),
                            int(self.indx_center_x-self.array.Lxa_trunc//(2*self.array.delta_x)):int((self.indx_center_x+self.array.Lxa_trunc//(2*self.array.delta_x))),
                            self.f_indx]
            self.measurement_truncated = measurements

            measurements_pad = np.pad(measurements, ((self.pad_amount, self.pad_amount), (self.pad_amount, self.pad_amount), (0, 0)), mode='constant')
            
            measurements_ft = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(
                        measurements_pad,
                        axes=(0,1)), axes=(0,1)), axes=(0,1))
            
            measurements_ft[self.evanescent_indx] = 0
            measurements_ft = measurements_ft * np.exp(1j * self.Kx * self.volume.x_offset) * np.exp(1j * self.Ky * self.volume.y_offset) * np.exp(1j * self.Kz * self.volume.z_offset) / self.filt
            measurements_reshape = measurements_ft.flatten()

            self.image = self.NufftObj.adjoint(measurements_reshape)

        if self.method == 'RMA-interp':
            measurements = measurements[int(self.indx_center_y-self.array.Lya_trunc//(2*self.array.delta_y)):int((self.indx_center_y+self.array.Lya_trunc//(2*self.array.delta_y))),
                            int(self.indx_center_x-self.array.Lxa_trunc//(2*self.array.delta_x)):int((self.indx_center_x+self.array.Lxa_trunc//(2*self.array.delta_x))),
                            self.f_indx]
            self.measurement_truncated = measurements            

            measurements_pad = self.padortruncate(measurements, self.pad_y, self.pad_x, measurements.shape[2])
            measurements_ft = np.fft.fftshift(np.fft.fftn(np.fft.ifftshift(
                        measurements_pad,
                        axes=(0,1)), axes=(0,1)), axes=(0,1))
            measurements_ft[self.evanescent_indx] = 0
            measurements_ft = measurements_ft * np.exp(1j * self.Kx * self.volume.x_offset) * np.exp(1j * self.Ky * self.volume.y_offset) * np.exp(1j * self.Kz * self.volume.z_offset) / self.filt

            delta_kz = 2*np.pi/self.volume.Lz
            kz_linear = np.arange(np.amin(self.Kz), np.amax(self.Kz), delta_kz)

            measurements_stolt = np.zeros((measurements_ft.shape[0], measurements_ft.shape[1], kz_linear.size), dtype=np.complex128)
            for i in range(measurements_ft.shape[0]):
                for j in range(measurements_ft.shape[1]):
                    if (self.Kz[i,j,self.Kz[i,j,:]!=0]).size >= 2:
                        measurements_stolt[i,j,:] = self.interp1d(kz_linear, self.Kz[i,j,self.Kz[i,j,:]!=0], measurements_ft[i,j,self.Kz[i,j]!=0], kind='cubic')
            measurements_stolt = np.transpose(np.nan_to_num(measurements_stolt), (1, 0, 2))

            self.image = np.fft.fftshift(np.fft.ifftn(np.fft.ifftshift(measurements_stolt)))
            self.volume.x = np.linspace(-self.volume.Lx/2, self.volume.Lx/2, measurements_stolt.shape[0]) + self.volume.x_offset
            self.volume.y = np.linspace(-self.volume.Ly/2, self.volume.Ly/2, measurements_stolt.shape[1]) + self.volume.y_offset
            self.volume.z = np.linspace(-self.volume.Lz/2, self.volume.Lz/2, measurements_stolt.shape[2]) + self.volume.z_offset
            self.volume.X, self.volume.Y, self.volume.Z = np.meshgrid(self.volume.x, self.volume.y, self.volume.z, indexing='ij')

            delta_x = self.volume.x[1] - self.volume.x[0]
            self.resample(delta_x / self.volume.delta_x)

    def resample(self, scale):
        # self.image = self.fft_resample(self.image, int(scale*self.image.shape[0]), int(scale*self.image.shape[1]), int(scale*self.image.shape[2]))
        self.image = zoom(self.image, (scale, scale, scale))
        self.volume.x = np.linspace(np.amin(self.volume.x), np.amax(self.volume.x), self.image.shape[0])
        self.volume.y = np.linspace(np.amin(self.volume.y), np.amax(self.volume.y), self.image.shape[1])
        self.volume.z = np.linspace(np.amin(self.volume.z), np.amax(self.volume.z), self.image.shape[2])
        self.volume.X, self.volume.Y, self.volume.Z = np.meshgrid(self.volume.x, self.volume.y, self.volume.z, indexing='ij')

    def plot(self, plot_type='3D', **kwargs):

        scale = kwargs.get('scale', 'dB')

        if scale == 'linear':
            colormin = kwargs.get('colormin', 0)
            colormax = kwargs.get('colormax', 1)
        else:
            colormin = kwargs.get('colormin', -10)
            colormax = kwargs.get('colormax', 0)
            
        if plot_type == '3D':
            cmap = kwargs.get('cmap', 'Turbo')

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
            cmap = kwargs.get('cmap', 'turbo')
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
            cmap = kwargs.get('cmap', 'turbo')
            
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

    def focus_sweep(self, measurements, L_list, a_horn=None, resample=1):
        self.L_list = L_list
        if a_horn is None:
            a_horn = 10.668*mm      # WR42 waveguide width

        lam = C/self.f
        k = 2*np.pi/lam
        beta_g = np.sqrt(k**2 - (np.pi/a_horn)**2)

        self.corrected_images = []
        for i in tqdm(range(L_list.size)):
            phase_error = np.exp(-1j*2*beta_g*self.L_list[i])    # 2 for transmit and receive path
            measurements_corrected = measurements / phase_error
            self.reconstruct(measurements_corrected)
            self.resample(resample)
            im_plot = np.transpose(np.mean(np.abs(self.image)**2, 2), (1,0))
            self.corrected_images.append(im_plot/np.amax(im_plot))

        self.set_font(fontsize=12)

        ### plot
        fig, ax = plt.subplots()
        ax_slider = plt.axes([0.20, 0.01, 0.65, 0.03], facecolor='yellow')
        slider = Slider(
                ax=ax_slider,
                label='Image Index',
                valmin=0,
                valmax=len(self.corrected_images)-1,
                valinit=0,
            )
        image = self.corrected_images[0]
        img = ax.imshow(image, origin='lower', cmap='turbo')
        ax.set_title('L = {}'.format(L_list[int(slider.val)]))
        def update(val):
            ax.imshow(self.corrected_images[int(slider.val)], origin='lower', cmap='turbo')
            ax.set_title('L = {}'.format(L_list[int(slider.val)]))
            fig.canvas.draw_idle()
        slider.on_changed(update)
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

    @staticmethod
    def interp1d(x_new, x, y, kind='linear'):
        f_real = scipy.interpolate.interp1d(x, np.real(y), kind=kind, bounds_error=False, fill_value=0)
        f_imag = scipy.interpolate.interp1d(x, np.imag(y), kind=kind, bounds_error=False, fill_value=0)
        f_interp = f_real(x_new) + 1j*f_imag(x_new)
        return f_interp










