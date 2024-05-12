
## *** All parameters should be defined in millimeters ***

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks", {'grid.linestyle': '--'})
import gc
import datetime,time
import pandas as pa
from tqdm import tqdm
import sys
from scipy.signal import argrelmax
import json
from ..utils.utilities import set_params,ToJsonEncoder
import cupy as cp

import pydicom
from pydicom.uid import ImplicitVRLittleEndian
from pydicom.dataset import Dataset, FileDataset

__all__ = [
'TuringPattern',
]

class TuringPattern:
    def __init__(self):
        self.dtype = 'float32'
        self.save_dicom=False
        self.params ={
        'grid':[40, 40, 40],
        'dx':1 / 40,
        'dt':1,
        'du':0.0002,
        'dv':0.01,
        'length':13,
        'repetition':100,
        'bv_tv':0.138,
        'voxelsize':0.0295,
        'seed':False,
        'ct_coef':4.5e4,
        'tile_num_xz':1,
        'tile_num_y':1,
        }
        self.coef = np.array([-7.67681281,1.65199492,-0.45314158,0.60424417])
        self.keys_params  = list(self.params.keys())
        self._setatter_params()
        self.f = lambda u,v: u+self.dt*(0.6*u-v-u**3)
        self.g = lambda u,v: v+self.dt*(1.5*u-2*v)

    def modeling(self,path,save_dicom=False):
        self.save_dicom=save_dicom
        mempool = cp.get_default_memory_pool()
        pinned_mempool = cp.get_default_pinned_memory_pool()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        self._calc_kukv()
        u,v = self._get_inital_vector()
        for i in tqdm(range(self.repetition)):
            u,v = self._calc_onestep(u,v)
        self.model_shape=u.shape
        print("Model Size: %s Mb"%(sys.getsizeof(u)/1e6))
        U = cp.asnumpy(u)
        del self.ku, self.kv, u,v
        gc.collect()
        mempool.free_all_blocks()
        pinned_mempool.free_all_blocks()
        if save_dicom:
            self._save_dicom(U,path)
        U = self._adjust_vbtv(U)
        self._calc_microarchitecture(U)
        self._save_info(path)
        U = self._model_binarization(U)
        if self.tile_num_xz != 0:
            U = np.tile(U, (self.tile_num_xz,self.tile_num_y,self.tile_num_xz))
        return U

    def set_threshold_func_coef(self,coef):
        self.coef = np.array(coef)

    def set_params(self, params):
        set_params(self.params, **params)
        self._setatter_params()

    def set_inhibitor_function(self,g):
        self.g = g

    def set_activator_function(self,f):
        self.f = f

    def _setatter_params(self):
        for k in self.params:
            setattr(self,k,self.params[k])
        self.gridsize=[int(self.length * grid) for grid in self.grid]

    def _kernel_vector3(self,dd):
        ke = np.float32(-self.dt*dd/self.dx**2)
        kernel = np.zeros(tuple(self.gridsize)).astype(self.dtype)
        kernel[0,0,0] = np.float32(1+6*self.dt*dd/self.dx**2)
        kernel[1,0,0] = ke
        kernel[-1,0,0] = ke
        kernel[0,-1,0] = ke
        kernel[0,1,0] = ke
        kernel[0,0,-1] = ke
        kernel[0,0,1] = ke
        return kernel

    def _calc_kukv(self):
        self.ku = cp.asarray(self._kernel_vector3(self.du))
        self.kv = cp.asarray(self._kernel_vector3(self.dv))
        self.ku = 1/(cp.fft.fftn(self.ku).real.astype('float32'))
        self.kv = 1/(cp.fft.fftn(self.kv).real.astype('float32'))


    def _calc_onestep(self,u,v):
        a = cp.real(cp.fft.ifftn(self.ku*cp.real(cp.fft.fftn(self.f(u,v)))))
        b = cp.real(cp.fft.ifftn(self.kv*cp.real(cp.fft.fftn(self.g(u,v)))))
        return a,b


    def _get_inital_vector(self):
        gs = self.gridsize
        if self.seed:
            np.random.seed(seed=int(self.seed))
        ui = (1-np.random.rand(*gs).astype(self.dtype))
        if self.seed:
            np.random.seed(seed=int(self.seed*2))
        vi = (1-np.random.rand(*gs).astype(self.dtype))
        return cp.asarray(ui),cp.asarray(vi)

    def _get_threshold(self):
        x = self.bv_tv
        self.threshold = np.poly1d(self.coef)(x)

    def _adjust_vbtv(self,u):
        self._get_threshold()
        ind = np.where(u<=self.threshold)
        u[ind[0],ind[1],ind[2]] = 0
        self.bv_tv_real = 1-ind[0].shape[0]/(u.shape[0]*u.shape[1]*u.shape[2])
        print("BV/TV = %.4f"%self.bv_tv_real)
        return u

    def _calc_microarchitecture(self,u):
        def count_peak_par_axis(sub_pic):
            a = 0
            for i in sub_pic:
                a += argrelmax(i,order = 5)[0].shape[0]
            return a/sub_pic.shape[0]
        self.num_peak = 0
        for i in u:
            self.num_peak+=count_peak_par_axis(i)
        tbn = self.num_peak/(self.voxelsize*self.gridsize[1]*self.gridsize[2])

        self.microarchitecture = {
        'BV/TV':self.bv_tv,
        'BV/TV real':self.bv_tv_real,
        'Tb.N* [/mm]':tbn,
        'Tb.Th [mm]':(self.bv_tv_real/tbn),
        'Tb.Sp [mm]':((1-self.bv_tv_real)/tbn),
        'Dtorb [mg/cm3]':(self.bv_tv_real*1200),
        }

    def _model_binarization(self,u):
        ind = np.where(u>0)
        u[ind[0],ind[1],ind[2]] = 1
        return u.astype("int8")

    def _save_info(self,path,coment=""):
        info = {
            'Date':datetime.datetime.now().isoformat(),
            'coment':coment,
            'model_params':self.params,
            'microarchitecture':self.microarchitecture,
            'model_shape':self.model_shape,
        }
        save_name = path+"/_info.json"
        with open(save_name, 'w') as fp:
            json.dump(info,fp,indent=4,cls= ToJsonEncoder)

    def _save_dicom(self,pixel_array,path):
        pixel_array = (pixel_array.copy()*self.ct_coef).astype("uint16")
        for i in range(0, np.shape(pixel_array)[2]):
            self._write_dicom(pixel_array[:,:,i], i,path,self.voxelsize)
        del pixel_array
        gc.collect()

    def _write_dicom(self,pixel_array, level,path,voxelsize):
        suffix=  f'{level:04}' + ".dcm"

        filename_endian= path+"/" + suffix
        file_meta= Dataset()
        file_meta.TransferSyntaxUID= ImplicitVRLittleEndian
        file_meta.MediaStorageSOPClassUID = pydicom.uid.generate_uid()
        file_meta.MediaStorageSOPInstanceUID = pydicom.uid.generate_uid()
        file_meta.ImplementationClassUID = pydicom.uid.generate_uid()

        ds = FileDataset(filename_endian, {}, file_meta= file_meta, preamble= b"\0"*128)
        ds.Modality = 'WSD'
        ds.ContentDate = str(datetime.date.today()).replace('-','')
        ds.ContentTime = '010101.000000' #milliseconds since the epoch
        ds.StudyInstanceUID =  pydicom.uid.generate_uid()
        ds.SeriesInstanceUID = pydicom.uid.generate_uid()
        ds.SOPInstanceUID =    pydicom.uid.generate_uid()
        ds.SOPClassUID = pydicom.uid.generate_uid()
        #ds.SecondaryCaptureDeviceManufctur = 'Python 3.6.5'

        ## These are the necessary imaging components of the FileDataset object.
        ds.SamplesPerPixel = 1
        ds.PhotometricInterpretation = "MONOCHROME2"
        ds.PixelRepresentation = 0
        ds.HighBit = 15
        ds.BitsStored = 16
        ds.BitsAllocated = 16
        #ds.SmallestImagePixelValue = pixel_array.min()
        #ds[0x00280106].VR= 'US'
        ds.LargestImagePixelValue = pixel_array.max()
        ds[0x00280107].VR= 'US'
        ds.Columns = pixel_array.shape[0]
        ds.Rows = pixel_array.shape[1]
        ds.PixelSpacing = [voxelsize, voxelsize]
        """if pixel_array.dtype != np.uint16:
            pixel_array = pixel_array.astype(np.uint16)"""
        ds.PixelData = pixel_array.tobytes()

        ds.save_as(filename_endian)


    def linear_stability_analysis(self):
        dp = self.du
        dq = self.dv

        import sympy
        p = sympy.Symbol('p')
        q = sympy.Symbol('q')
        intialUV = sympy.solve([self.f(p,q),self.g(p,q)])

        print('initial value')
        print(intialUV)
        try:
            intialUV = intialUV[0]
        except:
            pass

        fu = sympy.diff(self.f(p,q),p).subs([(p, intialUV[p])])
        fv = sympy.diff(self.f(p,q),q).subs([(q, intialUV[q])])
        gu = sympy.diff(self.g(p,q),p).subs([(p, intialUV[p])])
        gv = sympy.diff(self.g(p,q),q).subs([(q, intialUV[q])])
        print()
        print("fu = %.3f"%fu)
        print("fv = %.3f"%fv)
        print("gu = %.3f"%gu)
        print("gv = %.3f"%gv)

        la = sympy.Symbol('la')
        k = sympy.Symbol('k')
        dispersion_relation = sympy.solve((la-fu+dp*k**2)*(la-gv+dq*k**2)-fv*gu,la)
        print()
        print("Solution of λ")
        print(dispersion_relation[0])
        print(dispersion_relation[1])

        wave_range =np.arange(100)
        growth_rate = []
        for i in wave_range:
            ans0 = dispersion_relation[0].subs([(k, i)])
            ans1 = dispersion_relation[1].subs([(k, i)])
            growth_rate.append(np.array([ans0,ans1]).astype(float))
        growth_rate = np.array(growth_rate).T

        fig, axis = plt.subplots(ncols=2,sharex=True,figsize=(10,4), dpi = 100)
        axis[0].plot(wave_range/(2*np.pi),growth_rate[0],c = 'k')
        axis[0].plot(np.array([wave_range[0],wave_range[-1]])/(2*np.pi),[0,0],'--',c = 'k')
        axis[0].set_title("Top : %.3f" %(wave_range[growth_rate[0].argmax()]/(2*np.pi)))
        axis[0].set_xlabel('Number of repetition per unit length')
        axis[1].plot(wave_range/(2*np.pi),growth_rate[1],c = 'k')
        axis[1].plot(np.array([wave_range[0],wave_range[-1]])/(2*np.pi),[0,0],'--',c = 'k')
        axis[1].set_title("Top : %.3f" %(wave_range[growth_rate[1].argmax()]/(2*np.pi)))
        axis[1].set_xlabel('Number of repetition per unit length')
        plt.show()
