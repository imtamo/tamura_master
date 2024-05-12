#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from ._kernel import vmc_kernel

import os
import gc
import cv2
import bz2
import nrrd
import tqdm
import time
import json
import pickle
import pydicom
import datetime
import warnings
import numpy as np
import pandas as pd
import pyvista as pv
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib import colors
from abc import ABCMeta, abstractmethod
from ..utils.utilities import set_params, ToJsonEncoder, calTime
from .. import visualization as viz
from ..preprocessing import Segment

warnings.filterwarnings("ignore", category=RuntimeWarning)
# os.system("taskset -p 0xff %d" % os.getpid())

__all__ = [
    'VoxelPlateModel', 'VoxelTuringModel'
]


# =============================================================================
# Base solid model
# =============================================================================

class BaseVoxelMonteCarlo(metaclass=ABCMeta):
    # @_deprecate_positional_args
    @abstractmethod
    def __init__(self, *, nPh, model, dtype_f=np.float32, dtype=np.int32,
                 beam_type='TEM00', w_beam=0,
                 beam_angle=0, initial_refrect_by_angle=False,
                 first_layer_clear=False,
                 ):

        def __check_list_name(name, name_list):
            if not (name in name_list):
                raise ValueError('%s is not a permitted for factor. Please choose from %s.' % (name, name_list))

        self.beam_type_list = ['TEM00', False]
        __check_list_name(beam_type, self.beam_type_list)
        self.beam_type = beam_type

        self.dtype = dtype
        self.dtype_f = dtype_f
        self.nPh = nPh
        self.w_beam = w_beam

        self.initial_refrect_by_angle = initial_refrect_by_angle
        self.beam_angle = beam_angle

        self.model = model
        self.first_layer_clear = first_layer_clear

    def start(self):
        self.nPh = int(self.nPh)
        self._reset_results()
        self._generate_initial_coodinate(self.nPh)

        self.add = self.add.astype(np.int32)
        self.p = self.p.astype(np.float32)
        self.v = self.v.astype(np.float32)
        self.w = self.w.astype(np.float32)
        print("")
        print("###### Start ######")
        print("")
        start_ = time.time()
        self.add, self.p, self.v, self.w = vmc_kernel(
            self.add, self.p, self.v, self.w,
            self.model.ma, self.model.ms, self.model.n, self.model.g,
            self.model.voxel_model, self.model.voxel_space,
            np.int32(self.nPh - self.inital_del_num), np.int8(self.model.end_point)
        )

        self._end_process()
        print("###### End ######")
        calTime(time.time(), start_)
        # del func
        return self

    def _end_process(self):
        # index = np.where(~np.isnan(self.w))[0]
        self.v_result = self.v  # [:,index]
        self.p_result = self.p  # [:,index]
        self.add_result = self.add  # [:,index]
        self.w_result = self.w  # [index]

    def _reset_results(self):
        self.v_result = np.empty((3, 1)).astype(self.dtype_f)
        self.p_result = np.empty((3, 1)).astype(self.dtype_f)
        self.add_result = np.empty((3, 1)).astype(self.dtype)
        self.w_result = np.empty(1).astype(self.dtype_f)
        return self

    def get_voxel_model(self):
        return self.model.voxel_model

    def _generate_initial_coodinate(self, nPh):
        self._set_inital_add()
        self._set_beam_distribution()
        self._set_inital_vector()
        self._set_inital_w()

    def _set_inital_add(self):
        if self.beam_type == 'TEM00':
            self.add = np.zeros((3, self.nPh), dtype=self.dtype)
        self.add[0] = self._get_center_add(self.model.voxel_model.shape[0])
        self.add[1] = self._get_center_add(self.model.voxel_model.shape[1])
        if self.first_layer_clear:
            self.add[2] = self.model.get_second_layer_addz()
        else:
            self.add[2] = 1

    def _get_center_add(self, length):
        # addの中心がローカル座標（ボックス内）の中心となるため、
        # ボクセル数が偶数の時は、1/2小さい位置を中心とし光を照射し、
        # 逆変換時（_encooder）も同様に1/2小さい位置を中心として元のマクロな座標に戻す。
        return int((length - 1) / 2)

    def _set_inital_vector(self):
        if self.beam_type == 'TEM00':
            self.v = np.zeros((3, self.nPh)).astype(self.dtype_f)
            self.v[2] = 1
            if self.beam_angle != 0 and self.w_beam == 0:
                # ビーム径がある場合はとりあえず無視
                # 角度はrad表記
                ni = self.model.n[-1]
                nt = self.model.n[0]
                ai = self.beam_angle
                at = np.arcsin(np.sin(ai) * ni / nt)
                self.v[0] = np.sin(at)
                self.v[2] = np.cos(at)
                if self.initial_refrect_by_angle:
                    Ra = ((np.sin(ai - at) / np.sin(ai + at)) ** 2 \
                          + (np.tan(ai - at) / np.tan(ai + at)) ** 2) / 2

                    self.inital_del_num = np.count_nonzero(Ra >= np.random.rand(self.nPh))
                    self.v = np.delete(self.v, np.arange(self.inital_del_num), 1)
                    self.p = np.delete(self.p, np.arange(self.inital_del_num), 1)
                    self.add = np.delete(self.add, np.arange(self.inital_del_num), 1)
                    sub_v = np.zeros((3, self.inital_del_num)).astype(self.dtype_f)
                    sub_v[0] = np.sin(ai)
                    sub_v[2] = -np.cos(ai)
                    self.v_result = np.concatenate([self.v_result,
                                                    sub_v], axis=1)
                    self.p_result = np.concatenate([self.p_result,
                                                    self.p[:, :self.inital_del_num]], axis=1)
                    self.add_result = np.concatenate([self.add_result,
                                                      self.add[:, :self.inital_del_num]], axis=1)
        else:
            print("ビームタイプが設定されていません")

    def _set_inital_w(self):
        if self.beam_type == 'TEM00':
            self.w = np.ones(self.nPh).astype(self.dtype_f)
            Rsp = 0
            n1 = self.model.n[-1]
            n2 = self.model.n[0]
            if n1 != n2:
                Rsp = ((n1 - n2) / (n1 + n2)) ** 2
                if self.beam_angle != 0 and self.w_beam == 0:
                    ai = self.beam_angle
                    at = np.arcsin(np.sin(ai) * n1 / n2)
                    Rsp = ((np.sin(ai - at) / np.sin(ai + at)) ** 2 \
                           + (np.tan(ai - at) / np.tan(ai + at)) ** 2) / 2
                elif self.first_layer_clear:
                    n3 = self.model.n[1]
                    r2 = ((n3 - n2) / (n3 + n2)) ** 2
                    Rsp = Rsp + r2 * (1 - Rsp) ** 2 / (1 - Rsp * r2)
                self.w -= Rsp

            if self.beam_angle != 0 and self.w_beam == 0:
                if self.initial_refrect_by_angle:
                    self.w[:] = 1
                    self.w = np.delete(self.w, np.arange(self.inital_del_num), 0)
                    self.w_result = np.concatenate([self.w_result,
                                                    self.w[:self.inital_del_num]], axis=0)
        else:
            print("ビームタイプが設定されていません")

    def _set_beam_distribution(self):
        if self.beam_type == 'TEM00':
            self.p = np.zeros((3, self.nPh)).astype(self.dtype_f)
            self.p[2] = -self.model.voxel_space / 2
            if self.w_beam != 0:
                print("%sを入力" % self.beam_type)
                # ガウシアン分布を生成
                gb = np.array(self.gaussianBeam(self.w_beam)).astype(self.dtype_f)
                # ガウシアン分布を各アドレスに振り分ける

                l = self.model.voxel_space
                pp = (gb / l).astype("int16")
                ind = np.where(gb < 0)
                pp[ind[0].tolist(), ind[1].tolist()] -= 1
                pa = gb - (pp + 1 / 2) * l
                ind = np.where((np.abs(pa) >= l / 2))
                pa[ind[0].tolist(), ind[1].tolist()] = \
                    np.sign(pa[ind[0].tolist(), ind[1].tolist()]) * (l / 2)
                pa += l / 2
                self.add[:2] = self.add[:2] + pp
                self.p[:2] = pa.astype(self.dtype_f)
        else:
            print("ビームタイプが設定されていません")

    def _get_beam_dist(self, x, y):
        fig = plt.figure(figsize=(10, 6), dpi=70)
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        H = ax.hist2d(x, y, bins=100, cmap="plasma")
        ax.set_title('Histogram for laser light intensity')
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        fig.colorbar(H[3], ax=ax)
        plt.show()

    def gaussianBeam(self, w=0.54):
        # TEM00のビームを生成します
        r = np.linspace(-w * 2, w * 2, 100)
        # Ir = 2*np.exp(-2*r**2/(w**2))/(np.pi*(w**2))
        Ir = np.exp(-2 * r ** 2 / (w ** 2))
        normd = stats.norm(0, w / 2)
        x = normd.rvs(self.nPh)
        y = normd.rvs(self.nPh)
        # z = np.zeros(self.nPh)

        fig, ax1 = plt.subplots()
        ax1.set_title('Input laser light distribution')
        ax1.hist(x, bins=100, color="C0")
        ax1.set_ylabel('Number of photon')
        ax2 = ax1.twinx()
        ax2.plot(r, Ir, color="k")
        ax2.set_xlabel('X [mm]')
        ax2.set_ylabel('Probability density')
        plt.show()
        self._get_beam_dist(x, y)
        return x, y

    def get_result(self):
        encoded_position = self._encooder(self.p_result, self.add_result)
        df_result = {
            'p': encoded_position,
            'v': self.v_result,
            'w': self.w_result,
            'nPh': self.nPh
        }
        return df_result

    def get_model_params(self):
        return self.model.get_params()

    def _encooder(self, p, add):
        space = self.model.voxel_space
        center_add_x = self._get_center_add(self.model.voxel_model.shape[0])
        center_add_y = self._get_center_add(self.model.voxel_model.shape[1])
        encoded_position = p.copy()
        encoded_position[0] = space * (add[0] - center_add_x) + p[0]
        encoded_position[1] = space * (add[1] - center_add_y) + p[1]
        encoded_position[2] = np.round(space * (add[2] - 1) + p[2] + space / 2, 6)
        return encoded_position

    def set_monte_params(self, *, nPh, model, dtype_f=np.float32, dtype=np.int32, w_beam=0):
        self.dtype_f = dtype_f
        self.dtype = dtype
        self.nPh = nPh
        self.w_beam = w_beam
        self.model = model

    def build(self, *initial_data, **kwargs):
        if initial_data == () and kwargs == {}:
            pass
        else:
            self.model.set_params(*initial_data, **kwargs)
        self.model.build()

    def getRdTtRate(self):
        self.Tt_index = np.where(self.v_result[2] > 0)[0]
        self.Rd_index = np.where(self.v_result[2] < 0)[0]
        self.Rdw = self.w_result[self.Rd_index].sum() / self.nPh
        self.Ttw = self.w_result[self.Tt_index].sum() / self.nPh
        print('######')
        print('Mean Rd %0.6f' % self.Rdw)
        print('Mean Td %0.6f' % self.Ttw)
        print()

    def save_result(self, fname, coment=''):
        start_ = time.time()

        res = self.get_result()
        save_name = fname + "_LID.pkl.bz2"
        with bz2.open(save_name, 'wb') as fp:
            fp.write(pickle.dumps(res))
        print("Monte Carlo results saved in ")
        print("-> %s" % (save_name))
        print('')
        info = self._calc_info(coment)
        save_name = fname + "_info.json"
        with open(save_name, 'w') as fp:
            json.dump(info, fp, indent=4, cls=ToJsonEncoder)
        print("Calculation conditions are saved in")
        print("-> %s" % (save_name))
        print('')

        calTime(time.time(), start_)

    def _calc_info(self, coment=''):
        _params = self.model.get_params()
        calc_info = {
            'Date': datetime.datetime.now().isoformat(),
            'coment': coment,
            'number_of_photons': self.nPh,
            'calc_dtype': "32 bit",
            'model': {
                'model_name': self.model.model_name,
                'model_params': _params,
                'model_voxel_space': self.model.voxel_space,
                'model_xy_size': self.model.xy_size,
            },
            'w_beam': self.w_beam,
            'beam_angle': self.beam_angle,
            'initial_refrect_mode': self.initial_refrect_by_angle,
            'beam_mode': 'TEM00',
            'fluence_mode': self.fluence_mode,
        }
        return calc_info


# =============================================================================
# Modeling class
# =============================================================================

class VoxelModel:
    def build(self):
        pass

    def set_params(self):
        pass

    def getModelSize(self):
        print("Memory area size for voxel storage: %0.3f Mbyte" % (self.voxel_model.nbytes * 1e-6))


class TuringModel_Rectangular(VoxelModel):
    def __init__(self):
        self.model_name = 'TuringModel_Rectangular'
        self.dtype_f = np.float32
        self.dtype = np.int8
        self.ct_num = 2
        self.subc_num = 3
        self.skin_num = 4
        self.end_point = 5
        self.voxel_model = np.zeros((3, 3, 3), dtype=self.dtype)

        self.params = {
            'xz_size': 17.15, 'voxel_space': 0.0245, 'dicom_path': False, 'bv_tv': 0.138,
            'th_cortical': 1., 'th_subcutaneus': 2.6, 'th_dermis': 1.4,
            'n_space': 1., 'n_trabecular': 1.4, 'n_cortical': 1.4, 'n_subcutaneus': 1.4, 'n_dermis': 1.4, 'n_air': 1.,
            'ma_space': 1e-8, 'ma_trabecular': 0.02374, 'ma_cortical': 0.02374, 'ma_subcutaneus': 0.011,
            'ma_dermis': 0.037,
            'ms_space': 1e-8, 'ms_trabecular': 20.54, 'ms_cortical': 17.67, 'ms_subcutaneus': 20, 'ms_dermis': 20,
            'g_space': 0.90, 'g_trabecular': 0.90, 'g_cortical': 0.90, 'g_subcutaneus': 0.90, 'g_dermis': .90,
        }
        self.keys = list(self.params.keys())
        self._make_model_params()
        self.voxel_space = self.params['voxel_space']

    def build(self, bone_model):
        # thickness,xy_size,voxel_space,ma,ms,g,n,n_air
        del self.voxel_model
        gc.collect()
        self.voxel_model = bone_model

        self.voxel_space = self.params['voxel_space']
        self._make_voxel_model()
        self.getModelSize()

    def set_params(self, *initial_data, **kwargs):
        set_params(self.params, self.keys, *initial_data, **kwargs)
        self._make_model_params()

    def _make_model_params(self):
        # パラメーターは、[骨梁間隙, 海綿骨, 緻密骨, 皮下組織, 皮膚, 外気]のように設定されています。
        name_list = ['_space', '_trabecular', '_cortical', '_subcutaneus', '_dermis']
        _n = [];
        _ma = [];
        _ms = [];
        _g = []
        for i in name_list:
            _n.append(self.params['n' + i])
            _ma.append(self.params['ma' + i])
            _ms.append(self.params['ms' + i])
            _g.append(self.params['g' + i])
        _n.append(self.params['n_air'])
        self.n = np.array(_n).astype(self.dtype_f)
        self.ma = np.array(_ma).astype(self.dtype_f)
        self.ms = np.array(_ms).astype(self.dtype_f)
        self.g = np.array(_g).astype(self.dtype_f)

    def get_params(self):
        return {
            'ms': self.ms,
            'ma': self.ma,
            'n': self.n,
            'g': self.g
        }

    def _read_dicom(self):
        path = self.params['dicom_path']
        files = os.listdir(path)
        files.sort()
        self.params['voxel_space'] = round(float(pydicom.dcmread(path + "/" + files[0], force=True).PixelSpacing[0]), 5)

        ds = []
        for i in files:
            ds.append(pydicom.dcmread(path + "/" + i, force=True).pixel_array)
        ds = np.array(ds).astype("int8")
        return ds

    def add_array(self, X, num_pix, val, dtype, y_axis=False):
        # Z方向
        ct = np.zeros((X.shape[0], X.shape[1], num_pix), dtype=dtype) + val
        X = np.concatenate((ct, X), 2)
        X = np.concatenate((X, ct), 2)
        # X方向
        ct = np.zeros((num_pix, X.shape[1], X.shape[2]), dtype=dtype) + val
        X = np.concatenate((ct, X), 0)
        X = np.concatenate((X, ct), 0)
        # Y方向
        if y_axis:
            ct = np.zeros((X.shape[0], num_pix, X.shape[2]), dtype=dtype) + val
            X = np.concatenate((ct, X), 1)
            X = np.concatenate((X, ct), 1)

        return X

    def _make_voxel_model(self):
        if self.params['dicom_path']:
            self.voxel_model = self._read_dicom()
        A = np.zeros_like(self.voxel_model).astype(bool)
        list_num = [self.ct_num, self.subc_num, self.skin_num]
        num_s = np.round(np.array(
            [self.params['th_cortical'], self.params['th_subcutaneus'], self.params['th_dermis']]
        ) / self.params["voxel_space"]).astype(np.int)

        int_num = int(
            self.voxel_model.shape[0] / 2 - round(self.params["xz_size"] / (self.params["voxel_space"] * 2))) + num_s[0]
        A[int_num:-int_num, :, int_num:-int_num] = 1

        x = 0
        for i in A[:, int(A.shape[2] / 2), int(A.shape[0] / 2)]:
            if i:
                break
            x += 1
        A = A[x:-x, :, x:-x]
        self.voxel_model = self.voxel_model[x:-x, :, x:-x]

        for i in tqdm(range(3)):
            self.voxel_model = self.add_array(self.voxel_model, num_s[i], list_num[i], np.int8)

        self.voxel_model = self.add_array(self.voxel_model, 1, self.end_point, np.int8, y_axis=True)
        print("Shape of voxel_model ->", self.voxel_model.shape)


class TuringModel_DICOM(TuringModel_Rectangular):
    def __init__(self):
        self.model_name = 'TuringModel_DICOM'
        self.dtype_f = np.float32
        self.dtype = np.int8
        self.voxel_model = np.zeros((3, 3, 3), dtype=self.dtype)
        self.end_point = 5

        self.params = {
            'xz_size': False, 'voxel_space': 0.0245, 'dicom_path': False, 'bv_tv': 0.138,
            'th_cortical': False, 'th_subcutaneus': False, 'th_dermis': False,
            'n_space': 1., 'n_trabecular': 1.4, 'n_cortical': 1.4, 'n_subcutaneus': 1.4, 'n_dermis': 1.4, 'n_air': 1.,
            'ma_space': 1e-8, 'ma_trabecular': 0.02374, 'ma_cortical': 0.02374, 'ma_subcutaneus': 0.011,
            'ma_dermis': 0.037,
            'ms_space': 1e-8, 'ms_trabecular': 20.54, 'ms_cortical': 17.67, 'ms_subcutaneus': 20, 'ms_dermis': 20,
            'g_space': 0.90, 'g_trabecular': 0.90, 'g_cortical': 0.90, 'g_subcutaneus': 0.90, 'g_dermis': .90,
            'vol_trabecular': np.zeros((0, 0, 0)), 'vol_cortical': np.zeros((0, 0, 0)), 'vol_subcutaneus': np.zeros((0, 0, 0)),
            'vol_dermis': np.zeros((0, 0, 0)), 'distal_radius_idx': 0, 
        }
        self.keys = list(self.params.keys())
        self._make_model_params()
        self.voxel_space = self.params['voxel_space']

    def set_params(self, params):
        set_params(self.params, **params)
        idx = int(self.params["distal_radius_idx"])
        self.params["area_obs"] = self.params['vol_cortical'].calc_area(idx)
        self.params["area_trabecular"] = self.params['vol_trabecular'].calc_area(idx)
        print(f"A_obs: {self.params['area_obs']:.2f}[cm2]")
        print(f"A_tra: {self.params['area_trabecular']:.2f}[cm2]")
        self._make_model_params()

    def build(self, bone_model):
        # thickness,xy_size,voxel_space,ma,ms,g,n,n_air
        del self.voxel_model
        gc.collect()
        self.voxel_model = bone_model

        self.voxel_space = self.params['voxel_space']
        self._make_voxel_model()
        self.getModelSize()
        
    def _make_voxel_model(self):
        model, *others = [self.params[key] for key in self.params if key.split('_')[0] == "vol"]
        for seg in others:
            model += seg
        model.name = 'model'
        voxel_model = model.add_endpoint()
        voxel_model = voxel_model.synthesize(self.voxel_model)
        self.voxel_model = self.add_array(voxel_model.array, 1, voxel_model.id_['endpoint'], np.int8, y_axis=True)
        print("Shape of voxel_model ->", self.voxel_model.shape)
        self.params["shape"] = self.voxel_model.shape

        for key in list(self.params):
            if key.split('_')[0] == "vol":
                del self.params[key]
        gc.collect()


# =============================================================================
# Public montecalro model
# =============================================================================


class DicomTuringModel(BaseVoxelMonteCarlo):
    def __init__(
            self, *, nPh=1000, dtype_f=np.float32, dtype=np.int32,
            beam_type='TEM00', w_beam=0,
            beam_angle=0, initial_refrect_by_angle=False,
            first_layer_clear=False, model_name='TuringModel',
    ):
        model = TuringModel_DICOM()
        self.params = {"mode": "encooder option",
                       "origin": ("x", "y", "z"),
                       "vec": ("x", "y", "z"),
                       }
        self.keys_params = list(self.params.keys())
        initial_refrect_by_angle = True

        super().__init__(
            nPh=nPh, model=model, dtype_f=dtype_f, dtype=dtype,
            w_beam=w_beam, beam_angle=beam_angle, beam_type=beam_type,
            initial_refrect_by_angle=initial_refrect_by_angle,
            first_layer_clear=first_layer_clear,
        )

        self.bone_model = False

    def _set_inital_add(self):
        if self.beam_type == 'TEM00':
            self.add = np.zeros((3, self.nPh), dtype=self.dtype)
        def __get_last_num_xyz(a, xyz):
            aa = np.round(a + xyz).astype(np.int_)
            if self.model.voxel_model[aa[0], aa[1], aa[2]] == self.model.end_point:
                return np.round(a).astype(np.int_)
            return __get_last_num_xyz(a + xyz, xyz)

        a = self.origin
        xyz = -self.vec
        add = __get_last_num_xyz(a, xyz)
        print(f"Last add for xyz-axis is {add}")
        self.params["address"] = add
        self.add[0] = add[0]
        self.add[1] = add[1]
        self.add[2] = add[2]

    def _set_inital_vector(self):
        if self.beam_type == 'TEM00':
            self.v = np.zeros((3, self.nPh)).astype(self.dtype_f)
            if self.beam_angle != 0 and self.w_beam == 0:
                # ビーム径がある場合はとりあえず無視
                # 角度はrad表記
                ni = self.model.n[-1]
                nt = self.model.n[0]
                ai = self.beam_angle
                print(f"angle of incidence:{np.rad2deg(ai):.2f}[deg]")
                self.params["incidence_angle"] = ai
                at = np.arcsin(np.sin(ai) * ni / nt)
                print(f"angle of transmission:{np.rad2deg(at):.2f}[deg]")
                self.params["transmit_angle"] = at
                vec_t = self._calc_transmit_vec(ai, at)
                self.params["incidence_vec"] = self.vec
                self.params["transmit_vec"] = vec_t
                self.v[0] = vec_t[0]
                self.v[1] = vec_t[1]
                self.v[2] = vec_t[2]
                if self.initial_refrect_by_angle:
                    Ra = ((np.sin(ai - at) / np.sin(ai + at)) ** 2 + (np.tan(ai - at) / np.tan(ai + at)) ** 2) / 2
                    self.inital_del_num = np.count_nonzero(Ra >= np.random.rand(self.nPh))
                    self.v = np.delete(self.v, np.arange(self.inital_del_num), 1)
                    self.p = np.delete(self.p, np.arange(self.inital_del_num), 1)
                    self.add = np.delete(self.add, np.arange(self.inital_del_num), 1)
                    sub_v = np.zeros((3, self.inital_del_num)).astype(self.dtype_f)
                    sub_v[0] = -self.vec[0]
                    sub_v[1] = -self.vec[1]
                    sub_v[2] = self.vec[2]
                    self.v_result = sub_v
                    self.p_result = self.p[:, :self.inital_del_num]
                    self.add_result = self.add[:, :self.inital_del_num]
                viz.laser_position_3d(volume=self.model.voxel_model,
                                      end_point=self.model.end_point,
                                      spacing=self.model.voxel_space,
                                      address=self.add.T[0],
                                      vector=self.vec,
                                      norm=100,
                                      color="red",
                                      )
        else:
            print("ビームタイプが設定されていません")

    def _set_inital_w(self):
        if self.beam_type == 'TEM00':
            self.w = np.ones(self.nPh).astype(self.dtype_f)
            Rsp = 0
            n1 = self.model.n[-1]
            n2 = self.model.n[0]
            if n1 != n2:
                Rsp = ((n1 - n2) / (n1 + n2)) ** 2
                if self.beam_angle != 0 and self.w_beam == 0:
                    ai = self.beam_angle
                    at = np.arcsin(np.sin(ai) * n1 / n2)
                    Rsp = ((np.sin(ai - at) / np.sin(ai + at)) ** 2 + (np.tan(ai - at) / np.tan(ai + at)) ** 2) / 2
                elif self.first_layer_clear:
                    n3 = self.model.n[1]
                    r2 = ((n3 - n2) / (n3 + n2)) ** 2
                    Rsp = Rsp + r2 * (1 - Rsp) ** 2 / (1 - Rsp * r2)
                self.w -= Rsp

            if self.beam_angle != 0 and self.w_beam == 0:
                if self.initial_refrect_by_angle:
                    self.w[:] = 1
                    self.w = np.delete(self.w, np.arange(self.inital_del_num), 0)
                    self.w_result = self.w[:self.inital_del_num]
        else:
            print("ビームタイプが設定されていません")

    def _set_beam_distribution(self):
        if self.beam_type == 'TEM00':
            self.p = np.zeros((3, self.nPh)).astype(self.dtype_f)
            ratio = self.model.voxel_space * 0.5 / np.max(np.abs(self.vec))
            self.p[0] = ratio * (-self.vec[0])
            self.p[1] = ratio * (-self.vec[1])
            self.p[2] = ratio * (-self.vec[2])
            self.params["position"] = ratio * (-self.vec)
            if self.w_beam != 0:
                print("%sを入力" % self.beam_type)
                # ガウシアン分布を生成
                gb = np.array(self.gaussianBeam(self.w_beam)).astype(self.dtype_f)
                # ガウシアン分布を各アドレスに振り分ける

                l = self.model.voxel_space
                pp = (gb / l).astype("int16")
                ind = np.where(gb < 0)
                pp[ind[0].tolist(), ind[1].tolist()] -= 1
                pa = gb - (pp + 1 / 2) * l
                ind = np.where((np.abs(pa) >= l / 2))
                pa[ind[0].tolist(), ind[1].tolist()] = \
                    np.sign(pa[ind[0].tolist(), ind[1].tolist()]) * (l / 2)
                pa += l / 2
                self.add[:2] = self.add[:2] + pp
                self.p[:2] = pa.astype(self.dtype_f)
        else:
            print("ビームタイプが設定されていません")

    def _calc_transmit_vec(self, ai, at):
        vec = np.empty(3)
        vec[0] = np.sign(self.vec[0]) * np.cos(at)
        vec[1] = self.vec[1] * np.sin(at) / np.sin(ai)
        vec[2] = self.vec[2] * np.sin(at) / np.sin(ai)
        return vec

    def _end_process(self):
        if self.initial_refrect_by_angle:
            self.v_result = np.concatenate([self.v_result, self.v], axis=1)
            self.p_result = np.concatenate([self.p_result, self.p], axis=1)
            self.add_result = np.concatenate([self.add_result, self.add], axis=1)
            self.w_result = np.concatenate([self.w_result, self.w])
        else:
            self.v_result = self.v
            self.p_result = self.p
            self.add_result = self.add
            self.w_result = self.w

    def _encooder(self, p, add):
        space = self.model.voxel_space
        if self.mode == "R1":
            center_add = self.origin
            encoded_position = p.copy()
            encoded_position[0] = space * (add[0] - center_add[0]) + p[0]
            encoded_position[1] = space * (add[1] - center_add[1]) + p[1]
            encoded_position[2] = space * (add[2] - center_add[2]) + p[2]
        elif self.mode == "3DS":
            center_add = self.params["address"]
            encoded_position = p.copy()
            encoded_position[0] = space * (add[0] - center_add[0]) + p[0]
            encoded_position[1] = space * (add[1] - center_add[1]) + p[1]
            encoded_position[2] = space * (add[2] - center_add[2]) + p[2]
        else:
            raise KeyError(self.option)
        return encoded_position

    def build(self, params):
        self.model.set_params(params)
        self.model.build(self.bone_model)
        del self.bone_model
        gc.collect()

    def set_model(self, u):
        self.bone_model = u

    def set_params(self, params):
        set_params(self.params, **params)
        self._setatter_params()

    def _setatter_params(self):
        for k in self.params:
            if isinstance(self.params[k], list):
                setattr(self, k, np.array(self.params[k]))
            else:
                setattr(self, k, self.params[k])

        def __calc_beam_angle(vec):
            return np.arccos(np.max(np.abs(vec)))

        self.beam_angle = __calc_beam_angle(self.vec)

    def get_model_fig_3d(self):
        viz.model_3d(volume=self.model.voxel_model,
                     end_point=self.model.end_point,
                     spacing=self.model.voxel_space,
                     )

    def _calc_info(self, coment=''):
        calc_info = {
            'Date': datetime.datetime.now().isoformat(),
            'coment': coment,
            'number_of_photons': self.nPh,
            'calc_dtype': "32 bit",
            'model': {
                'model_name': self.model.model_name,
                'model_params': self.model.params,
            },
            'w_beam': self.w_beam,
            'beam_angle': self.beam_angle,
            'initial_refrect_mode': self.initial_refrect_by_angle,
            'beam_mode': 'TEM00',
            'laser': self.params
        }
        return calc_info