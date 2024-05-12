import numpy as np
from .utilities import set_params

__all__ = [
    'generate_variable_params']


class generate_variable_params:
    def __init__(self):
        self.params = {
            'th_dermis': [1, 2],  # 皮膚厚さの範囲
            'ma_dermis': [0.00633, 0.08560],  # 皮膚吸収係数の範囲
            'msd_dermis': [1.420, 2.506],  # 皮膚減衰散乱係数の範囲
            'th_subcutaneus': [1, 6],  # 皮下組織厚さの範囲
            'ma_subcutaneus': [0.005, 0.012],  # 皮下組織吸収係数の範囲
            'msd_subcutaneus': [0.83, 1.396],  # 皮下組織減衰散乱係数の範囲
            # 'ma_marrow':[0.005,0.012],     # 骨髄吸収係数の範囲
            # 'msd_marrow':[0.83,1.396],     # 骨髄減衰散乱係数の範囲
            'bv_tv': [0.115, 0.02],  # 海綿骨BV/TVの平均と分散
            'th_cortical': [0.669, 0.133],  # 皮質骨厚さの平均と分散
            'corr': 0.54,  # 皮質骨と海綿骨の相関係数 Boutry2005
        }
        self.keys = list(self.params.keys())

    def set_params(self, params):
        set_params(self.params, **params)

    def generate(self, n):
        drmis = self._get_dermis_params(n)
        subcut = self._get_subcut_params(n)
        bone = self._get_bone_params(n)
        # marrow = self._get_marrow_params(n)
        self.int_params = {
            'th_dermis': drmis[0],
            'ma_dermis': drmis[1],
            'msd_dermis': drmis[2],
            'th_subcutaneus': subcut[0],
            'ma_subcutaneus': subcut[1],
            'msd_subcutaneus': subcut[2],
            # 'ma_marrow':marrow[0],
            # 'msd_marrow':marrow[1],
            'bv_tv': bone[0],
            'th_cortical': bone[1],
        }
        return self.int_params

    def get_variable_params(self):
        return self.int_params

    def _uniform_dist(self, range_, n):
        return np.random.rand(n) * (range_[1] - range_[0]) + range_[0]

    def _get_dermis_params(self, n):
        th_ = self._uniform_dist(self.params['th_dermis'], n)
        ma_ = self._uniform_dist(self.params['ma_dermis'], n)
        msd_ = self._uniform_dist(self.params['msd_dermis'], n)
        return th_, ma_, msd_

    def _get_subcut_params(self, n):
        th_ = self._uniform_dist(self.params['th_subcutaneus'], n)
        ma_ = self._uniform_dist(self.params['ma_subcutaneus'], n)
        msd_ = self._uniform_dist(self.params['msd_subcutaneus'], n)
        return th_, ma_, msd_

    def _get_bone_params(self, n):
        mean = [self.params['bv_tv'][0], self.params['th_cortical'][0]]  # [BV/TV, CTh]
        std = [self.params['bv_tv'][1], self.params['th_cortical'][1]]  # [BV/TV, CTh]

        corr = self.params['corr']
        cov = [[std[0] ** 2, std[0] * std[1] * corr],
               [std[0] * std[1] * corr, std[1] ** 2]]

        bvtv, cth = np.random.multivariate_normal(mean, cov, n).T
        return bvtv, cth

    def _get_marrow_params(self, n):
        ma_ = self._uniform_dist(self.params['ma_marrow'], n)
        msd_ = self._uniform_dist(self.params['msd_marrow'], n)
        return ma_, msd_
