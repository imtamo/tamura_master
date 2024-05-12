#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 13:32:41 2020

@author: kaname
"""
import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from .. import visualization as viz

__all__ = [
    'radialDistance',
    'angularyResolved',
    'lineDistance',
    'allround',
    'ellipse',
    'for_voxnet',
    'for_surface',
]


def radialDistance(p, w, nn, dr, nPh):  # 三浦さんの四角柱モデル（前方、後方散乱）
    alpha = np.array([(i) * dr for i in range(nn + 1)])
    da = np.array([2 * np.pi * (i + 0.5) * dr ** 2 for i in range(nn)])
    r = np.sqrt(p[0] ** 2 + p[1] ** 2)
    Rdr = []
    for i in range(nn):
        index = np.where((alpha[i] < r) & (alpha[i + 1] >= r))[0]
        Rdr.append(w[index].sum())
    Rdr = np.array(Rdr) / (da * nPh)
    return alpha[:-1], Rdr

def radialDistance2(p, w, nn, dr, nPh, R):  # 田村のDICOMモデル（前方、後方散乱）
    p = _homogeneous_xformation(p, R)
    alpha = np.array([(i) * dr for i in range(nn + 1)])
    da = np.array([2 * np.pi * (i + 0.5) * dr ** 2 for i in range(nn)])
    r = np.sqrt(p[1] ** 2 + p[2] ** 2)
    Rdr = []
    for i in range(nn):
        index = np.where((alpha[i] < r) & (alpha[i + 1] >= r))[0]
        Rdr.append(w[index].sum())
    Rdr = np.array(Rdr) / (da * nPh)
    return alpha[:-1], Rdr


def angularyResolved(v, w, nn, nPh):  # 三浦さんの四角柱モデル（角度分解）
    da = np.pi / (2 * nn)
    alpha = np.array([(i + 0.5) * da for i in range(nn + 1)])
    alpha2 = np.array([(i) * da for i in range(nn + 1)])
    do = 4 * np.pi * np.sin(alpha) * np.sin(da / 2)
    at = np.arccos(np.sign(v[2]) * (v[2]))
    Rda = []
    for i in range(nn):
        index = np.where((alpha2[i] < at) & (alpha2[i + 1] >= at))[0]
        Rda.append(w[index].sum())
    Rda = np.array(Rda) / (do[:-1] * nPh)
    return alpha[:-1], Rda


def lineDistance(p, w, nn, dr, nPh, y_range=5):  # 三浦さんの四角柱モデル（側方散乱）
    alpha = np.array([(i) * dr for i in range(nn + 1)])
    da = np.ones(nn) * dr * y_range * 2
    ind = np.where((np.abs(p[1]) < y_range))[0]
    p = p[:, ind].copy()
    r = p[2]
    Rdr = []
    for i in range(nn):
        index = np.where((alpha[i] < r) & (alpha[i + 1] >= r))[0]
        Rdr.append(w[index].sum())
    Rdr = np.array(Rdr) / (da * nPh)
    return alpha[:-1], Rdr

def lineDistance2(p, w, nn, dr, nPh, R, z_range=5):  # 田村のDICOMモデル（側方散乱）
    p = _homogeneous_xformation(p, R)
    alpha = np.array([(i) * dr for i in range(nn + 1)])
    da = np.ones(nn) * dr * z_range * 2
    ind = np.where((np.abs(p[2]) < z_range))[0]
    p = p[:, ind].copy()
    r = -p[0]
    Rdr = []
    for i in range(nn):
        index = np.where((alpha[i] < r) & (alpha[i + 1] >= r))[0]
        Rdr.append(w[index].sum())
    Rdr = np.array(Rdr) / (da * nPh)
    return alpha[:-1], Rdr


def allround(p, w, dth, nn, y, dy, r, nPh):  # 大竹さんの円柱モデル
    da = 2 * np.pi / nn

    a = np.array([i * da for i in range(nn + 1)])
    alpha = np.array([(i + 0.5) * da for i in range(nn)])
    A = 0.5 * da * dy * np.ones_like(alpha)

    c_ = p[0].astype(complex)
    c_ *= 1j
    c = np.angle(p[2] - p[2].max() + c_) + np.pi

    Rdr = []
    for j in range(y):
        rdr = []
        for i in range(nn):
            index = np.where(
                (a[i] < c) & (c <= a[i + 1]) &
                (-(y * dy / 2) + j * dy <= p[1]) & (p[1] <= -(y * dy / 2) + (j + 1) * dy)
            )[0]
            rdr.append(w[index].sum())
        Rdr.append(rdr)
    Rdr = np.array(Rdr) / (A * nPh)

    return alpha, Rdr


def ellipse(p, w, dth, nn, y, dy, nPh):  # 田村のDICOMモデル
    """
    上腕の解剖学的な原点を円の中心に利用して、表面データをスコアリングする
    同次変換：未対応
    GPU：未対応
    """
    alpha = np.array([np.deg2rad(i) for i in range(-nn, nn + 1)])
    beta = np.array([-y * 0.5 + dy * j for j in range(int(y / dy) + 1)])
    c_ = p[1].astype(complex)
    c_ *= 1j
    c = np.angle(p[0] + c_)

    Rdr = []
    dA = []
    for j in range(int(y / dy)):
        rdr = []
        da = []
        for i in range(2 * nn):
            index = np.where((alpha[i] < c) & (c <= alpha[i + 1]) & (beta[j] < p[2]) & (p[2] <= beta[j + 1]))[0]
            rdr.append(w[index].sum())
            r = np.mean(np.sqrt(p[0, index] ** 2 + p[1, index] ** 2))
            da.append(r * dth * dy)
        Rdr.append(rdr)
        dA.append(da)
    Rdr = np.array(Rdr) / (np.array(dA) * nPh)
    return np.rad2deg(alpha[:-1]), beta[:-1], Rdr

def ellipse2(p, w, dth, nn, y, dy, nPh, R): # 田村のDICOMモデル
    """
    上腕の解剖学的な原点を楕円の中心に利用して、表面データをスコアリングする
    同次変換：対応
    GPU：未対応
    """
    p = _homogeneous_xformation(p, R)
    viz.point_clouds_3d(p.T)
    
    return ellipse(p, w, dth, nn, y, dy, nPh)

def ellipse3(p, w, dth, nn, y, dy, nPh):  # 田村のDICOMモデル
    """
    上腕の解剖学的な原点を楕円の中心に利用して、表面データをスコアリングする
    同次変換：未対応
    GPU：対応
    """
    #cp.get_default_memory_pool().free_all_blocks()
    #cp.get_default_pinned_memory_pool().free_all_blocks()
    p = cp.asarray(p)
    w = cp.asarray(w)
    alpha = cp.asarray([np.deg2rad(i) for i in range(-nn, nn+1)])
    beta = cp.asarray([-y * 0.5 + dy * j for j in range(int(y / dy) + 1)])
    c_ = p[1].astype(complex)
    c_ *= 1j
    c = cp.angle(p[0] + c_)

    Rdr = cp.empty((int(y / dy), 2*nn))
    dA = cp.empty((int(y / dy), 2*nn))
    for j in range(int(y / dy)):
        rdr = cp.empty(2*nn)
        da = cp.empty(2*nn)
        for i in range(2*nn):
            index = cp.where((alpha[i] < c) & (c <= alpha[i + 1]) & (beta[j] < p[2]) & (p[2] <= beta[j + 1]))[0]
            rdr[i] = w[index].sum()
            r = cp.mean(cp.sqrt(p[0, index] ** 2 + p[1, index] **2))
            da[i] = r * dth * dy
        Rdr[j] = rdr
        dA[j] = da
    Rdr = Rdr / (dA * nPh)
    return cp.asnumpy(cp.rad2deg(alpha[:-1])), cp.asnumpy(beta[:-1]), cp.asnumpy(Rdr)

def ellipse4(p, w, dth, nn, y, dy, nPh, R): # 田村のDICOMモデル
    """
    上腕の解剖学的な原点を楕円の中心に利用して、表面データをスコアリングする
    同次変換：対応
    GPU：未対応
    """
    cp.get_default_memory_pool().free_all_blocks()
    cp.get_default_pinned_memory_pool().free_all_blocks()
    p = cp.asarray(p)
    w = cp.asarray(w)
    R = cp.asarray(R)
    p = _homogeneous_xformation(p, R, 'gpu')
    #viz.point_clouds_3d(cp.asnumpy(p).T)
   
    return ellipse3(p, w, dth, nn, y, dy, nPh)

def _homogeneous_xformation(point_clouds, mtx, kernel='cpu'):
    if kernel=='cpu':
        return np.dot(mtx, point_clouds)
    elif kernel=='gpu':
        return cp.dot(mtx, point_clouds)
    else:
        raise KeyError(kernel)
