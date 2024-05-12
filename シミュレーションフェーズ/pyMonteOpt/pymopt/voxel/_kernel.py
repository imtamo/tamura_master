#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 22 18:14:23 2022
@author: Kaname Miura
"""
import numpy as np
from tqdm import tqdm
from numba import njit


@njit(fastmath=True)
def theta(g,rand):
    th = 0
    if g != 0.:
        th = (1 + g**2 - ((1 - g**2) / (1 - g + 2 * g * rand))** 2) / (2 * g)
        if th < -1:
            th = -1
    else:
        th = 2 * rand - 1
    return th

@njit(fastmath=True)
def vmc_kernel(
    add, p,v, w, ma, ms, n, g,
    voxel_model, l,
    nPh, end_point
    ):
    ma = ma.astype(np.float32)
    ms = ms.astype(np.float32)
    n = n.astype(np.float32)
    g = g.astype(np.float32)
    nPh = np.int32(nPh)
    count = 0
    counter = np.int32(nPh/10)
    for idx in range(nPh):
        add_ = np.array([0,0,0]).astype(np.int32)
        v_ = np.array([0,0,0]).astype(np.float32)
        zero_vec = np.array([0,0,0]).astype(np.int32)
        one_vec = np.array([1,1,1]).astype(np.int32)

        wth = np.float32(0.0001)
        roulette_m = np.int32(10)
        index_ = voxel_model[add[0,idx], add[1,idx], add[2,idx]]
        index_next = np.int8(0)
        index_end = np.int8(end_point)

        fi = np.float32(0);
        cos_fi = np.float32(0); cos_th = np.float32(0)
        sin_fi = np.float32(0); sin_th = np.float32(0);

        valf = np.float32(0.); dbnum = np.int32(0); db = np.float32(1000.)

        ni = np.float32(n[index_])
        nt = np.float32(0)
        mt = np.float32(ma[index_] + ms[index_])
        st = np.float32(0)
        flag_tr = True
        flag_end = False
        while True:
            st = np.float32(-np.log(np.random.rand()))
            while True:
                valf = 0.; dbnum = 0; db = 1000.
                for i in range(3):
                    if (np.abs(v[i,idx]) > 0):
                        valf = (l / 2 - np.sign(v[i,idx]) * p[i,idx])/ np.abs(v[i,idx])
                        if (valf < db):
                            dbnum = i; db = valf
                if st>= db*mt:
                    for i in range(3):
                        p[i,idx]+=v[i,idx]*db
                    p[dbnum,idx] = l/2*np.sign(v[dbnum,idx])
                    st-=db*mt
                    for i in range(3):
                        add_[i] = add[i,idx]
                    add_[dbnum] += np.sign(v[dbnum,idx])
                    index_next = voxel_model[add_[0], add_[1], add_[2]]

                    nt = n[index_next]
                    flag_tr = True

                    if (ni != nt):
                        ra = 0;at = 0
                        ai = np.arccos(abs(v[dbnum,idx]))
                        if (ni > nt)&(ai >= np.arcsin(nt/ni)):
                            ra = 1
                        else:
                            at = np.arcsin((ni/nt)*np.sin(ai))
                            if ai!=0:
                                ra = ((np.sin(ai - at) / np.sin(ai + at))**2 \
                                +(np.tan(ai - at) / np.tan(ai + at))**2)/2
                            else:
                                ra = 0
                        rand_ = np.random.rand()

                        if ra < rand_:
                            flag_tr = True
                            zero_vec[dbnum] = 1; one_vec[dbnum] = 0
                            for i in range(3):
                                v[i,idx] = one_vec[i]*v[i,idx]*ni/nt\
                                +zero_vec[i] * np.sign(v[i,idx]) * np.cos(at)
                                zero_vec[i] = 0; one_vec[i] = 1

                            valf = np.sqrt(v[0,idx]**2+v[1,idx]**2+v[2,idx]**2)
                            for i in range(3):
                                v[i,idx] /= valf
                        else:
                            flag_tr = False

                    if flag_tr:
                        add[dbnum,idx] += np.sign(v[dbnum,idx])
                        p[dbnum,idx] *= -1
                        index_ = index_next
                        if index_ == index_end:
                             flag_end = True
                             break
                        mt = ma[index_] + ms[index_]
                        ni = nt
                    else:
                        v[dbnum,idx] *= -1
                else:
                    for i in range(3):
                        p[i,idx]+=v[i,idx]*st/mt
                    st = 0
                    break

            if flag_end:
                break

            w[idx] -= np.float32(w[idx]*ma[index_]/mt)

            if(w[idx]<=wth):
                if (1/roulette_m) < np.float32(np.random.rand()):
                    for i in range(3):
                        p[i,idx] = 0
                        v[i,idx] = 0
                        add[i,idx] = 0
                    w[idx] = 0
                    break
                else:
                    w[idx]*=roulette_m

            cos_th = theta(g[index_], np.float32(np.random.rand()))
            sin_th = np.sqrt(1-cos_th**2)

            fi = 2 * 3.1415927*np.float32(np.random.rand())
            cos_fi = np.cos(fi)
            sin_fi = np.sin(fi)

            if 0.99999 < abs(v[2,idx]):
                v[0,idx] = sin_th * cos_fi
                v[1,idx] = sin_th * sin_fi
                v[2,idx] = np.sign(v[2,idx]) * cos_th

            else:
                valf = np.sqrt(1 - v[2,idx]**2)
                for i in range(3):
                    v_[i] = v[i,idx]
                v[0,idx] = sin_th * (v_[0] * v_[2] * cos_fi - v_[1] * sin_fi) / valf + v_[0] * cos_th
                v[1,idx] = sin_th * (v_[1] * v_[2] * cos_fi + v_[0] * sin_fi) / valf + v_[1] * cos_th
                v[2,idx] = -sin_th * cos_fi * valf + v_[2] * cos_th

            valf = np.sqrt(v[0,idx]**2+v[1,idx]**2+v[2,idx]**2)
            for i in range(3):
                v[i,idx] /= valf

        flag_end = False
        if idx%counter == 0:
            count+=10
            print(count, " ％")

    return add, p,v, w
