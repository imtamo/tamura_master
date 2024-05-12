import os
import gc
import nrrd
import glob
import json
import time
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import namedtuple
from .pymopt import visualization as viz
from .pymopt import metrics as met
from .pymopt.preprocessing import Preprocessing
from .pymopt.modeling_gpu import TuringPattern
from .pymopt.voxel import DicomTuringModel
from .pymopt.utils import generate_variable_params

repetitions = 1211
nPh = 1e7
mode = '3DS'
path = './pyMonteOpt/subjects/subject0/'
reso = 0.25 # resolution for simulation [mm/px]
Split = namedtuple('Split', ['before', 'after', 'thickness'])
splits = [Split('else', ['subcutaneus', 'dermis'], 1.5),
          Split('bone', ['trabecular', 'cortical'], 0.487),
          ]

range_params_norm = {
    "th_dermis": [1, 2],
    "ma_dermis": [0.00633, 0.08560],
    "msd_dermis": [1.420, 2.506],
    "th_subcutaneus": [1, 6],
    "ma_subcutaneus": [0.00485, 0.01239],
    "msd_subcutaneus": [0.83, 1.396],
    "bv_tv": [0.134, 0.028],
    "th_cortical": [0.804, 0.149],
    "corr": 0.54,
}
range_params_osteoporosis = range_params_norm.copy()
range_params_osteoporosis["bv_tv"] = [0.085, 0.022]
range_params_osteoporosis["th_cortical"] = [0.487, 0.138]

range_params_osteopenia = range_params_norm.copy()
range_params_osteopenia["bv_tv"] = [0.103, 0.030]
range_params_osteopenia["th_cortical"] = [0.571, 0.173]

def determining_params_range():
    a = np.random.rand()
    range_params = 0
    if a <= 0.4:
        print("Osteoporosis")
        range_params = range_params_osteoporosis
    elif a >= 0.6:
        print("Normal")
        range_params = range_params_norm
    else:
        print("Osteopenia")
        range_params = range_params_osteopenia
    return range_params

model_params = {
    "grid": [30, 30, 30],
    "dx": 1 / 30,
    "dt": 1,
    "du": 0.0002,
    "dv": 0.01,
    "length": 10,
    "repetition": 100,
    "voxelsize": 0.0306,
    "seed": False,
    "ct_coef": 4.5e4,
    "tile_num_xz": 1,
    "tile_num_y": 1,
}

th_coef = np.array([-10.93021385, 2.62630274, -0.50913966, 0.60371039])

monte_params = {
    "voxel_space": model_params["voxelsize"],
    "xz_size": 17.15,

    "n_space": 1.4,
    "n_trabecular": 1.55,
    "n_cortical": 1.55,
    "n_subcutaneus": 1.4,
    "n_dermis": 1.4,
    "n_air": 1,

    "ma_trabecular": 0.02374,
    "ma_cortical": 0.02374,

    "ms_trabecular": 20.588,
    "ms_cortical": 20.588,

    "g_space": 0.90,
    "g_trabecular": 0.90,
    "g_cortical": 0.90,
    "g_subcutaneus": 0.90,
    "g_dermis": 0.90,
}

def preprocessing(path, reso, by, alias, save=False):
    path_ = {'input': f'{path}from_3dslicer/Segmentation.seg.nrrd',
             'dicom': f'{path}dicom/',
             'output': f'{path}to_simulation/{alias}/',
             }
    prep = Preprocessing()
    prep.set_path(path_)
    prep.reso = reso
    prep.load_data()
    
    prep.define_coordinate_system()
    prep.merge_bone()
    prep.slice_array(outermost='else')
    prep.upsampling()
    prep.track_control_points()
    prep.split_segment(by)
    if save:
      os.makedirs(path_['output'])
    seg, info = prep.save(save)
    return seg, info

def generate_bone_model(bv_tv, seg_params, opath, model_params):
    model_params['voxelsize'] = seg_params['reso']
    model_params["grid"] = np.array(seg_params["shape"]) / 10
    model_params["bv_tv"] = bv_tv

    tp = TuringPattern()
    tp.set_params(model_params)
    tp.set_threshold_func_coef(th_coef)
    if not os.path.exists(opath):
        os.makedirs(opath)
    u = tp.modeling(opath, save_dicom=True)
    bvtv = tp.bv_tv_real
    del tp
    gc.collect()
    return u, bvtv

def calc_montecalro(vp, iteral, params, seg, seg_params, path, u, mode):
    print()
    print("#" * 20)
    print("#%s" % iteral)

    nn = 0
    params["th_dermis"] = vp["th_dermis"][nn]
    params["ma_dermis"] = vp["ma_dermis"][nn]
    params["ms_dermis"] = vp["msd_dermis"][nn] / (1 - params["g_dermis"])

    params["th_subcutaneus"] = vp["th_subcutaneus"][nn]
    params["ma_subcutaneus"] = vp["ma_subcutaneus"][nn]
    params["ms_subcutaneus"] = vp["msd_subcutaneus"][nn] / (1 - params["g_subcutaneus"])

    params["ma_space"] = vp["ma_subcutaneus"][nn]
    params["ms_space"] = vp["msd_subcutaneus"][nn] / (1 - params["g_space"])

    params["bv_tv"] = vp["bv_tv"][nn]
    params["th_cortical"] = vp["th_cortical"][nn]
    
    params['vol_trabecular'] = seg['trabecular']
    params['vol_cortical'] = seg['cortical']
    params['vol_subcutaneus'] = seg['subcutaneus']
    params['vol_dermis'] = seg['dermis']
    
    params["voxel_space"] = seg_params['reso']
    params['distal_radius_idx'] = seg_params['cpts']['radial_styloid'][2]
    vec = -np.array(seg_params["xform"]["e_xL"])
    origin = seg_params["cpts"]["origin"]
    
    params_ = {"origin": origin,
               "mode": mode,
               "vec": vec,
               }

    model = DicomTuringModel(
        nPh=nPh,
        model_name="TuringModel_DICOM",
    )
    model.set_model(u)
    model.set_params(params_)  # set params for simulation
    model.build(params)  # set params for model used to simulation
    start = time.time()
    # model.get_model_fig_3d()
    model = model.start()
    print("%s sec" % (time.time() - start))
    print("#%s" % iteral)
    print("Save->%s" % path)
    model.save_result(path, coment="test")
    res = model.get_result()

    del model
    gc.collect()
    return res, params

def calc_ray_tracing(res, seg_params, path, alias_name, mode):    
    #viz.point_clouds_3d(point_clouds=res['p'].T)
    viz.light_intensity_map_3d(res['p'], res['w'])
    R = np.array(seg_params['xform']['R'])
    if mode == 'R1':
        nn = 180
        dth = np.pi / nn
        dy = 0.4
        y = 20
        R = np.array(seg_params['xform']['R'])
        alpha, beta, Rdr = met.ellipse4(res['p'], res['w'], dth, nn, y, dy, res['nPh'], R)
        #viz.light_intensity_map_2d(np.log(Rdr), x=alpha, y=beta)
    
        df = pd.DataFrame(Rdr)
        df.columns = alpha
        df.index = beta
        df.to_csv(path + ".csv")     
    elif mode == '3DS':
        nn = 300; dr = 30/nn
        nn_ = 400; dr_ = 40/nn_
    
        idx = np.where(res['v'][0]>0)[0]
        alphaRd, Rd = met.radialDistance2(res['p'][:, idx], res['w'][idx], nn, dr, res['nPh'], R)
        
        idx = np.where(res['v'][0]<0)[0]
        alphaTt, Tt = met.radialDistance2(res['p'][:, idx], res['w'][idx], nn, dr, res['nPh'], R)
        
        idx = np.where(res['v'][1]>0)[0]
        alpha_ssyz, Ssyz= met.lineDistance2(res['p'][:, idx], res['w'][idx], nn_, dr_, res['nPh'], R)
        
        path_ = path + "_B"
        aa = alias_name + "_B"
        df = pd.DataFrame()
        df[aa] = Rd
        df.index = alphaRd
        df.to_csv(path_ + ".csv")
    
        path_ = path + "_F"
        aa = alias_name + "_F"
        df = pd.DataFrame()
        df[aa] = Tt
        df.index = alphaTt
        df.to_csv(path_ + ".csv")
    
        path_ = path + "_L"
        aa = alias_name + "_L"
        df = pd.DataFrame()
        df[aa] = Ssyz
        df.index = alpha_ssyz
        df.to_csv(path_ + ".csv")     
    else:
        raise KeyError(mode)

def calc(iteral, path):
    gvp = generate_variable_params()
    range_params = determining_params_range()

    gvp.set_params(range_params)
    vp = gvp.generate(1)
    
    if vp["bv_tv"][0] > 0 and vp["th_cortical"][0] > 0:
        alias_name = "-".join(
            (str(datetime.datetime.now().isoformat()).split('.')[0]).split(':')) + "_it" + f"{iteral:04}"
        print("### iteral number", iteral)
        print("Alias name:", alias_name)
        
        seg, seg_params = preprocessing(path, reso, splits, alias_name)
        
        model_path = "./pyMonteOpt/model_result/"
        monte_path = "./pyMonteOpt/monte_result/"
        opt_path = "./pyMonteOpt/opt_result/"

        path = model_path + alias_name + "_dicom"
        u, bv_tv = generate_bone_model(vp["bv_tv"][0], seg_params, path, model_params)
        print("it:", iteral, ", change bvtv:", vp["bv_tv"][0], "-->", bv_tv)
        vp["bv_tv"][0] = bv_tv
        
        path = monte_path + alias_name
        res, params_ = calc_montecalro(vp, iteral, monte_params, seg, seg_params, path, u, mode)
        print("###### end monte carlo in it: ", iteral)

        path = opt_path + alias_name
        calc_ray_tracing(res, seg_params, path, alias_name, mode)
        print()
        print("########## End %s it ##########" % iteral)
        print()
    else:
        print("Invalid parameter was generated")
        print("BV/Tv : ", vp["bv_tv"][0])
        print("th_cortical : ", vp["th_cortical"][0])

def main():
    for iteral in range(repetitions):
        calc(iteral, path)
