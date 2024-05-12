import nrrd
import glob
import cv2
import json
import bisect
import pydicom as dcm
import numpy as np
import matplotlib.pyplot as plt
from pprint import pprint
from scipy import ndimage
from .. import visualization as viz
from ..utils import set_params, unit_conversion, ToJsonEncoder



class Preprocessing:
    LENGTH = 10 # distance between ulna styloid and laser incident position defined by Miuea[mm]
    SLICE_LENGTH = 35.28 # Bone axial length of synthetic biological tissue model defined by Miura [mm]
    
    def __init__(self):
        self.path = {'input': 'store segmentation',
                     'dicom': 'store dicom',
                     'output': 'for save result',
                     }
        self._reso = None # [mm/px]
        self.segments = {} # store segment objects
        self.dicom_params = {} # store dicom tag
        self.xform_params = {} # store anatomical coordinate system as matrix

    def set_path(self, path):
        set_params(self.path, **path)
        
    @property
    def reso(self):
        return self._reso
    
    @reso.setter
    def reso(self, reso):
        self._reso = reso
        Segment.reso = reso

    def load_data(self):
        self._load_dicom()
        self._load_segment()
        
    def _load_segment(self):
        array, header = nrrd.read(self.path['input'], index_order='F')
        seg_num = array.max() - 1
        
        for i in range(seg_num):
            name_i = header[f'Segment{i}_Name']
            label_i = int(header[f'Segment{i}_LabelValue'])
            array_i = (array == label_i).astype(np.uint8)
            self.segments[name_i] = Segment(name_i, array_i)
            #viz.segment_3d(self.segments[name_i].array, self.dicom_params['reso'])
        
    def _load_dicom(self):
        dcmfiles = [file for file in sorted(glob.glob(f"{self.path['dicom']}*.dcm"))]
        df = dcm.read_file(dcmfiles[0])
        self.dicom_params["shape"] = (df.Columns, df.Rows, len(list(dcmfiles)))
        self.dicom_params["reso"] = (float(df.PixelSpacing[1]), float(df.PixelSpacing[0]), float(df.SliceThickness))
        self.dicom_params["protocol_name"] = df.ProtocolName
        #viz.dicom_3d(self.path['dicom'])
      
    def merge_bone(self):
        bone = Segment('bone', np.zeros(self.dicom_params['shape']))
        is_bone = {'radius', 'ulna', 'scaphoid', 'lunate', 'triquetrum',
                   'capitate', 'pisiform', 'trapezium', 'trapezoid'}
        for key in list(self.segments):
            if key in is_bone:
                bone += self.segments[key]
                self.segments.pop(key)
        self.segments['bone'] = bone
        """viz.control_points_3d(volume=self.segments['bone'].array,
                              spacing=self.dicom_params['reso'],
                              radius=3,
                              color='maroon',
                              points=self.control_points,
                              )"""
        #viz.segment_3d(self.segments['bone'].array, self.dicom_params['reso'])
        #xformed = self._homogeneous_transformation(self.segments['bone'].array, self.xform_params['H_L2G'])
        #viz.segment_3d(xformed, self.dicom_params['reso'])
        """viz.coordinate_system_3d(volume=self.segments['bone'].array,
                                 spacing=np.array(self.dicom_params["reso"]),
                                 address=self.control_points["origin"],
                                 vector=np.array(self.xform_params['R']),
                                 norm=50,
                                 color=["red", "blue", "yellow"],
                                 )"""

    def define_coordinate_system(self):
        self._define_control_points()
        self._set_control_point()
        self._calc_rot_mtx()
        self._calc_euler_angle()
        self._calc_trans_mtx()
        self._calc_H_mtx()
        
    def _define_control_points(self):
        seg = self.segments
        
        name = 'radius'; edge = 'proximal'
        idx = seg[name].explore_last_index(edge)
        proximal_radius = seg[name].calc_centroid_2d(idx)
        
        name = 'radius'; edge = 'distal'
        idx = seg[name].explore_last_index(edge)
        distal_radius = seg[name].calc_centroid_2d(idx)
        
        name = 'ulna'; edge = 'proximal'
        idx = seg[name].explore_last_index(edge)
        proximal_ulna = seg[name].calc_centroid_2d(idx)
        
        name = 'ulna'; edge = 'distal'
        idx = seg[name].explore_last_index(edge)
        distal_ulna = seg[name].calc_centroid_2d(idx)
        
        scaphoid_centroid = seg['scaphoid'].calc_centroid_3d()
        lunate_centroid = seg['lunate'].calc_centroid_3d()
        
        radial_styloid = seg['radius'].explore_futherest_index(from_=proximal_ulna)
        radioscaphoid_fossa = 0.5 * (scaphoid_centroid + distal_radius)
        radiolunate_fossa = 0.5 * (lunate_centroid + distal_radius)
        sigmoid_notch = 0.5 * (distal_ulna + distal_radius)
        ridge_between = 0.5 * (radioscaphoid_fossa + radiolunate_fossa)

        origin = proximal_radius + ((distal_ulna[2] - proximal_radius[2]) / (
                ridge_between[2] - proximal_radius[2])) * (ridge_between - proximal_radius)
        
        self.control_points = {"origin": origin,
                               "ridge_between": ridge_between,
                               "sigmoid_notch": sigmoid_notch,
                               "radial_styloid": radial_styloid,
                               'proximal_radius': proximal_radius,
                               }
        
    def _set_control_point(self):
        for key in self.control_points:
            campus = np.zeros(self.dicom_params["shape"], dtype=bool)
            pt = np.round(self.control_points[key]).astype(np.int_)
            campus[pt[0], pt[1], pt[2]] = True
            self.segments[key] = Segment(key, campus)
            
    def _calc_rot_mtx(self):
        pt = unit_conversion(self.dicom_params['reso'], **self.control_points)
        e_zL = (pt["ridge_between"] - pt["origin"]) / np.linalg.norm(pt["ridge_between"] - pt["origin"])
        tmp = np.cross(pt["sigmoid_notch"] - pt["origin"], pt["radial_styloid"] - pt["origin"]) / np.linalg.norm(
            np.cross(pt["sigmoid_notch"] - pt["origin"], pt["radial_styloid"] - pt["origin"]))
        e_xL = np.cross(tmp, e_zL) / np.linalg.norm(np.cross(tmp, e_zL))
        e_yL = np.cross(e_zL, e_xL) / np.linalg.norm(np.cross(e_zL, e_xL))
        
        self.xform_params["R"] = np.array([e_xL, e_yL, e_zL])
        self.xform_params["e_xL"] = np.array(e_xL)
        self.xform_params["e_yL"] = np.array(e_yL)
        self.xform_params["e_zL"] = np.array(e_zL)
     
    def _calc_euler_angle(self):
        """ref: Gregory G. Slabaugh, Computing Euler angles from a rotation matrix"""
        R = self.xform_params["R"]
        if R[2][0] != 1 or R[2][0] != -1:
            theta = np.rad2deg(-np.arcsin(R[2][0]))
            psi = np.rad2deg(np.arctan2(R[2][1] / np.cos(np.deg2rad(theta)), R[2][2] / np.cos(np.deg2rad(theta))))
            phi = np.rad2deg(np.arctan2(R[1][0] / np.cos(np.deg2rad(theta)), R[0][0] / np.cos(np.deg2rad(theta))))
        else:  # gimbal lock
            phi = 0.0
            if R[2][0] == -1:
                theta = 90.0
                psi = phi + np.arctan2(R[0][1], R[0][2])
            else:
                theta = -90.0
                psi = -phi + np.arctan2(-R[0][1], -R[0][2])
        self.xform_params["psi"] = psi
        self.xform_params["theta"] = theta
        self.xform_params["phi"] = phi
        
    def _calc_trans_mtx(self):
        self.xform_params["t"] = (
                np.array(self.control_points["origin"]) - np.array(self.dicom_params["shape"]) // 2)
        
    def _calc_H_mtx(self):
        """ref: https://programming-surgeon.com/script/coordinate-script/"""
        t = self.xform_params["t"]
        R = self.xform_params["R"]
        H_L2G = np.identity(4)
        H_L2G[:3, :3] = np.array(R).T
        H_L2G[:3, 3] = np.array(t).T

        H_G2L = np.identity(4)
        H_G2L[:3, :3] = np.array(R)
        H_G2L[:3, 3] = - np.dot(np.array(R), np.array(t))
        self.xform_params["H_L2G"] = H_L2G
        self.xform_params["H_G2L"] = H_G2L
        
    def _homogeneous_transformation(self, vol, mtx):
        """ref: https://nbviewer.org/gist/lhk/f05ee20b5a826e4c8b9bb3e528348688"""
        pixnum_x, pixnum_y, pixnum_z = vol.shape
        global_points = np.mgrid[-0.5 * pixnum_x:0.5 * pixnum_x,
                                 -0.5 * pixnum_y:0.5 * pixnum_y,
                                 -0.5 * pixnum_z:0.5 * pixnum_z].reshape((3, vol.size))
        global_points = np.vstack([global_points, np.ones(vol.size)])
        local_points = np.dot(mtx, global_points)
        local_points = local_points[:-1] + np.array([0.5 * pixnum_x, 0.5 * pixnum_y, 0.5 * pixnum_z]).reshape([-1, 1])

        x_new, y_new, z_new = local_points.reshape((3, pixnum_x, pixnum_y, pixnum_z), order='C')
        return ndimage.map_coordinates(vol, [x_new, y_new, z_new], order=0, mode="constant", cval=0)
    
    def slice_array(self, outermost):
        idx_z = self._get_idx_z()
        idx_x, idx_y = self._get_idx_xy(outermost, idx_z)
        for key in self.segments:
            self.segments[key] = self.segments[key].slice(idx_x, idx_y, idx_z)
 
    def _get_idx_z(self):
        left, right = -self.SLICE_LENGTH * 0.5, self.SLICE_LENGTH * 0.5
        list_ = (np.arange(self.dicom_params["shape"][2]) - self.control_points["origin"][2]) * \
                          self.dicom_params["reso"][2]
        left = bisect.bisect_left(list_, left)
        right = bisect.bisect_right(list_, right)
        idx_z = (left-1, right)
        return idx_z
    
    def _get_idx_xy(self, outermost, idx_z):
        vol = self.segments[outermost].array[:, :, idx_z[0]: idx_z[1]].T
        y_min = y_max = round(self.control_points["origin"][1])
        x_min = x_max = round(self.control_points["origin"][0])
        for img in vol:
            contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                xi, yi, width_i, height_i = cv2.boundingRect(cnt)
                if xi < x_min:
                    x_min = xi
                if yi < y_min:
                    y_min = yi
                if xi + width_i > x_max:
                    x_max = xi + width_i
                if yi + height_i > y_max:
                    y_max = yi + height_i
        idx_x = [x_min, x_max]
        idx_y = [y_min, y_max]
        return idx_x, idx_y
    
    def upsampling(self):
        mag = (np.array(self.dicom_params["reso"]) / np.array(self.reso))
        for key in self.segments:
            self.segments[key] = self.segments[key].zoom(mag)
            self.shape = self.segments[key].array.shape

    def track_control_points(self):
        for key in self.control_points:
            pt = self.segments[key].track()
            self.control_points[key] = pt
            del self.segments[key]
            
    def split_segment(self, splits):
        for split in splits:
            if split.thickness== 0:
                pass
            elif split.thickness > 0:
                inner, outer = Segment.separate(self.segments[split.before], split)
                self.segments[split.after[0]] = inner
                self.segments[split.after[1]] = outer
                del self.segments[split.before]
            else:
                raise ValueError(f"thickness must be >=0: {split.thickness}")
    
    def save(self, save):
        info = {'dicom': self.dicom_params,
                'xform': self.xform_params,
                'cpts': self.control_points,
                'reso': self.reso,
                'shape': self.shape
                }
        if save:
          with open(f"{self.path['output']}info.json", 'w') as fp:
              json.dump(info, fp, indent=4, cls=ToJsonEncoder)
              print(f'Preprocessing condition is saved in')
              print(f"-> {self.path['output']}info.json")
              print()
          for key in self.segments:
              self.segments[key].save(self.path['output'])
        return self.segments, info
        
class Segment:
    id_ = {'trabecular': 1,
           'cortical': 2,
           'subcutaneus': 3,
           'dermis': 4,
           'endpoint': 5,
           }
    reso = 0
    def __init__(self, name, array):
        self.name = name
        self.array = array
        
    def __add__(self, other):
        self_id = self.id_.get(self.name, 1)
        other_id = other.id_.get(other.name, 1)
        self_array = np.where(self.array != 0, self.array, 0).astype(np.uint8)
        other_array = np.where(other.array != 0, other_id, 0).astype(np.uint8)
        return Segment(other.name, self_array + other_array)
    
    def explore_last_index(self, which_edge=None):
        if which_edge == 'proximal':
            return(np.where(self.array != 0)[2].min())
        elif which_edge == 'distal':
            return(np.where(self.array != 0)[2].max())
        else:
            raise KeyError(which_edge)
        
    def calc_centroid_2d(self, gk):
        img = self.array[:, :, gk]
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            M = cv2.moments(cnt)
            if M["m00"] == 0:
                continue
            gi = int(M["m01"] / M["m00"])
            gj = int(M["m10"] / M["m00"])
            '''#for visualization of centroid
            plt.plot(gi, gj, marker='.')
        plt.imshow(img.T, cmap="gray")
        plt.xlabel("i [px]")
        plt.ylabel("j [px]")
        plt.tick_params(bottom=False)
        plt.show()
        '''
        return np.array([gi, gj, gk])
    
    def calc_centroid_3d(self):
        vol = self.array
        pixnum_i, pixnum_j, pixnum_k = vol.shape
        i = np.arange(pixnum_i).reshape([-1, 1, 1])
        j = np.arange(pixnum_j).reshape([-1, 1])
        k = np.arange(pixnum_k)

        mi = vol * i
        mj = vol * j
        mk = vol * k

        gi = np.sum(mi) / np.sum(vol)
        gj = np.sum(mj) / np.sum(vol)
        gk = np.sum(mk) / np.sum(vol)
        return np.array([gi, gj, gk])
    
    def calc_area(self, idx):
        def __mm2cm(mm):return mm*0.1
        img = self.array[:, :, idx].astype(np.uint8)
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        area = cv2.contourArea(contours[0])
        return area * __mm2cm(self.reso)**2
    
    def explore_futherest_index(self, from_):
        idx = np.array(np.where(self.array != 0))
        distance = np.sqrt(np.sum((idx - from_.reshape([-1, 1]))**2, axis=0))
        futherest_idx = idx[:, np.argmax(distance)]
        return futherest_idx
    
    @classmethod
    def separate(cls, seg, split):
        vol = seg.array
        outer_segment = np.empty_like(vol)
        for i, img in enumerate(vol.T):
            contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            img_campus = cv2.UMat(np.zeros_like(img, dtype=np.uint8))
            cv2.drawContours(img_campus, contours, -1, color=1,
                             thickness=int(split.thickness / cls.reso))
            outer_segment[:, :, i] = img_campus.get().T

        inner_segment = np.multiply(vol, cls._invert_binarization(outer_segment))
        return cls(split.after[0], inner_segment), cls(split.after[1], outer_segment)
    
    @staticmethod
    def _invert_binarization(vol):
        return (~np.copy(vol).astype(bool)).astype("int8")
        
    def slice(self, x, y, z):
        return Segment(self.name, self.array[x[0]:x[1], y[0]:y[1], z[0]:z[1]])
    
    def zoom(self, magnification):
        return Segment(self.name, ndimage.zoom(self.array, magnification, order=0))
        
    def track(self):
        return np.median(np.where(self.array), axis=1).astype(np.int_)
    
    def save(self, path):
        print(f'Segment {self.name} result saved in')
        print(f'--> {path}{self.name}.nrrd')
        print()
        nrrd.write(f'{path}{self.name}.nrrd', self.array)
        
    def add_endpoint(self):
        return Segment(self.name, np.where(self.array==0, self.id_['endpoint'], self.array))

    def synthesize(self, u):
        return Segment(self.name, np.where(self.array==1, u, self.array))
    