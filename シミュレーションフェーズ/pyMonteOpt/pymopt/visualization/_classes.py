import pyvista as pv
import numpy as np
import cupy as cp
import seaborn as sns
import matplotlib.pyplot as plt
from ..utils.utilities import set_params


class Visualization:
    def __init__(self, volume=None, point_clouds=None):
        self.volume = volume
        self.point_clouds = point_clouds
        self.params = {"origin": [0, 0, 0],
                       "spacing": [1, 1, 1],
                       "color": None,
                       "cmap": None,
                       "opacity": "linear",
                       "clim": None
                       }
        self.key_params = list(self.params.keys())
        self.plotter = None

    def set_params(self, **kwargs):
        set_params(self.params, **kwargs)

    def plot_voxel_3d(self):
        grid = pv.ImageData()
        grid.dimensions = np.array(self.volume.shape) + 1
        grid.origin = self.params["origin"]
        grid.spacing = self.params["spacing"]
        grid["data"] = self.volume.ravel(order='F')

        self.plotter = pv.Plotter()
        self.plotter.add_volume(volume=grid,
                                opacity=self.params["opacity"],
                                cmap=self.params["cmap"],
                                clim=self.params["clim"]
                                )

    def plot_point_clouds_3d(self):
        point_clouds = pv.PolyData(self.point_clouds)
        self.plotter = pv.Plotter()
        self.plotter.add_mesh(point_clouds,
                              color=self.params["color"],
                              render_points_as_spheres=True,
                              )

    def add_arrows(self, center, direction, color):
        self.plotter.add_arrows(cent=center,
                                direction=direction,
                                color=color,
                                )
        
    def add_sphere(self, radius, color, **kwargs):
        for key in kwargs:
            sphere = pv.Sphere(radius=radius,
                               center=kwargs[key],
                               )
            self.plotter.add_mesh(sphere,
                                  color=color,
                                  )
            self.plotter.add_point_labels(points=kwargs[key],
                                          labels=[key],
                                          font_size=30,
                                          show_points=False,
                                          shape_opacity=0,
                                          )
        
    def add_coordinate_system(self, center, direction, color):
        self.plotter.add_arrows(cent=center,
                                direction=direction[0],
                                color=color[0],
                                )
        self.plotter.add_arrows(cent=center,
                                direction=direction[1],
                                color=color[1],
                                )
        self.plotter.add_arrows(cent=center,
                                direction=direction[2],
                                color=color[2],
                                )
        self.plotter.add_point_labels(points=center + direction,
                                      labels=['x', 'y', 'z'],
                                      font_size=30,
                                      show_points=False,
                                      shape_opacity=0,
                                      )

    def show_plot(self):
        self.plotter.show_grid()
        self.plotter.show()
        
    
def dicom_3d(path):
    reader = pv.DICOMReader(path)
    data = reader.read()
    data.plot(volume=True, show_scalar_bar=False, show_bounds=True, cmap='gray')

def segment_3d(volume, spacing):
    data = np.where(volume != 0, 255, volume)
    params = {'spacing': spacing,
              'cmap': 'summer',
              }
    viz = Visualization(volume=data)
    viz.set_params(**params)
    viz.plot_voxel_3d()
    viz.show_plot()
    
def control_points_3d(volume, spacing, radius, color, points):
    data = np.where(volume != 0, 25, volume)
    params = {'spacing': spacing,
              'cmap': 'summer',
              }
    points = {key: spacing*points[key] for key in points}
    viz = Visualization(volume=data)
    viz.set_params(**params)
    viz.plot_voxel_3d()
    viz.add_sphere(radius, color, **points)
    viz.show_plot()
    
def coordinate_system_3d(volume, spacing, address, vector, norm, color):
    data = np.where(volume != 0, 25, volume)
    center = spacing * address
    direction = norm * vector
    params = {"spacing": spacing,
              "opacity": "linear",
              "cmap": "summer",
              }
    viz = Visualization(volume=data)
    viz.set_params(**params)
    viz.plot_voxel_3d()
    viz.add_coordinate_system(center=center, direction=direction, color=color)
    viz.show_plot()
    
def laser_position_3d(volume, end_point, spacing, address, vector, norm, color):
    data = np.where(volume == end_point, np.nan, volume)
    center = spacing * (address - norm * vector)
    direction = spacing * (norm * vector)
    params = {"spacing": [spacing, spacing, spacing],
              "opacity": "linear",
              "cmap": ['black', 'gray', 'white', 'salmon', 'bisque'],
              }
    viz = Visualization(volume=data)
    viz.set_params(**params)
    viz.plot_voxel_3d()
    viz.add_arrows(center=center, direction=direction, color=color)
    viz.show_plot()
    
    
def model_3d(volume, end_point, spacing):
    data = np.where(volume == end_point, np.nan, volume)
    params = {"spacing": [spacing, spacing, spacing],
              "opacity": "linear",
              "cmap": ['black', 'gray', 'white', 'salmon', 'bisque'],
              }
    viz = Visualization(volume=data)
    viz.set_params(**params)
    viz.plot_voxel_3d()
    viz.show_plot()
    
def point_clouds_3d(point_clouds):
    data = point_clouds
    params = {"color": "maroon",
              }
    viz = Visualization(point_clouds=data)
    viz.set_params(**params)
    viz.plot_point_clouds_3d()
    viz.show_plot()
    
def light_intensity_map_2d(Rdr, *, x, y, step=10):
    xticks = [i for i in range(x.shape[0]) if i%step==0]
    yticks = [j for j in range(y.shape[0]) if j%step==0]
    sns.set(font_scale=1.8)
    fig, ax = plt.subplots(figsize=(20, 3))
    sns.heatmap(Rdr, square=True, cmap='jet')
    ax.set_xlabel(r'$\theta[^\circ]$')
    ax.set_ylabel('y[mm]')
    ax.set_xticks(xticks)
    ax.set_xticklabels(np.round(x[xticks]))
    ax.set_yticks(yticks)
    ax.set_yticklabels(np.round(y[yticks]))
    plt.show()

def light_intensity_map_3d(p, w, const=5):
    pts = np.round(p.T).astype(int)
    pts_min, pts_max = pts.min(0), pts.max(0)
    data = np.zeros(pts_max - pts_min + 1)
    for uniq in np.unique(pts, axis=0):
        idx = np.where((pts[:, 0]==uniq[0]) & (pts[:, 1]==uniq[1]) & (pts[:, 2]==uniq[2]))
        i, j, k = (uniq - pts_min)
        data[i, j, k] = np.log(w[idx[0]].sum() + const)
    params = {"cmap": 'jet',
              }
    viz = Visualization(volume=data)
    viz.set_params(**params)
    viz.plot_voxel_3d()
    viz.show_plot()