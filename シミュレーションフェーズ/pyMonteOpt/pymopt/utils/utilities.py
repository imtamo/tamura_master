import json
import numpy as np

def set_params(data, **kwargs):
    keys = set(data.keys())
    for key in kwargs:
        if not key in keys:
            raise KeyError(key)
        data[key] = kwargs[key]
        
def unit_conversion(reso, **kwargs):
    return {key: np.array(kwargs[key]) * np.array(reso) for key in kwargs}

def calTime(end, start):
    elapsed_time = end - start
    q, mod = divmod(elapsed_time, 60)
    if q < 60:
        print('Calculation time: %d minutes %0.3f seconds.' % (q, mod))
    else:
        q2, mod2 = divmod(q, 60)
        print('Calculation time: %d h %0.3f minutes.' % (q2, mod2))

class ToJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)
