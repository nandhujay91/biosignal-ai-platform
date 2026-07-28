import numpy as np

SIGNAL_INFO = {
    "Aux": {
        "dtype": np.int16,
        "channels": 3,
        "sampling_rate": 32,
        "window_duration": 5,
        "overlap": 0.5,
        "normalization": "z_score",
        "filter": {
            "enabled": False,
            "type": None,
            "lowcut": None,
            "highcut": None,
            "order": None,
        },
    },
    "IMU": {
        "dtype": np.int16,
        "channels": 9,
        "sampling_rate": 64,
        "window_duration": 5,
        "overlap": 0.5,
        "normalization": "z_score",
        "filter": {
            "enabled": False,
            "type": None,
            "lowcut": None,
            "highcut": None,
            "order": None,
        },
    },
    "Ephy": {
        "dtype": np.int16,
        "channels": 8,
        "sampling_rate": 256,
        "window_duration": 5,
        "overlap": 0.5,
        "normalization": "z_score",
        "filter": {
            "enabled": True,
            "type": "butter_bandpass",
            "lowcut": 0.5,
            "highcut": 40.0,
            "order": 4,
        },
    },
    "Oxym": {
        "dtype": np.int32,
        "channels": 2,
        "sampling_rate": 128,
        "window_duration": 5,
        "overlap": 0.5,
        "normalization": "z_score",
        "filter": {
            "enabled": False,
            "type": None,
            "lowcut": None,
            "highcut": None,
            "order": None,
        },
    },
}