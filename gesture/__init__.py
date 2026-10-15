"""手势控制电脑 · 本地核心库。

导入本包时会先压低 MediaPipe / absl 的日志噪音，
因此后续模块可以放心地在任何位置 import mediapipe。
"""

import os

os.environ.setdefault("GLOG_minloglevel", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("ABSL_MIN_LOG_LEVEL", "2")
