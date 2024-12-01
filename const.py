import os

# 获取项目根路径
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIRNAME = "data"  # 数据集放在根目录的data文件夹下

# 确定文件名正确
RT_MATRIX_NAME = "rtMatrix.txt"
TP_MATRIX_NAME = "tpMatrix.txt"
USERLIST_NAME = "userlist.txt"
WSLIST_NAME = "wslist.txt"

DATASET_DIR = os.path.join(BASE_DIR, DATASET_DIRNAME)

WSDREAM_1_DIR = os.path.join(DATASET_DIR, "dataset#1")
WSDREAM_2_DIR = os.path.join(DATASET_DIR, "wsdream_2")

WSDREAM_1_RT_MATRIX = os.path.join(WSDREAM_1_DIR, RT_MATRIX_NAME)
WSDREAM_1_TP_MATRIX = os.path.join(WSDREAM_1_DIR, TP_MATRIX_NAME)
WSDREAM_1_USERLIST = os.path.join(WSDREAM_1_DIR, USERLIST_NAME)
WSDREAM_1_WSLIST = os.path.join(WSDREAM_1_DIR, WSLIST_NAME)

# outlier
OULIER_NAME = "Full_I_outlier_rt.npy"
WSDREAM_1_OULIER = os.path.join(WSDREAM_1_DIR, OULIER_NAME)

# 输出路径
OUTPUT_DIRNAME = "output"

OUTPUT_DIR = os.path.join(BASE_DIR, OUTPUT_DIRNAME)

__all__ = ["WSDREAM_1_RT_MATRIX", "WSDREAM_1_TP_MATRIX", "WSDREAM_1_USERLIST", "WSDREAM_1_WSLIST", "OUTPUT_DIR", "WSDREAM_1_OULIER"]


# 数据集头文件
user_headers = [
    "User ID",
    "IP Address",
    "Country",
    "IP No.",
    "AS",
    "Latitude",
    "Longitude",
]
service_headers = [
    "Service ID",
    "WSDL Address",
    "Service Provider",
    "IP Address",
    "Country",
    "IP No.",
    "AS",
    "Latitude",
    "Longitude",
]
