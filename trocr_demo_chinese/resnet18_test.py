default='D:/trocr-data/02 SCUT_EPT_DATA/02 SCUT_EPT_DATA/test_jpg/*.jpg'
from glob import glob
path=glob(default)
print(path[50][58:])