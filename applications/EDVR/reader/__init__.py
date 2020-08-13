from .reader_utils import regist_reader, get_reader
from .edvr_reader import EDVRReader

regist_reader("EDVR", EDVRReader)
