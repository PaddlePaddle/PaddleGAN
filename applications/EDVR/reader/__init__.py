from .reader_utils import regist_reader, get_reader
#from .feature_reader import FeatureReader
#from .kinetics_reader import KineticsReader
#from .nonlocal_reader import NonlocalReader
#from .ctcn_reader import CTCNReader
#from .bmn_reader import BMNReader
#from .bsn_reader import BSNVideoReader
#from .bsn_reader import BSNProposalReader
#from .ets_reader import ETSReader
#from .tall_reader import TALLReader
from .edvr_reader import EDVRReader

# regist reader, sort by alphabet
#regist_reader("ATTENTIONCLUSTER", FeatureReader)
#regist_reader("ATTENTIONLSTM", FeatureReader)
#regist_reader("NEXTVLAD", FeatureReader)
#regist_reader("NONLOCAL", NonlocalReader)
#regist_reader("TSM", KineticsReader)
#regist_reader("TSN", KineticsReader)
#regist_reader("STNET", KineticsReader)
#regist_reader("CTCN", CTCNReader)
#regist_reader("BMN", BMNReader)
#regist_reader("BSNTEM", BSNVideoReader)
#regist_reader("BSNPEM", BSNProposalReader)
#regist_reader("ETS", ETSReader)
#regist_reader("TALL", TALLReader)
regist_reader("EDVR", EDVRReader)
