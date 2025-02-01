# src/algorithms/__init__.py

from .base_algorithm import IterativeAlgorithm
from .extragradient import Extragradient
from .solodov_svaiter import SolodovSvaiter
from .eag_c import EAGC
from .graal import GRAAL
from .agraal import AGRAAL
from .curvature_eg import CurvatureEG
from .eg_aa1 import EGAA
from .fista import FISTA
from .adaptive_proximal_gradient import AdaptiveProximalGradient 
from .flex import FLEX
from .iflex import IFLEX
from .proxflex import ProxFLEX
from .aa_type_i import AAI
from .aa_type_ii import AAII
from .broyden import Broyden
from .jsymmetric import Jsymmetric

__all__ = [
    'IterativeAlgorithm',
    'Extragradient',
    'SolodovSvaiter',
    'EAGC',
    'GRAAL',
    'AGRAAL',
    'CurvatureEG',
    'EGAA',
    'FISTA',
    'AdaptiveProximalGradient',
    'FLEX',
    'IFLEX',
    'ProxFLEX',
    'AAI',
    'AAII',
    'Broyden',
    'Jsymmetric'
]
