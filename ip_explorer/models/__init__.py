from .schnet import SchNetModelWrapper
from .nequip import NequIPModelWrapper
from .ace import ACEModelWrapper
from .mace import MACEModelWrapper
from .valle_oganov import ValleOganovModelWrapper
from .vgop import VGOPModelWrapper
from .test import TestModelWrapper
from .soap import SOAPModelWrapper

implemented_wrappers = {
    'test':         TestModelWrapper,
    'schnet':       SchNetModelWrapper,
    'painn':        SchNetModelWrapper,
    'nequip':       NequIPModelWrapper,
    'allegro':      NequIPModelWrapper,
    'ace':          ACEModelWrapper,
    'mace':         MACEModelWrapper,
    'valle-oganov': ValleOganovModelWrapper,
    'vgop':         VGOPModelWrapper,
    'soap':         SOAPModelWrapper,
}

def get_model_wrapper(model_type):
    global implemented_wrappers

    if model_type not in implemented_wrappers:
        raise RuntimeError("Wrapper for model type '{}' has not been implemented.".format(model_type))

    return implemented_wrappers[model_type]
