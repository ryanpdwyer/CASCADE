import os
from os import path
import pandas as pd
import numpy as np
from rdkit import Chem
from nfp.preprocessing import MolAPreprocessor, GraphSequence


from keras.layers import Input

from keras.models import load_model

from nfp.layers import (Squeeze, EdgeNetwork,
                               ReduceBondToPro, ReduceBondToAtom, GatherAtomToBond, ReduceAtomToPro)
from nfp.models import GraphModel

_dir = path.dirname(path.abspath(__file__))

_modelpath_C = os.path.join(_dir, 'trained_model', 'best_model.hdf5')
_modelpath_H = os.path.join(_dir, 'trained_model', 'best_model_H_DFTNN.hdf5')

NMR_model_C = load_model(_modelpath_C, custom_objects={'GraphModel': GraphModel,
                                             'ReduceAtomToPro': ReduceAtomToPro,
                                             'Squeeze': Squeeze,
                                             'GatherAtomToBond': GatherAtomToBond,
                                             'ReduceBondToAtom': ReduceBondToAtom})
NMR_model_H = load_model(_modelpath_H, custom_objects={'GraphModel': GraphModel,
                                             'ReduceAtomToPro': ReduceAtomToPro,
                                             'Squeeze': Squeeze,
                                             'GatherAtomToBond': GatherAtomToBond,
                                             'ReduceBondToAtom': ReduceBondToAtom})