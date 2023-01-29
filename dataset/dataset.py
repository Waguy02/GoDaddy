import json
import os
from unicodedata import category
import numpy as np
import pandas as pd
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from torchvision.transforms import transforms
from tqdm import tqdm
from constants import DATA_DIR


from enum import Enum
class DatasetType(Enum):
    TRAIN="train"
    VALID="valid"
    TEST="test"
