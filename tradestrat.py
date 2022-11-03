# Imports

import numpy as np
import pandas as pd

from numpy.typing import NDArray
from collections.abc import Callable

import abc


class Strategy(abc.ABC):
    
    def __init__(self, data: list[str] | dict[str, pd.DataFrame]) -> None:
        ''' 
        Initialize Strategy class
        
        Args:
            data: list of tickers to be considered in universe OR 
                  dictionary of DataFrames, each containing dates along rows and tickers along columns, 
                  with one DataFrame per value (e.g. data = {'price': ..., 'PE': ...})
            
        Return:
            None
        '''
        
        # TODO
        ...
        
    @abc.abstractmethod
    def get_weights(self) -> pd.DataFrame:
        '''
        Get strategy weights over time
        
        Args:
            ...
            
        Return:
            DataFrame containing dates along rows and tickers along columns, with values being the strategy weights
        
        '''
        
        ...


        
        