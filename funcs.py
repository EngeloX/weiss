import numpy as np
import pandas as pd


def reduce_mem_usage(data, use_float16=False, verbose=False):
    """
    Function reduces memory usage of the dataset.
    
    
    Params
    ------
    data: pd.DataFrame
    use_float16: bool, default = False
        тип float16 может привести к потери точности из за его размера в 2 байта
    verbose: bool, default = False
    """
    mem_before = data.memory_usage().sum() / 1024**2
        
    for col in data.columns:
        col_type = data[col].dtype
        
        if str(col_type)[:3] == 'int':
            c_min = data[col].min()
            c_max = data[col].max()
            
            if np.iinfo(np.int8).min <= c_min <= c_max <= np.iinfo(np.int8).max:
                data[col] = data[col].astype(np.int8)
            elif np.iinfo(np.int16).min <= c_min <= c_max <= np.iinfo(np.int16).max:
                data[col] = data[col].astype(np.int16)
            elif np.iinfo(np.int32).min <= c_min <= c_max <= np.iinfo(np.int32).max:
                data[col] = data[col].astype(np.int32)
            elif np.iinfo(np.int64).min <= c_min <= c_max <= np.iinfo(np.int64).max:
                data[col] = data[col].astype(np.int64)
        

        elif str(col_type)[:5] == 'float':
            c_min = data[col].min()
            c_max = data[col].max()
            
            if use_float16 and np.finfo(np.float16).min <= c_min <= c_max <= np.finfo(np.float16).max:
                data[col] = data[col].astype(np.float16)
            elif np.finfo(np.float32).min <= c_min <= c_max <= np.finfo(np.float32).max:
                data[col] = data[col].astype(np.float32)
            elif np.finfo(np.float64).min <= c_min <= c_max <= np.finfo(np.float64).max:
                data[col] = data[col].astype(np.float64)

                
    mem_after = data.memory_usage().sum() / 1024**2
    if verbose:
        print(f"Memory usage of the DataFrame before optimization: {mem_before:.2f} MB")
        print(f"Memory usage of the DataFrame after optimization: {mem_after:.2f} MB")
        print(f"Decreased by {(100 * (mem_before - mem_after) / mem_before):.1f}%")
            
    return data


# __________________________________________________________________________________________________

def generate_bootstrap(X, y, random_state=None, verbose=False):
    """
    Функция генерирует бутстрапированную выборку
    
    Params:
    -------
    X: pd.DataFrame
        Набор независимых переменных
    y: pd.Series
        Зависимая переменная
    random_state: int, default = None
        Фиксация случайности для воспроизводимости
    verbose: bool, default = False
        Отображение индексов попавших в бутстрап-выборку
    Return:
    -------
    
    """
    rng = np.random.RandomState(random_state)
    sample_idx = np.arange(len(X))
    bootstrap_idx = rng.choice(sample_idx, size=len(sample_idx))
    X_boot = X.iloc[bootstrap_idx]
    y_boot = y.iloc[bootstrap_idx]

    X_out_boot = X.iloc[~X.index.isin(X_boot.index)]
    y_out_boot = y.iloc[~X.index.isin(X_boot.index)]
    
    if verbose:
        print(f'Bootstrap-индексы: {X_boot.index.to_list()}')
        print(f'Out of Bag-индексы: {X_out_boot.index.to_list()}')
        
    return X_boot, y_boot, X_out_boot, y_out_boot