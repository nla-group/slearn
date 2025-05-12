import numpy as np
from slearn import symbolicML, slearn
from slearn.symbols import *


if __name__ == "__main__":
    # Generate test time series
    np.random.seed(42)
    t = np.linspace(0, 10, 100)
    ts = np.sin(t) + np.random.normal(0, 0.1, 100)  # Univariate, main test
    ts_multi = np.vstack([np.sin(t), np.cos(t)]).T + np.random.normal(0, 0.1, (100, 2))  # Multivariate
    ts_constant = np.ones(100)  # Edge case: constant series
    t_short = np.linspace(0, 10, 20)
    ts_short = np.sin(t_short)  # Edge case: short series
    
    
    RMSE = list()
    # Main test: standard time series
    print("Testing standard time series (length=100)")
    sax = SAX(window_size=10, alphabet_size=8)
    rmse = test_sax_variant(sax, ts, t, "SAX")
    RMSE.append(rmse)
    if check_list is not None:
        assert check_list[0] == rmse
    
    saxtd = SAXTD(window_size=10, alphabet_size=8)
    rmse = test_sax_variant(saxtd, ts, t, "SAX-TD")
    RMSE.append(rmse)
    if check_list is not None:
        assert check_list[1] == rmse
        
    esax = ESAX(window_size=10, alphabet_size=8)
    rmse = test_sax_variant(esax, ts, t, "eSAX")
    RMSE.append(rmse)
    if check_list is not None:
        assert check_list[2] == rmse
        
    msax = MSAX(window_size=10, alphabet_size=8)
    rmse = test_sax_variant(msax, ts_multi, t, "mSAX", is_multivariate=True)
    RMSE.append(rmse)
    if check_list is not None:
        assert check_list[3] == rmse
        
    asax = ASAX(n_segments=10, alphabet_size=8)
    rmse = test_sax_variant(asax, ts, t, "aSAX")
    RMSE.append(rmse)
    if check_list is not None:
        assert check_list[4] == rmse
        
    # Edge case: constant time series
    print("\nTesting constant time series (length=100)")
    sax = SAX(window_size=10, alphabet_size=8)
    rmse = test_sax_variant(sax, ts_constant, t, "SAX")
    RMSE.append(rmse)
    if check_list is not None:
        assert check_list[5] == rmse
        
    saxtd = SAXTD(window_size=10, alphabet_size=8)
    rmse = test_sax_variant(saxtd, ts_constant, t, "SAX-TD")
    RMSE.append(rmse)
    if check_list is not None:
        assert check_list[6] == rmse
        
    esax = ESAX(window_size=10, alphabet_size=8)
    rmse = test_sax_variant(esax, ts_constant, t, "eSAX")
    RMSE.append(rmse)
    if check_list is not None:
        assert check_list[7] == rmse
        
    asax = ASAX(n_segments=10, alphabet_size=8)
    rmse = test_sax_variant(asax, ts_constant, t, "aSAX")
    RMSE.append(rmse)
    if check_list is not None:
        assert check_list[8] == rmse
        
    # Edge case: short time series
    print("\nTesting short time series (length=20)")
    sax = SAX(window_size=5, alphabet_size=8)
    rmse = test_sax_variant(sax, ts_short, t_short, "SAX")
    RMSE.append(rmse)
    if check_list is not None:
        assert check_list[9] == rmse
        
    saxtd = SAXTD(window_size=5, alphabet_size=8)
    rmse = test_sax_variant(saxtd, ts_short, t_short, "SAX-TD")
    RMSE.append(rmse)
    if check_list is not None:
        assert check_list[10] == rmse
        
    esax = ESAX(window_size=5, alphabet_size=8)
    rmse = test_sax_variant(esax, ts_short, t_short, "eSAX")
    RMSE.append(rmse)
    if check_list is not None:
        assert check_list[11] == rmse
        
    asax = ASAX(n_segments=5, alphabet_size=8)
    rmse = test_sax_variant(asax, ts_short, t_short, "aSAX")
    RMSE.append(rmse)
    if check_list is not None:
        assert check_list[12] == rmse
    
    if check_list is None:
        with open('test_rmse.pkl', 'wb') as f:
            pickle.dump(RMSE, f)
    
    # TEST4
    ts = [np.sin(0.05*i) for i in range(1000)]
    sl = slearn(series=ts, method='SAX',
                ws=5, step=2, 
                n_paa_segments=50, k=20,
                form='numeric', classifier_name="GaussianNB", 
                random_seed=1, verbose=0)
    sl.predict(**params)