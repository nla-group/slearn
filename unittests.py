import numpy as np
import pickle
from slearn import symbolicML, slearn
from slearn.symbols import *
import pytest

def test_sax_variant_with_rmse(test_function, ts, t, name, is_multivariate=False):
    rmse = test_sax_variant(test_function, ts, t, name, is_multivariate=is_multivariate)
    return rmse

def run_standard_tests(t, ts, ts_multi, ts_constant, ts_short):
    RMSE = []

    # Main test: standard time series
    print("Testing standard time series (length=100)")

    # SAX
    sax = SAX(window_size=10, alphabet_size=8)
    rmse = test_sax_variant_with_rmse(sax, ts, t, "SAX")
    RMSE.append(rmse)

    # SAX-TD
    saxtd = SAXTD(window_size=10, alphabet_size=8)
    rmse = test_sax_variant_with_rmse(saxtd, ts, t, "SAX-TD")
    RMSE.append(rmse)

    # eSAX
    esax = ESAX(window_size=10, alphabet_size=8)
    rmse = test_sax_variant_with_rmse(esax, ts, t, "eSAX")
    RMSE.append(rmse)

    # mSAX (multivariate)
    msax = MSAX(window_size=10, alphabet_size=8)
    rmse = test_sax_variant_with_rmse(msax, ts_multi, t, "mSAX", is_multivariate=True)
    RMSE.append(rmse)

    # aSAX
    asax = ASAX(n_segments=10, alphabet_size=8)
    rmse = test_sax_variant_with_rmse(asax, ts, t, "aSAX")
    RMSE.append(rmse)

    return RMSE

def run_edge_case_tests(t, ts_constant, ts_short):
    RMSE = []

    # Constant time series
    print("\nTesting constant time series (length=100)")

    # SAX
    sax = SAX(window_size=10, alphabet_size=8)
    rmse = test_sax_variant_with_rmse(sax, ts_constant, t, "SAX")
    RMSE.append(rmse)

    # SAX-TD
    saxtd = SAXTD(window_size=10, alphabet_size=8)
    rmse = test_sax_variant_with_rmse(saxtd, ts_constant, t, "SAX-TD")
    RMSE.append(rmse)

    # eSAX
    esax = ESAX(window_size=10, alphabet_size=8)
    rmse = test_sax_variant_with_rmse(esax, ts_constant, t, "eSAX")
    RMSE.append(rmse)

    # aSAX
    asax = ASAX(n_segments=10, alphabet_size=8)
    rmse = test_sax_variant_with_rmse(asax, ts_constant, t, "aSAX")
    RMSE.append(rmse)

    # Short time series
    print("\nTesting short time series (length=20)")

    # SAX
    sax = SAX(window_size=5, alphabet_size=8)
    rmse = test_sax_variant_with_rmse(sax, ts_short, t, "SAX")
    RMSE.append(rmse)

    # SAX-TD
    saxtd = SAXTD(window_size=5, alphabet_size=8)
    rmse = test_sax_variant_with_rmse(saxtd, ts_short, t, "SAX-TD")
    RMSE.append(rmse)

    # eSAX
    esax = ESAX(window_size=5, alphabet_size=8)
    rmse = test_sax_variant_with_rmse(esax, ts_short, t, "eSAX")
    RMSE.append(rmse)

    # aSAX
    asax = ASAX(n_segments=5, alphabet_size=8)
    rmse = test_sax_variant_with_rmse(asax, ts_short, t, "aSAX")
    RMSE.append(rmse)

    return RMSE

def save_results(RMSE):
    # Save RMSE values for later analysis
    with open('test_rmse.pkl', 'wb') as f:
        pickle.dump(RMSE, f)

@pytest.fixture
def setup_data():
    np.random.seed(42)
    t = np.linspace(0, 10, 100)
    ts = np.sin(t) + np.random.normal(0, 0.1, 100)  # Univariate, main test
    ts_multi = np.vstack([np.sin(t), np.cos(t)]).T + np.random.normal(0, 0.1, (100, 2))  # Multivariate
    ts_constant = np.ones(100)  # Edge case: constant series
    t_short = np.linspace(0, 10, 20)
    ts_short = np.sin(t_short)  # Edge case: short series
    
    return t, ts, ts_multi, ts_constant, ts_short

def test_code(setup_data):
    t, ts, ts_multi, ts_constant, ts_short = setup_data
    RMSE = []

    # Run standard tests (e.g., main series tests)
    RMSE.extend(run_standard_tests(t, ts, ts_multi, ts_constant, ts_short))

    # Run edge case tests (e.g., constant and short series)
    RMSE.extend(run_edge_case_tests(t, ts_constant, ts_short))

    # Save results
    save_results(RMSE)

    # TEST4 (additional test)
    ts = [np.sin(0.05 * i) for i in range(1000)]
    sl = slearn(series=ts, method='SAX',
                ws=5, step=2, 
                n_paa_segments=50, k=20,
                form='numeric', classifier_name="GaussianNB", 
                random_seed=1, verbose=0)
    sl.predict(**params)

# Running the tests with pytest
if __name__ == "__main__":
    test_code(setup_data)
