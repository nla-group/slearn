import numpy as np
import pickle
from slearn import symbolicML, slearn
from slearn.symbols import *
from slearn.dmetric import *
import pytest

def test_sax_variant(model, ts, t, name, is_multivariate=False):
    symbols = model.fit_transform(ts)
    recon = model.inverse_transform()
    print(f"{name} reconstructed length: {len(recon)}")
    rmse = np.sqrt(np.mean((ts - recon) ** 2))
    return rmse
    
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


def test_distances():
    # Test Levenshtein
    assert levenshtein_distance("cat", "act") == 2  # Two substitutions
    assert levenshtein_distance("cat", "hat") == 1  # One substitution
    assert levenshtein_distance("cat", "cats") == 1  # One insertion
    assert levenshtein_distance("cat", "") == 3  # Deletion
    assert levenshtein_distance("kitten", "sitting") == 3  # Complex case
    
    # Test Hamming
    assert hamming_distance("karolin", "kathrin") == 3  # Three differences
    assert hamming_distance("10110", "11110") == 1  # One difference
    try:
        hamming_distance("cat", "cats")  # Should raise ValueError
        assert False, "Hamming should raise ValueError for unequal lengths"
    except ValueError:
        pass
    
    # Test Jaro
    assert abs(jaro_similarity("martha", "marhta") - 0.944) < 0.001  # Standard example
    assert jaro_similarity("cat", "") == 0.0  # Empty string
    assert jaro_similarity("same", "same") == 1.0  # Identical strings
    
    # Test Jaro-Winkler
    assert abs(jaro_winkler_distance("martha", "marhta") - 0.961) < 0.001  # Standard example
    assert jaro_winkler_distance("dixon", "dicksonx") > 0.8  # Prefix match
    assert jaro_winkler_distance("cat", "") == 0.0  # Empty string
    
    # Test Cosine Similarity (Word-Based)
    assert abs(cosine_similarity("cat hat", "hat cat") - 1.0) < 0.001  # Same words
    assert cosine_similarity("cat", "dog") == 0.0  # No shared words
    assert cosine_similarity("", "dog") == 0.0  # Empty string
    
    # Test Cosine Bigram Similarity
    assert abs(cosine_bigram_similarity("cat", "cap") - 0.5) < 0.001  # Shared bigram "ca"
    assert cosine_bigram_similarity("cat", "act") == 0.0  # No shared bigrams
    assert cosine_bigram_similarity("cat", "dog") < 0.1  # Few shared bigrams
    assert cosine_bigram_similarity("", "dog") == 0.0  # Empty string
    
    # Test LCS Distance
    assert lcs_distance("kitten", "sitting") == 5  # LCS = "ittn"
    assert lcs_distance("cat", "act") == 2  # LCS = "at" or "ct"
    assert lcs_distance("cat", "") == 3  # Empty string
    
    
    # Test Dice’s Coefficient
    assert abs(dice_coefficient("night", "nacht") - 0.25) < 0.001  # One shared bigram
    assert dice_coefficient("cat", "cat") == 1.0  # Identical
    assert dice_coefficient("cat", "") == 0.0  # Empty string
    
    # Test Smith-Waterman
    assert smith_waterman_distance("kitten", "sitting") < 0  # Negative due to inverse scoring
    assert smith_waterman_distance("cat", "act") < 0  # Local alignment
    assert smith_waterman_distance("cat", "") == 0  # No alignment possible
    
    
    # Test Damerau-Levenshtein
    assert damerau_levenshtein_distance("cat", "act") == 1  # Transposition
    assert damerau_levenshtein_distance("cat", "hat") == 1  # Substitution
    assert damerau_levenshtein_distance("cat", "cats") == 1  # Insertion
    # Test Damerau-Levenshtein
    assert damerau_levenshtein_distance("cat", "act") == 1  # Transposition
    assert damerau_levenshtein_distance("cat", "hat") == 1  # Substitution
    assert damerau_levenshtein_distance("cat", "cats") == 1  # Insertion
    assert damerau_levenshtein_distance("cat", "") == 3  # Deletion
    assert damerau_levenshtein_distance("kitten", "sitting") == 3  # Complex case
    
    # Test Jaro-Winkler
    assert abs(jaro_winkler_distance("martha", "marhta") - 0.961) < 0.001  # Common example
    assert jaro_winkler_distance("dixon", "dicksonx") > 0.8  # Prefix match
    assert jaro_winkler_distance("cat", "") == 0.0  # Empty string
    assert jaro_winkler_distance("same", "same") == 1.0  # Identical strings
    assert jaro_winkler_distance("abc", "xyz") < 0.1  # No similarity
    
    print("All tests passed!")



def test_code(setup_data):
    t, ts, ts_multi, ts_constant, ts_short = setup_data
    RMSE = []

    # Run standard tests (e.g., main series tests)
    RMSE.extend(run_standard_tests(t, ts, ts_multi, ts_constant, ts_short))

    # Run edge case tests (e.g., constant and short series)
    RMSE.extend(run_edge_case_tests(t, ts_constant, ts_short))
    
    # Save results
    save_results(RMSE)

    # Test distance
    test_distances()

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
