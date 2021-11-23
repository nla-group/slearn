import numpy as np
from slearn.slearn import symbolicML, slearn, SAX

ts = np.random.normal(size = 500)
ts = np.cumsum(ts)
ts = ts - np.mean(ts)
ts /= np.std(ts, ddof=1)

n_sax_symbols = 8
n_paa_segments = 10

# TEST1
width = len(ts) // n_paa_segments
sax = SAX(w = width, k = n_sax_symbols)
sax_ts = sax.transform(ts)
recon_ts = sax.inverse_transform(sax_ts)

# TEST2
sax = SAX(n_paa_segments=n_paa_segments, k = n_sax_symbols)
sax_ts = sax.transform(ts)
recon_ts = sax.inverse_transform(sax_ts)

# TEST3
ts = [np.sin(0.05*i) for i in range(1000)]
sl = slearn(series=ts, method='ABBA', 
            gap=10, step=1, tol=0.5,  alpha=0.5,
            form='numeric', classifier_name="GaussianNB",
            random_seed=1, verbose=0)
params = {'var_smoothing':0.001}
sl.predict(**params)

# TEST4
ts = [np.sin(0.05*i) for i in range(1000)]
sl = slearn(series=ts, method='SAX',
            gap=5, step=2, 
            n_paa_segments=50, k=20,
            form='numeric', classifier_name="GaussianNB", 
            random_seed=1, verbose=0)
sl.predict(**params)
