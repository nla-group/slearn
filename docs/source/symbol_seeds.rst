LZW String Library
==================

Overview
--------

This module provides tools for generating and manipulating strings with controlled properties using the Lempel-Ziv-Welch (LZW) compression algorithm. It is designed for applications such as symbolic time series analysis, where strings with specific numbers of unique symbols and LZW complexities are needed. The primary function, ``lzw_string_seeds``, generates a library of strings and stores them in a pandas DataFrame, with options to save to a CSV file. Supporting functions handle symbol dictionary creation, LZW compression, decompression, string reduction, and individual string generation.

Dependencies
------------

- Python 3.6+
- NumPy
- pandas
- tqdm (optional, for progress tracking in ``lzw_string_seeds``)

Functions
---------

.. function:: _symbols(n=52)

   Creates a dictionary mapping alphabetical characters (A-z) to numerical codes, limited to a maximum of 52 symbols.

   :param n: Number of symbols in the dictionary (max 52). Default: 52.
   :type n: int
   :return: Dictionary mapping characters to integer codes (e.g., ``{'A': 0, 'B': 1, ..., 'z': 51}``).
   :rtype: dict

   **Example**:

   .. code-block:: python

      >>> from lzw_string_library import _symbols
      >>> _symbols(3)
      {'A': 0, 'B': 1, 'C': 2}

.. function:: lzwcompress(uncompressed)

   Compresses a string using the LZW algorithm, restricted to alphabetical characters (A-z). Adapted from `Rosetta Code LZW Compression <https://rosettacode.org/wiki/LZW_compression#Python>`_.

   :param uncompressed: String to compress, containing only alphabetical characters.
   :type uncompressed: str
   :return: List of integer codes representing the compressed string.
   :rtype: list of int

   **Example**:

   .. code-block:: python

      >>> from lzw_string_library import lzwcompress
      >>> lzwcompress("AABAB")
      [0, 0, 1, 52]

.. function:: lzwdecompress(compressed)

   Decompresses a list of LZW integer codes back to a string.

   :param compressed: List of integer codes from LZW compression.
   :type compressed: list of int
   :return: Decompressed string.
   :rtype: str
   :raises ValueError: If an invalid code is encountered.

   **Example**:

   .. code-block:: python

      >>> from lzw_string_library import lzwdecompress
      >>> lzwdecompress([0, 0, 1, 52])
      'AABAB'

.. function:: reduce(s)

   Reduces a string to its shortest periodic substring (e.g., "ABABAB" reduces to "AB").

   :param s: String to reduce.
   :type s: str
   :return: Shortest periodic substring or the original string if no reduction is possible.
   :rtype: str

   **Example**:

   .. code-block:: python

      >>> from lzw_string_library import reduce
      >>> reduce("ABABAB")
      'AB'
      >>> reduce("ABC")
      'ABC'

.. function:: lzw_string_generator(nr_symbols, target_complexity, priorise_complexity=True, random_state=42)

   Generates a string with a specified number of unique symbols and target LZW complexity. If ``priorise_complexity=True``, stops when the target complexity is reached; otherwise, continues until the specified number of symbols is used.

   :param nr_symbols: Number of unique symbols to use (max 52).
   :type nr_symbols: int
   :param target_complexity: Target LZW complexity (number of unique substrings in the LZW dictionary).
   :type target_complexity: int
   :param priorise_complexity: If True, prioritizes target complexity; if False, prioritizes using all specified symbols. Default: True.
   :type priorise_complexity: bool
   :param random_state: Seed for random number generation.
   :type random_state: int
   :return: Tuple of the generated string and its LZW complexity. Returns ``(np.nan, 0)`` if ``nr_symbols > target_complexity``.
   :rtype: tuple (str or np.nan, int)
   :raises Warning: If ``nr_symbols > 52`` (capped at 52) or if ``nr_symbols=1`` and ``target_complexity>1`` (returns ``("A", 1)``).

   .. note:: The LZW complexity is computed after reducing the string with ``reduce`` and applying ``lzwcompress``.

   **Example**:

   .. code-block:: python

      >>> from lzw_string_library import lzw_string_generator
      >>> str_, str_complex = lzw_string_generator(2, 3, priorise_complexity=True, random_state=2)
      >>> print(f"string: {str_}, complexity: {str_complex}")
      string: BAA, complexity: 3
      >>> str_, str_complex = lzw_string_generator(2, 3, priorise_complexity=False, random_state=2)
      >>> print(f"string: {str_}, complexity: {str_complex}")
      string: BAB, complexity: 3

.. function:: lzw_string_seeds(symbols=(1,10,5), complexity=(5,25,5), symbols_range_distribution=None, complexity_range_distribution=None, iterations=1, save_csv=False, priorise_complexity=True, random_state=42)

   Generates a library of strings with specified ranges of unique symbols and LZW complexities, stored in a pandas DataFrame. Optionally saves the results to a CSV file.

   :param symbols: Number of unique symbols. Can be an integer, a tuple of (start, stop, [step]), or a list of values. Default: (1, 10, 5).
   :type symbols: int or array-like
   :param complexity: Target LZW complexity. Can be an integer, a tuple of (start, stop, [step]), or a list of values. Default: (5, 25, 5).
   :type complexity: int or array-like
   :param symbols_range_distribution: Distribution for symbol range ('linear' or 'geometrical'). Default: None (uses provided values directly).
   :type symbols_range_distribution: str or None
   :param complexity_range_distribution: Distribution for complexity range ('linear' or 'geometrical'). Default: None.
   :type complexity_range_distribution: str or None
   :param iterations: Number of strings to generate per symbol-complexity combination. Default: 1.
   :type iterations: int
   :param save_csv: If True, saves the DataFrame to a CSV file. Default: False.
   :type save_csv: bool
   :param priorise_complexity: If True, prioritizes target complexity; if False, prioritizes using all symbols. Default: True.
   :type priorise_complexity: bool
   :param random_state: Seed for random number generation (incremented per iteration).
   :type random_state: int
   :return: DataFrame with columns ``nr_symbols`` (unique symbols), ``LZW_complexity`` (LZW complexity), ``length`` (string length), and ``string`` (generated string). Returns empty DataFrame if ``iterations < 1``.
   :rtype: pandas.DataFrame
   :raises ValueError: If distribution types are invalid ('linear' or 'geometrical' only).
   :raises Warning: If ``iterations < 1`` (returns empty DataFrame).

   .. note:: Infeasible cases (``nr_symbols > target_complexity``) are skipped, with a message printed for each.

   .. warning:: The ``random_state`` is incremented by the iteration index to ensure unique strings. For exact reproducibility, use a single iteration or provide a list of seeds.

   **Example**:

   .. code-block:: python

      >>> from lzw_string_library import lzw_string_seeds
      >>> df = lzw_string_seeds(symbols=[2, 3], complexity=[3, 6, 7], priorise_complexity=False, random_state=0)
      >>> print(df)
         nr_symbols  LZW_complexity  length       string
      0           2               3       3          ABA
      1           2               6       8     BABBABBA
      2           2               7      11  BAAABABAAAA
      3           3               3       3          BAC
      4           3               6       6       ABCACB
      5           3               7       8     ABCAAABB

   **CSV Output** (if ``save_csv=True``):
   Saves to a file named like ``StrLib_Symb2-3_LZWc3-7_Iters1.csv`` with filtered, sorted, and deduplicated strings.

Usage Guide
-----------

The module generates strings for applications requiring controlled complexity, such as symbolic time series analysis. Key features include:

- **String Generation**: Use ``lzw_string_generator`` for single strings or ``lzw_string_seeds`` for a library of strings.
- **LZW Complexity**: Calculated as the length of the output from ``lzwcompress`` after applying ``reduce`` to simplify periodic strings.
- **Symbol Restriction**: Limited to 52 alphabetical characters (A-z).
- **Flexibility**: Supports ranges of symbols and complexities with linear or geometrical distributions.

**Example Workflow**:

Generate a library with 2-4 symbols, complexity of 5, and save to CSV:

.. code-block:: python

   from lzw_string_library import lzw_string_seeds
   df = lzw_string_seeds(
       symbols=(2, 4, 2),
       complexity=5,
       symbols_range_distribution='linear',
       iterations=2,
       save_csv=True,
       priorise_complexity=True,
       random_state=42
   )
   print(df)

This generates strings with 2 and 4 symbols, each with a target LZW complexity of 5, repeated twice, and saves to a CSV file.

Limitations
-----------

- **Symbol Limit**: Maximum of 52 symbols due to the alphabetical restriction in ``_symbols``.
- **Performance**: The ``reduce`` function can be slow for long strings. Consider optimizing for large-scale use.
- **Randomness**: The ``random_state`` in ``lzw_string_seeds`` increments per iteration, which may affect reproducibility for multiple iterations.
- **Infeasible Cases**: Cases where ``nr_symbols > target_complexity`` are skipped, reducing the output size.

Recommendations
---------------

- **Progress Tracking**: Add ``tqdm`` for better progress visualization in ``lzw_string_seeds``:

  .. code-block:: python

     from tqdm import tqdm
     for n, i in tqdm(enumerate(iterator, 1), total=n_iter, desc="Processing"):
         ...

- **Input Validation**: Ensure ``nr_symbols`` and ``target_complexity`` are positive to avoid unexpected behavior.
- **Optimization**: Apply ``reduce`` only once at the end of string generation in ``lzw_string_generator`` to improve performance.

References
----------

- Welch, T. A. (1984). *A Technique for High-Performance Data Compression*. Computer, 17(6), 8-19.
- Rosetta Code LZW Compression: https://rosettacode.org/wiki/LZW_compression#Python