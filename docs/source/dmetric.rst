.. _string-metrics:

String Distance and Similarity Metrics
======================================

This module provides a collection of string distance and similarity metrics, each implemented to strictly adhere to their formal definitions. These functions are robust, handle edge cases (e.g., empty strings, identical strings), and are thoroughly tested for correctness. They are useful for tasks such as spell-checking, fuzzy matching, sequence alignment, and text similarity analysis.

The metrics include edit-based distances (e.g., Levenshtein, Damerau-Levenshtein), similarity measures (e.g., Jaro, Jaro-Winkler, Cosine), and alignment-based methods (e.g., Smith-Waterman). Each function is documented with its purpose, parameters, return values, and usage examples.

.. contents:: Table of Contents
   :local:
   :depth: 2

.. _metrics:


This project provides a collection of string distance and similarity metrics, both normalized and non-normalized, implemented in Python with thorough testing and robust edge-case handling. These metrics are useful for tasks like spell-checking, fuzzy matching, text similarity, and sequence alignment.

The following table summarizes the implemented metrics, including both normalized and non-normalized versions.

+------------------------------------------+------------+------------+-----------------------------------------------------------------------------------+------------+------------------------------------------+
| Metric Name                              | Type       | Normalized | Key Features                                                                      | Complexity | Primary Use Cases                        |
+==========================================+============+============+===================================================================================+============+==========================================+
| Levenshtein Distance                     | Distance   | No         | Counts insertions, deletions, substitutions; returns raw edit count               | O(n*m)     | Spell-checking, sequence alignment       |
+------------------------------------------+------------+------------+-----------------------------------------------------------------------------------+------------+------------------------------------------+
| Normalized Levenshtein Distance          | Distance   | Yes        | Counts insertions, deletions, substitutions; normalized by max length to [0,1]    | O(n*m)     | Spell-checking, sequence alignment       |
+------------------------------------------+------------+------------+-----------------------------------------------------------------------------------+------------+------------------------------------------+
| Hamming Distance                         | Distance   | No         | Counts differing positions in equal-length strings; returns raw count             | O(n)       | Error detection, fixed-length sequences  |
+------------------------------------------+------------+------------+-----------------------------------------------------------------------------------+------------+------------------------------------------+
| Normalized Hamming Distance              | Distance   | Yes        | Counts differing positions in equal-length strings; normalized by length to [0,1] | O(n)       | Error detection, fixed-length sequences  |
+------------------------------------------+------------+------------+-----------------------------------------------------------------------------------+------------+------------------------------------------+
| Jaro Similarity                          | Similarity | Yes        | Measures matching characters and transpositions, inherently normalized to [0,1]   | O(n*m)     | Record linkage, fuzzy matching           |
+------------------------------------------+------------+------------+-----------------------------------------------------------------------------------+------------+------------------------------------------+
| Jaro-Winkler Distance                    | Similarity | Yes        | Enhances Jaro with prefix weighting, inherently normalized to [0,1]               | O(n*m)     | Name matching, deduplication             |
+------------------------------------------+------------+------------+-----------------------------------------------------------------------------------+------------+------------------------------------------+
| Cosine Similarity                        | Similarity | Yes        | Compares word frequency vectors, inherently normalized to [0,1]                   | O(n)       | Text similarity, document comparison     |
+------------------------------------------+------------+------------+-----------------------------------------------------------------------------------+------------+------------------------------------------+
| Cosine Bigram Similarity                 | Similarity | Yes        | Compares bigram frequency vectors, inherently normalized to [0,1]                 | O(n)       | Character-level text similarity          |
+------------------------------------------+------------+------------+-----------------------------------------------------------------------------------+------------+------------------------------------------+
| LCS Distance                             | Distance   | No         | Based on longest common subsequence; returns raw difference                       | O(n*m)     | Sequence alignment, diff tools           |
+------------------------------------------+------------+------------+-----------------------------------------------------------------------------------+------------+------------------------------------------+
| Normalized LCS Distance                  | Distance   | Yes        | Based on longest common subsequence; normalized by max length to [0,1]            | O(n*m)     | Sequence alignment, diff tools           |
+------------------------------------------+------------+------------+-----------------------------------------------------------------------------------+------------+------------------------------------------+
| Dice’s Coefficient                       | Similarity | Yes        | Measures shared bigram overlap, inherently normalized to [0,1]                    | O(n)       | Short text similarity, fuzzy matching     |
+------------------------------------------+------------+------------+-----------------------------------------------------------------------------------+------------+------------------------------------------+
| Smith-Waterman Distance                  | Distance   | No         | Local alignment score (match=2, mismatch=-1, gap=-1); returns negative score      | O(n*m)     | Bioinformatics, partial sequence matching |
+------------------------------------------+------------+------------+-----------------------------------------------------------------------------------+------------+------------------------------------------+
| Normalized Smith-Waterman Distance       | Distance   | Yes        | Local alignment score (match=2, mismatch=-1, gap=-1); normalized to [0,1]         | O(n*m)     | Bioinformatics, partial sequence matching |
+------------------------------------------+------------+------------+-----------------------------------------------------------------------------------+------------+------------------------------------------+
| Damerau-Levenshtein Distance             | Distance   | No         | Extends Levenshtein with transpositions; returns raw edit count                   | O(n*m)     | Typo correction, spell-checking          |
+------------------------------------------+------------+------------+-----------------------------------------------------------------------------------+------------+------------------------------------------+
| Normalized Damerau-Levenshtein Distance  | Distance   | Yes        | Extends Levenshtein with transpositions; normalized by max length to [0,1]        | O(n*m)     | Typo correction, spell-checking          |
+------------------------------------------+------------+------------+-----------------------------------------------------------------------------------+------------+------------------------------------------+

Usage Illustration
------------------

Below are examples demonstrating how to use each metric. These examples use the strings ``"hello"`` and ``"helo"`` (or equal-length strings for Hamming distance) to illustrate typical outputs.

.. code-block:: python

    from string_metrics import (
        levenshtein_distance, normalized_levenshtein_distance,
        hamming_distance, normalized_hamming_distance,
        jaro_similarity, jaro_winkler_distance,
        cosine_similarity, cosine_bigram_similarity,
        lcs_distance, normalized_lcs_distance,
        dice_coefficient,
        smith_waterman_distance, normalized_smith_waterman_distance,
        damerau_levenshtein_distance, normalized_damerau_levenshtein_distance
    )

    # Example strings
    s1 = "hello"
    s2 = "helo"
    s3 = "cat"
    s4 = "hat"  # For Hamming distance (equal length)

    # Levenshtein Distance
    print(f"Levenshtein Distance: {levenshtein_distance(s1, s2)}")  # Output: 1
    print(f"Normalized Levenshtein Distance: {normalized_levenshtein_distance(s1, s2):.4f}")  # Output: 0.2000

    # Hamming Distance (requires equal-length strings)
    try:
        print(hamming_distance(s1, s2))
    except ValueError as e:
        print(f"Hamming Distance Error: {e}")
    print(f"Hamming Distance: {hamming_distance(s3, s4)}")  # Output: 1
    print(f"Normalized Hamming Distance: {normalized_hamming_distance(s3, s4):.4f}")  # Output: 0.3333

    # Jaro Similarity
    print(f"Jaro Similarity: {jaro_similarity(s1, s2):.4f}")  # Output: 0.9333
    print(f"Jaro-Winkler Distance: {jaro_winkler_distance(s1, s2):.4f}")  # Output: 0.9533

    # Cosine Similarity
    print(f"Cosine Similarity: {cosine_similarity(s1, s2):.4f}")  # Output: 1.0000
    print(f"Cosine Bigram Similarity: {cosine_bigram_similarity(s1, s2):.4f}")  # Output: 0.8660

    # LCS Distance
    print(f"LCS Distance: {lcs_distance(s1, s2)}")  # Output: 2
    print(f"Normalized LCS Distance: {normalized_lcs_distance(s1, s2):.4f}")  # Output: 0.4000

    # Dice’s Coefficient
    print(f"Dice’s Coefficient: {dice_coefficient(s1, s2):.4f}")  # Output: 0.8571

    # Smith-Waterman Distance
    print(f"Smith-Waterman Distance: {smith_waterman_distance(s1, s2)}")  # Output: -8
    print(f"Normalized Smith-Waterman Distance: {normalized_smith_waterman_distance(s1, s2):.4f}")  # Output: 0.2000

    # Damerau-Levenshtein Distance
    print(f"Damerau-Levenshtein Distance: {damerau_levenshtein_distance(s1, s2)}")  # Output: 1
    print(f"Normalized Damerau-Levenshtein Distance: {normalized_damerau_levenshtein_distance(s1, s2):.4f}")  # Output: 0.2000


API Reference
-------------

Below are the detailed descriptions and usage examples for each function in the ``string_metrics`` module.

Levenshtein Distance
~~~~~~~~~~~~~~~~~~~

.. py:function:: levenshtein_distance(s1: str, s2: str) -> int

   Calculate the Levenshtein distance between two strings.

   :param s1: The first input string.
   :type s1: str
   :param s2: The second input string.
   :type s2: str
   :return: The minimum number of edit operations (insertions, deletions, or substitutions) required.
   :rtype: int

   **Example**:

   .. code-block:: python

      from string_metrics import levenshtein_distance

      print(levenshtein_distance("kitten", "sitting"))  # Output: 3
      print(levenshtein_distance("cat", "act"))         # Output: 2
      print(levenshtein_distance("cat", ""))            # Output: 3

Normalized Levenshtein Distance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:function:: normalized_levenshtein_distance(s1: str, s2: str) -> float

   Calculate the normalized Levenshtein distance between two strings.

   :param s1: The first input string.
   :type s1: str
   :param s2: The second input string.
   :type s2: str
   :return: The Levenshtein distance normalized by the maximum string length, ranging from 0 (identical) to 1 (completely different).
   :rtype: float

   **Example**:

   .. code-block:: python

      from string_metrics import normalized_levenshtein_distance

      print(normalized_levenshtein_distance("kitten", "sitting"))  # Output: 0.4286
      print(normalized_levenshtein_distance("cat", "act"))         # Output: 0.6667
      print(normalized_levenshtein_distance("cat", ""))            # Output: 1.0

Hamming Distance
~~~~~~~~~~~~~~~~

.. py:function:: hamming_distance(s1: str, s2: str) -> int

   Calculate the Hamming distance between two strings of equal length.

   :param s1: The first input string.
   :type s1: str
   :param s2: The second input string.
   :type s2: str
   :return: The number of positions where the strings differ.
   :rtype: int
   :raises ValueError: If the strings have different lengths.

   **Example**:

   .. code-block:: python

      from string_metrics import hamming_distance

      print(hamming_distance("karolin", "kathrin"))  # Output: 3
      print(hamming_distance("10110", "11110"))      # Output: 1
      # print(hamming_distance("cat", "cats"))       # Raises ValueError

Normalized Hamming Distance
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:function:: normalized_hamming_distance(s1: str, s2: str) -> float

   Calculate the normalized Hamming distance between two strings of equal length.

   :param s1: The first input string.
   :type s1: str
   :param s2: The second input string.
   :type s2: str
   :return: The Hamming distance normalized by the string length, ranging from 0 (identical) to 1 (completely different).
   :rtype: float
   :raises ValueError: If the strings have different lengths.

   **Example**:

   .. code-block:: python

      from string_metrics import normalized_hamming_distance

      print(normalized_hamming_distance("karolin", "kathrin"))  # Output: 0.4286
      print(normalized_hamming_distance("10110", "11110"))      # Output: 0.2
      # print(normalized_hamming_distance("cat", "cats"))       # Raises ValueError

Jaro Similarity
~~~~~~~~~~~~~~~

.. py:function:: jaro_similarity(s1: str, s2: str) -> float

   Calculate the Jaro similarity between two strings.

   :param s1: The first input string.
   :type s1: str
   :param s2: The second input string.
   :type s2: str
   :return: The Jaro similarity score, ranging from 0 (no similarity) to 1 (identical strings).
   :rtype: float

   **Example**:

   .. code-block:: python

      from string_metrics import jaro_similarity

      print(jaro_similarity("martha", "marhta"))  # Output: 0.9444
      print(jaro_similarity("cat", ""))           # Output: 0.0
      print(jaro_similarity("same", "same"))     # Output: 1.0

Jaro-Winkler Distance
~~~~~~~~~~~~~~~~~~~~~

.. py:function:: jaro_winkler_distance(s1: str, s2: str, p: float = 0.15) -> float

   Calculate the Jaro-Winkler similarity between two strings.

   :param s1: The first input string.
   :type s1: str
   :param s2: The second input string.
   :type s2: str
   :param p: Prefix scaling factor (default is 0.15 to match textdistance).
   :type p: float, optional
   :return: The Jaro-Winkler similarity score, ranging from 0 (no similarity) to 1 (identical strings).
   :rtype: float

   **Example**:

   .. code-block:: python

      from string_metrics import jaro_winkler_distance

      print(jaro_winkler_distance("martha", "marhta"))     # Output: 0.9667
      print(jaro_winkler_distance("dixon", "dicksonx"))    # Output: 0.8133
      print(jaro_winkler_distance("cat", ""))              # Output: 0.0

Cosine Similarity (Word-Based)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:function:: cosine_similarity(s1: str, s2: str) -> float

   Calculate the cosine similarity between two strings using word vectors.

   :param s1: The first input string.
   :type s1: str
   :param s2: The second input string.
   :type s2: str
   :return: The cosine similarity score, ranging from 0 (no similarity) to 1 (identical word sets).
   :rtype: float

   **Example**:

   .. code-block:: python

      from string_metrics import cosine_similarity

      print(cosine_similarity("cat hat", "hat cat"))  # Output: 1.0
      print(cosine_similarity("cat", "dog"))          # Output: 0.0
      print(cosine_similarity("", "dog"))             # Output: 0.0

Cosine Bigram Similarity
~~~~~~~~~~~~~~~~~~~~~~~~

.. py:function:: cosine_bigram_similarity(s1: str, s2: str) -> float

   Calculate the cosine similarity between two strings using bigram vectors.

   :param s1: The first input string.
   :type s1: str
   :param s2: The second input string.
   :type s2: str
   :return: The cosine similarity score, ranging from 0 (no similarity) to 1 (identical bigram sets).
   :rtype: float

   **Example**:

   .. code-block:: python

      from string_metrics import cosine_bigram_similarity

      print(cosine_bigram_similarity("cat", "cap"))   # Output: 0.5
      print(cosine_bigram_similarity("cat", "act"))   # Output: 0.0
      print(cosine_bigram_similarity("", "dog"))      # Output: 0.0

Longest Common Subsequence (LCS) Distance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:function:: lcs_distance(s1: str, s2: str) -> int

   Calculate the LCS-based distance between two strings.

   :param s1: The first input string.
   :type s1: str
   :param s2: The second input string.
   :type s2: str
   :return: The LCS distance, defined as len(s1) + len(s2) - 2 * len(LCS).
   :rtype: int

   **Example**:

   .. code-block:: python

      from string_metrics import lcs_distance

      print(lcs_distance("kitten", "sitting"))  # Output: 5
      print(lcs_distance("cat", "act"))         # Output: 2
      print(lcs_distance("cat", ""))            # Output: 3

Normalized LCS Distance
~~~~~~~~~~~~~~~~~~~~~~~

.. py:function:: normalized_lcs_distance(s1: str, s2: str) -> float

   Calculate the normalized LCS-based distance between two strings.

   :param s1: The first input string.
   :type s1: str
   :param s2: The second input string.
   :type s2: str
   :return: The LCS distance normalized by the maximum string length, ranging from 0 (identical) to 1 (completely different).
   :rtype: float

   **Example**:

   .. code-block:: python

      from string_metrics import normalized_lcs_distance

      print(normalized_lcs_distance("kitten", "sitting"))  # Output: 0.7143
      print(normalized_lcs_distance("cat", "act"))         # Output: 0.6667
      print(normalized_lcs_distance("cat", ""))            # Output: 1.0

Dice’s Coefficient
~~~~~~~~~~~~~~~~~~

.. py:function:: dice_coefficient(s1: str, s2: str) -> float

   Calculate Dice's coefficient between two strings using bigrams.

   :param s1: The first input string.
   :type s1: str
   :param s2: The second input string.
   :type s2: str
   :return: Dice's coefficient, ranging from 0 (no shared bigrams) to 1 (identical bigram sets).
   :rtype: float

   **Example**:

   .. code-block:: python

      from string_metrics import dice_coefficient

      print(dice_coefficient("night", "nacht"))  # Output: 0.25
      print(dice_coefficient("cat", "cat"))      # Output: 1.0
      print(dice_coefficient("cat", ""))         # Output: 0.0

Smith-Waterman Distance
~~~~~~~~~~~~~~~~~~~~~~~

.. py:function:: smith_waterman_distance(s1: str, s2: str, match_score: int = 2, mismatch_score: int = -1, gap_score: int = -1) -> int

   Calculate the Smith-Waterman distance between two strings.

   :param s1: The first input string.
   :type s1: str
   :param s2: The second input string.
   :type s2: str
   :param match_score: Score for matching characters (default is 2).
   :type match_score: int, optional
   :param mismatch_score: Score for mismatching characters (default is -1).
   :type mismatch_score: int, optional
   :param gap_score: Score for gaps (default is -1).
   :type gap_score: int, optional
   :return: The inverse of the maximum local alignment score (lower scores indicate greater distance).
   :rtype: int

   **Example**:

   .. code-block:: python

      from string_metrics import smith_waterman_distance

      print(smith_waterman_distance("kitten", "sitting"))  # Output: -10
      print(smith_waterman_distance("cat", "act"))         # Output: -5
      print(smith_waterman_distance("cat", ""))            # Output: 0

Normalized Smith-Waterman Distance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:function:: normalized_smith_waterman_distance(s1: str, s2: str, match_score: int = 2, mismatch_score: int = -1, gap_score: int = -1) -> float

   Calculate the normalized Smith-Waterman distance between two strings.

   :param s1: The first input string.
   :type s1: str
   :param s2: The second input string.
   :type s2: str
   :param match_score: Score for matching characters (default is 2).
   :type match_score: int, optional
   :param mismatch_score: Score for mismatching characters (default is -1).
   :type mismatch_score: int, optional
   :param gap_score: Score for gaps (default is -1).
   :type gap_score: int, optional
   :return: The normalized Smith-Waterman distance, ranging from 0 (identical) to 1 (completely different).
   :rtype: float

   **Example**:

   .. code-block:: python

      from string_metrics import normalized_smith_waterman_distance

      print(normalized_smith_waterman_distance("kitten", "sitting"))  # Output: 0.1667
      print(normalized_smith_waterman_distance("cat", "act"))         # Output: 0.3333
      print(normalized_smith_waterman_distance("cat", ""))            # Output: 0.0

Damerau-Levenshtein Distance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:function:: damerau_levenshtein_distance(s1: str, s2: str) -> int

   Calculate the Damerau-Levenshtein distance between two strings.

   :param s1: The first input string.
   :type s1: str
   :param s2: The second input string.
   :type s2: str
   :return: The minimum number of edit operations (insertions, deletions, substitutions, or transpositions) required.
   :rtype: int

   **Example**:

   .. code-block:: python

      from string_metrics import damerau_levenshtein_distance

      print(damerau_levenshtein_distance("cat", "act"))    # Output: 1
      print(damerau_levenshtein_distance("cat", "hat"))    # Output: 1
      print(damerau_levenshtein_distance("cat", "cats"))   # Output: 1

Normalized Damerau-Levenshtein Distance
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. py:function:: normalized_damerau_levenshtein_distance(s1: str, s2: str) -> float

   Calculate the normalized Damerau-Levenshtein distance between two strings.

   :param s1: The first input string.
   :type s1: str
   :param s2: The second input string.
   :type s2: str
   :return: The Damerau-Levenshtein distance normalized by the maximum string length, ranging from 0 (identical) to 1 (completely different).
   :rtype: float

   **Example**:

   .. code-block:: python

      from string_metrics import normalized_damerau_levenshtein_distance

      print(normalized_damerau_levenshtein_distance("cat", "act"))    # Output: 0.3333
      print(normalized_damerau_levenshtein_distance("cat", "hat"))    # Output: 0.3333
      print(normalized_damerau_levenshtein_distance("cat", "cats"))   # Output: 0.25

Usage Notes
-----------

- **Input Requirements**: All functions accept strings as input. For ``hamming_distance`` and ``normalized_hamming_distance``, strings must be of equal length.
- **Output Interpretation**:
  - **Non-normalized distance metrics** (Levenshtein, Hamming, LCS, Smith-Waterman, Damerau-Levenshtein): Return non-negative integers (except Smith-Waterman, which returns negative integers); higher values indicate greater difference.
  - **Normalized distance metrics** (Normalized Levenshtein, Normalized Hamming, Normalized LCS, Normalized Smith-Waterman, Normalized Damerau-Levenshtein): Return floats in [0,1]; 1 indicates completely different strings.
  - **Similarity metrics** (Jaro, Jaro-Winkler, Cosine, Cosine Bigram, Dice): Return floats in [0,1]; 1 indicates identical strings.
- **Edge Cases**: All functions handle empty strings, identical strings, and single-character strings appropriately.
- **Dependencies**: Requires Python's standard library (``collections.Counter`` and ``math``).
- **Jaro-Winkler Note**: The default prefix scaling factor is ``p = 0.15`` to match the ``textdistance`` library, differing from the standard ``p = 0.1``.

Example Application
-------------------

Here’s an example of using multiple metrics to compare two strings:

.. code-block:: python

   from slearn.symbols import (
       levenshtein_distance,
       jaro_winkler_distance,
       cosine_similarity,
       damerau_levenshtein_distance
   )

   s1 = "kitten"
   s2 = "sitting"

   print(f"Levenshtein Distance: {levenshtein_distance(s1, s2)}")         # Output: 3
   print(f"Jaro-Winkler Similarity: {jaro_winkler_distance(s1, s2):.3f}") # Output: ~0.746
   print(f"Cosine Similarity: {cosine_similarity(s1, s2):.3f}")           # Output: 0.0
   print(f"Damerau-Levenshtein Distance: {damerau_levenshtein_distance(s1, s2)}") # Output: 3
