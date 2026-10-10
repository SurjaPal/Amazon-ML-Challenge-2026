Amazon ML Challenge 2026: Business Entity Resolution at Scale

Matching noisy business records (name, address, country) across independent sources of roughly 5 million records each. Built for the Amazon ML Challenge 2026 (via Unstop), Business Entity Resolution track.

The pipeline is two-stage: a fast candidate-retrieval stage (character n-gram TF-IDF plus a rare-token inverted index) followed by a LightGBM matcher with 56 features that scores and ranks the candidates.

Local validation result: 0.9885 Macro F0.5 on a 20,000-entity validation sample (precision 0.9949, recall 0.9773). See Results for exactly how this was measured.

The task

There are three sources of business records:

S1: the reference entity table.
S2 and S3: large, noisy record sets, where each record is a variant of some S1 entity (misspellings, reordered or missing address parts, differing formatting).

For every S1 entity, the goal is to output the list of S2 and S3 records that refer to it. Scale makes naive pairwise comparison impossible: the test set has 1,732,544 S1 entities and about 9.97 million S2/S3 records to match (4,887,273 in S2 and 5,082,316 in S3). Submissions are scored with Macro F0.5, which weights precision more heavily than recall.

Output format (matching_results.tsv):

source1_entity_id    matched_entity_ids
S1-53356671          S2-256033433,S2-453334141,S2-904412198,S3-468658814,S3-996978203
Approach
Normalization. Names and addresses are cleaned and tokenized with one shared normalizer, used identically for training, validation and test.
Candidate retrieval (blocking). For each S2/S3 record, a small set of plausible S1 candidates is retrieved by combining:
character n-gram TF-IDF similarity within name, address and number/country blocks, and
a rare-token inverted index, which finds S1 entities sharing the rarest (most informative) tokens with the record.
On the validation sample, rare-token retrieval alone places the true S1 entity in the candidate set for about 99.9% of records (99.925% when using the four rarest tokens, with about 71 candidates per record on average).
Feature engineering. Each (record, candidate) pair gets 56 features: fuzzy string similarities (RapidFuzz), TF-IDF cosine similarity, address-number overlap, a strong-address indicator, combined similarity scores, interaction features, and query-relative ranking features computed across a record's candidates.
Matching model. A LightGBM gradient-boosted model scores each pair. Each record is then assigned to its single best-scoring S1 candidate (target exclusivity, top-1), and the assignment is kept only if the score clears a tuned threshold.
Threshold tuning. The decision threshold is tuned on the full validation sample with a fine-grained sweep to maximize Macro F0.5.
Results

All numbers below are local validation results on a 20,000-entity S1 sample (69,231 target records) drawn from the provided training data. They are not leaderboard scores.

Configuration	Macro F0.5	Notes
Rare-token + TF-IDF retrieval union, LightGBM, threshold 0.64	0.9885	Precision 0.9949, recall 0.9773
LambdaRank ranker variant (experiment)	0.9866	Did not beat the classifier
Rare-token-only retrieval (faster, used for large-scale inference)	0.9593	About 191 records/s; projected about 14.5 hours for about 10M records

The 0.9885 figure uses the richer retrieval union. The rare-token-only configuration is faster and was used for the full test-set inference, at a lower validation score.

Repository contents
File	What it is
notebooks/01_exploration_zingg.ipynb	Earlier exploration, including Zingg-based entity-resolution baselines (which scored below the handcrafted baseline) and early blocking experiments.
notebooks/02_pipeline_and_inference.ipynb	The main notebook: blocking benchmarks, feature engineering, LightGBM matcher, threshold tuning, error analysis, and checkpointed large-scale inference.

The notebooks are saved with their outputs so the results above can be checked directly. Cell numbers in the comments (for example "CELL 12B") reflect the order experiments were run in, not a clean top-to-bottom script.

Reproducing

The notebooks were developed in Google Colab with data and artifacts stored on Google Drive, so file paths (for example /content/drive/MyDrive/AMAZON ML 2026/...) need to be changed to match your setup.

Data: not included in this repository (competition data).
Python packages: pandas, numpy, scipy, scikit-learn, lightgbm, rapidfuzz, joblib.
Large artifacts (TF-IDF matrices, blocking indexes, the trained model) are rebuilt by the notebooks and are not committed.
Tech stack

Python, pandas, NumPy, SciPy, scikit-learn (TF-IDF), LightGBM, RapidFuzz, Google Colab.

Author

Surja Pal (GitHub | LinkedIn)
