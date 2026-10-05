"""Retrieval benchmark v2.1.

v2.1 re-splits and re-keys qrels from the frozen v2 judgments.
It does not generate questions, call a judge, or change product retrieval.
``multi_company_retrieval_v2`` stays a historical artifact.
"""

DATASET_ID = "multi_company_retrieval_v2_1"
PARENT_DATASET_ID = "multi_company_retrieval_v2"
PROMPT_VERSION = "retrieval-evalset-v2.1"
