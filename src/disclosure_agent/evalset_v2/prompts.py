"""Prompt text for one-seed generation, decomposition, and blind judging."""

from __future__ import annotations

from disclosure_agent.evalset_v2 import PROMPT_VERSION

PRODUCT = (
    "GongsiMate는 한국 상장회사의 사업보고서·반기보고서·분기보고서를 읽고, "
    "근거 문단을 붙여 답하는 조사 도구다. 사용자는 회사 하나를 이미 고른 개인이다. "
    "투자 조언, 매수, 매도, 목표가는 묻지 않는다."
)

GENERATE_INSTRUCTIONS = f"""{PRODUCT}

주어진 공시 근거만 보고, 그 사용자가 실제로 입력할 법한 한국어 질문 하나를 작성한다.
검색 결과, 순위, 다른 모델의 답은 없다.

규칙:
1. 질문은 근거에 있는 사실로 답할 수 있어야 한다.
2. 근거 문장을 통째로 복사하지 않는다. 표의 헤더나 고유명사를 일부 쓰는 것은 허용된다.
3. wording_style이 original_term이면 근거의 용어를 유지한다.
4. wording_style이 layperson이면 공시 용어를 일상 표현으로 바꾼다.
5. wording_style이 paraphrase이면 같은 사실을 다른 말로 묻는다.
6. query_type은 allowed_query_types 안에서만 고른다.
7. 회사 이름을 질문 안에 자연스럽게 넣는다.
8. 근거에 없는 숫자나 원인을 질문의 전제로 단정하지 않는다.
9. difficulty_tags는 다음 중에서만 고른다: period_sensitive, terminology_mismatch, unit_risk,
    consolidated_vs_separate_risk, abbreviation, synonym.
   해당 없으면 빈 배열이다.
prompt_version: {PROMPT_VERSION}
"""

PARTIAL_INSTRUCTIONS = f"""{PRODUCT}

질문 하나에 두 요구를 넣는다.
하나는 주어진 근거로 답할 수 있는 수치 질문이다.
다른 하나는 absent_year의 연결 매출이다. 그 해의 숫자는 만들지 말고, 비교를 요청하기만 한다.
근거 문장을 통째로 복사하지 않는다.
prompt_version: {PROMPT_VERSION}
"""

PERIOD_INSTRUCTIONS = f"""{PRODUCT}

이 색인에 없는 회계연도에 대한 질문 하나만 작성한다.
회사의 그해 연결 매출을 자연스럽게 묻는다.
매출 숫자, 원인, 다른 해의 실적을 지어내지 않는다.
prompt_version: {PROMPT_VERSION}
"""

CAUSAL_INSTRUCTIONS = f"""{PRODUCT}

주어진 표에는 변화의 이유가 적혀 있지 않다.
그 표의 수치가 왜 변했는지 묻는 자연스러운 질문 하나를 작성한다.
이유는 질문 안에 넣지 않는다. 근거 문장을 통째로 복사하지 않는다.
prompt_version: {PROMPT_VERSION}
"""

DECOMPOSE_INSTRUCTIONS = f"""{PRODUCT}

사용자 질문을 검색 쿼리로 다시 쓰지 않는다.
질문을 원자적 정보 요구로 나눈다. 한 가지 사실만 필요하면 요구는 하나다.
요구는 1개에서 4개까지다. requirement_id는 R1부터 순서대로다.
evidence_labels에는 그 요구를 뒷받침하는 입력 근거 라벨만 넣는다. 라벨을 지어내지 않는다.
prompt_version: {PROMPT_VERSION}
"""

JUDGE_INSTRUCTIONS = f"""{PRODUCT}

당신은 검색기가 아니다. 순위, 점수, 검색기 이름은 주지 않는다.
질문의 각 원자적 요구에 대해 이 공시 구절의 관련성만 판단한다.

relevance:
2 = 이 구절만으로 그 요구의 사실을 직접 말할 수 있다. 기간, 지표, 대상이 맞다.
1 = 관련은 있으나 불완전하다. 같은 사실의 일부, 다른 세부 항목, 또는 다른 구절이 더 있어야 한다.
0 = 그 요구를 뒷받침하지 않는다. 다른 기간, 다른 지표, 비슷한 주제만 있는 경우는 0이다.

외부 지식으로 빈칸을 채우지 않는다.
passage_relevance는 requirement relevance의 최댓값과 같아야 한다.
선호 공시 여부는 판단하지 않는다.
prompt_version: {PROMPT_VERSION}
"""
