# H11 portfolio summary

DEV 45에서 H10 Hit@10 failure 중 ranking failure는 1건, C20에 relevant unit이 없는 candidate-generation failure는 1건이다.
C20 oracle Hit@10은 44/45이고 H10은 43/45이다. Requirement recall oracle은 54/58이고 H10은 52/58이다.
판정은 `Partial Support` (candidate_generation_comparable_or_larger)이다.
Oracle은 C20을 완벽히 재정렬할 때의 upper bound다. reranker가 그 차이를 달성한다는 뜻이 아니다.
TEST는 이 진단의 development data로 쓰지 않았다.
