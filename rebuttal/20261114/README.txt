FidRef 반박 자료 보완판 — 2026-11-14, v1

1. 제출판과의 관계

기존 익명 저장소 revision 16815afa9f9533ddf10fe01b6000d1120cb83b1c와
제출 ZIP은 보존됩니다. 이 보완판은 독립 디렉터리에 압축을 풀어 사용합니다.
기존 제출판 위에 덮어쓰는 방식이 아닙니다.

2. 포함 자료

- 제출판에서 누락된 결과 540개: 예산 스윕 480개, 과거 Identifiable 60개.
- 인용 자료 2개: analysis/sweep_figure_20260910/all_cells.csv,
  paper/iclr2027/supplement/metrics_detail.pdf.
- 추가 실험 및 후속 결과 661개와 초기 후보, 호출 기록, 재개 체크포인트.
  Deployment 정보 조건 360, Retail 정보 조건 120,
  Deployment 절단 후속 36, Retail 절단 후속 15 및 미전송 후속 7,
  Identifiable 원 실험 40, 총예산 후속 40 및 마지막 출력 후속 3,
  basic Shopping B24 14B/32B 각 20개.
- 비LLM 대조: 원 관측 궤적 40개에 조건부인 139개 요청 경계의 기호 합성,
  LLM/기호 계약 평가 80개. source_spend/source_usage는 관측 수집 비용입니다.
- 도메인, DSL, 프롬프트, 실험 커널, Retail 동결 소스와 비LLM 학습기.

661은 저장된 원 실험 및 후속 파일의 수입니다. 같은 시드의 후속 결과는
원본에 대응되며, 조건별 기본 분모는 20입니다. 개발 및 스모크 실행은 별도입니다.

3. 재검증

Python 3.10 이상에서 압축을 푼 최상위 디렉터리에서 실행합니다.

    python3 -B verify_rebuttal.py

네트워크 및 모델 요청 없이 파일 SHA-256, 누락 목록, 1,201개 결과의
Exact/IoU/CFR, 24개 스윕 셀, Deployment 정보 조건, Retail/Identifiable
교체 정책, B24, 비LLM 139개 요청 경계의 결정론적 재생을 검사합니다.

IoU = 100(A-FR)/(A+FA), CFR = 100(A-FR-PV)/(A+FA).
A는 실제 성공 상태 수이며, 파싱 실패를 포함한 셀 분모 20을 유지합니다.
평가 가능한 계약이 없는 실행은 전체 시도 평균에서 0으로 집계합니다.

4. 결과 및 정책 위치

모든 아래 경로는 analysis/rebuttal_20261106/ 아래입니다.

- deployment_results_20261113/summary.json: Deployment 18셀과 후조건 오류 검사.
- info_launch_20261113_v2/main/validation/: 조건별 전수 평가 및 효과 검사.
- reply_draft_20261113/LATEST_INTEGRATION.json: Retail 120개 원본과 22개 교체 대응.
- identifiable_length3_20261114/integrated_summary.json: 원본, 총예산 후속,
  마지막 출력 후속의 시드별 경로와 지표.
- nonllm_main_20261114/run/: 관측 패킷, 학습 출력, 평가, 집계.
- reply_review_20261114/RETRY_CELLS.csv: 기존 주표/절제의 절단 교체 셀과 전후값.
- reply_review_20261114/MISSING_FILES.csv: 제출판 누락 542파일의 원본 해시.

프로토콜 및 예산은 각 코호트 manifest.json과 원시 결과에 보존됩니다.
후속에서 앞선 응답을 재사용한 범위는 호출 기록과 교체 대응표로 확인합니다.

5. 정정 안내

기존 Table 9의 두 행은 전수 오라클 보조 counterexample_guided입니다.
새 B24는 표본 기반 Active이며 40개 모두 oracle_feedback_queries=0입니다.
과거 Identifiable 방법은 No-Coverage이고 FA 중앙값은 기본 예산 20,
2배 예산 12입니다. 신규 Full Active와 No-Coverage는 별도 코호트입니다.
모델, 서빙, 디코딩, 예산 차이는 결과의 해당 메타데이터를 따릅니다.

6. 익명화와 해시

RELEASE_MANIFEST.json의 source_sha256은 원본 바이트, sha256은 배포 바이트의
해시입니다. 운영 메타데이터의 로컬 경로, 사용자 식별자, 엔드포인트를 익명화했습니다.
요청 메시지, 응답, 계약, 평가, 예산, 사용량은 원본과 대조했습니다.
실험 기록 내부의 역사적 소스/프로토콜 해시는 원래 값을 유지합니다.
이 값을 배포 파일과 대조할 때는 매니페스트의 source_sha256 대응을 사용합니다.

서버 기동·자원 회수·알림·인증 도구는 내부 운영 자료입니다.
이 보완판의 실행 진입점은 저장된 결과의 오프라인 검증 도구이며,
실험 커널과 동결 소스는 알고리즘·정보 정책 검토를 위해 제공합니다.
