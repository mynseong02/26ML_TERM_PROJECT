# 기계학습 기반 학생 우울증 분류

**2026-1 Machine Learning Term Project**

데이터사이언스학과 12214214 김민성

---

## 1. 프로젝트 개요

본 프로젝트는 학생의 우울증 여부를 분류하고, 각 변수의 영향력을 해석 가능한 형태로 규명하는 것을 목표로 한다. 분석의 두 축은 다음과 같다.

1. **False Negative 최소화** — 우울증을 가진 학생을 "건강"으로 오분류하는 임상적으로 가장 위험한 케이스를 줄인다.
2. **해석 가능성 확보** — 로지스틱 회귀의 오즈비, 랜덤포레스트의 Feature/Permutation Importance, SHAP을 결합한 다층적 해석을 통해 위험·보호 요인을 규명한다.

## 2. 데이터셋

- **출처**: [Student Depression Dataset on Kaggle](https://www.kaggle.com/datasets/hopesb/student-depression-dataset) (Shodolamu Opeyemi)
- **라이선스**: Apache License 2.0 (학술 사용 허용)
- **표본 크기**: 인도 학생 27,901명 설문 응답
- **타겟 변수**: `Depression` (binary)
- **파일**: `student_depression_dataset.csv`

## 3. 분석 파이프라인

| 단계 | 내용 |
|---|---|
| EDA | 클래스 분포 (≈ 6:4), 변수별 히스토그램, 상관계수 히트맵, IQR 기반 이상치 제거 |
| 전처리 | 원-핫 인코딩 (Sleep / Diet / Degree), 이진 인코딩 (Yes/No), `StandardScaler` (연속형) |
| Baseline | 로지스틱 회귀 (L2 정규화, `class_weight='balanced'`) |
| Threshold 최적화 | F1.3-score 기준 5-Fold 평균 threshold = 0.274 |
| 비교 모형 | 랜덤포레스트 (GridSearchCV, `scoring='recall'`) |
| 가정 검정 | Box-Tidwell로 로짓 선형성 가정 확인 |
| 변수 해석 | Feature Importance → Permutation Importance → SHAP |

## 4. 주요 결과 요약

### 모델 성능 (Test set, threshold 0.274 적용)

| Metric | Logistic Regression | Random Forest (Tuned) |
|---|---|---|
| Accuracy | 0.842 | 0.842 |
| Recall (Class 1) | **0.939** | 0.886 |
| Precision (Class 1) | 0.813 | 0.850 |
| F1.3-score | **0.888** | — |
| FN 감소 | 기본 0.5 대비 ↓ | 577 → 556 (21건) |

### 식별된 위험·보호 요인 (오즈비 기준)

| 분류 | 변수 | OR |
|---|---|---|
| 위험 | Suicidal_Thoughts | 11.74 |
| 위험 | Academic_Pressure | 3.05 |
| 위험 | Diet_Unhealthy | 2.75 |
| 위험 | Financial_Stress | 2.20 |
| 보호 | Age | 0.59 |
| 보호 | Study_Satisfaction | 0.72 |
| 보호 | Sleep_gt8 | 0.77 |

## 5. 재현 방법 (Reproduction)

### 5.1 환경 설정

Python 3.10+ 환경을 권장한다.

```bash
# 저장소 클론
git clone https://github.com/mynseong02/26ML_TERM_PROJECT.git
cd 26ML_TERM_PROJECT

# 가상환경 생성 (선택)
python -m venv venv
source venv/bin/activate          # macOS / Linux
venv\Scripts\activate             # Windows

# 의존성 설치
pip install -r requirements.txt
```

### 5.2 데이터셋 준비

`student_depression_dataset.csv` 파일이 저장소에 포함되어 있다. 만약 누락된 경우 [Kaggle 페이지](https://www.kaggle.com/datasets/hopesb/student-depression-dataset)에서 다운로드 후 노트북과 같은 디렉터리에 배치한다.

### 5.3 노트북 실행

```bash
jupyter notebook depression_classification.ipynb
```

또는 셀 단위 실행 없이 전체 자동 실행:

```bash
jupyter nbconvert --to notebook --execute depression_classification.ipynb \
    --output depression_classification.ipynb \
    --ExecutePreprocessor.timeout=1800
```

### 5.4 재현성 (Reproducibility)

- 모든 무작위 연산에 `random_state=42` 적용
- `StratifiedKFold(shuffle=True, random_state=42)` 사용
- 실행 환경에 따라 SHAP 시각화에서 minor 차이 가능 (계산값은 동일)

## 6. 파일 구성

```
26ML_TERM_PROJECT/
├── depression_classification.ipynb   # 분석 노트북 (전체 파이프라인)
├── student_depression_dataset.csv    # 데이터셋
├── requirements.txt                  # 의존성 명세
└── README.md                         # 본 문서
```

## 7. 한계 및 향후 연구

- **표본 한계**: 인도 학생 단일 문화권 데이터 → 다문화 검증 필요
- **횡단면 데이터**: 인과 방향성 규명 불가 → 종단 연구 설계 필요
- **자가보고식 진단**: DSM-5 등 임상 진단 라벨로 재학습 필요
- **선형성 가정 위배**: `Age`, `Academic_Pressure`에서 Box-Tidwell 위배 → 비모수 모형(RF)으로 우회

## 8. 생성형 AI 활용 명시

본 프로젝트에서 Claude를 Jupyter Notebook 코드 검증, 보고서 문장 첨삭·오탈자 정리에 활용하였다.
## 9. 참고문헌

[1] Bodden, D. H. M., Stikkelbroek, Y., & Dirksen, C. D. (2018). Societal burden of adolescent depression, an overview and cost-of-illness study. *Journal of Affective Disorders*, 241, 256-262.

[2] WHO Western Pacific. *Strengthening minds — Malaysia strengthens efforts to enhance the mental health of children and adolescents.* https://www.who.int/westernpacific/newsroom/feature-stories/item/strengthening-minds---malaysia-strengthens-efforts-to-enhance-the-mental-health-of-children-and-adolescents

[3] Predicting Student Depression Using Machine Learning. *ResearchGate*. https://www.researchgate.net/publication/389847479

[4] Shodolamu Opeyemi. (2024). *Student Depression Dataset* [Data set]. Kaggle. https://www.kaggle.com/datasets/hopesb/student-depression-dataset