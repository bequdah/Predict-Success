import pandas as pd
import numpy as np
import os


def generate_synthetic_data(num_samples=200):
    """
    Generates realistic synthetic student performance data.

    Grading System:
    - Coursework (Midterms + Quizzes): 0 to 60
    - Final Exam (Simulated):          0 to 40
    - Passing threshold:               Total >= 60

    Study hours are DAILY (realistic university context):
    - Most students: 0.5 - 1.5 h/day
    - Average: ~2 h/day
    - High achievers: 3-4 h/day
    - Rare: 5-6 h/day
    """
    np.random.seed(42)

    # ------------------------------------------------------------------
    # 1. FEATURE DISTRIBUTIONS
    # ------------------------------------------------------------------

    # Study hours: daily, log-normal clipped [0.5, 6]
    # avg ~2h, most students between 1-3h, rarely 5-6h
    study_hours = np.clip(
        np.random.lognormal(mean=0.7, sigma=0.5, size=num_samples), 0.5, 6
    )
    avg_study = study_hours.mean()

    # Attendance rate: Beta -> mostly high, clipped [50, 100]
    attendance_rate = 50 + 50 * np.random.beta(a=8, b=2, size=num_samples)

    # Coursework score: Normal centered ~40, clipped [0, 60]
    coursework_score = np.clip(
        np.random.normal(loc=40, scale=8, size=num_samples), 0, 60
    )

    # Extracurricular: 40% participate
    extracurricular = np.random.choice([0, 1], size=num_samples, p=[0.60, 0.40])

    # ------------------------------------------------------------------
    # 2. SIMULATE FINAL EXAM SCORE  (0-40)
    # ------------------------------------------------------------------

    # Base: correlated with coursework (same student ~ same ability)
    base_final = 0.45 * coursework_score * (40 / 60)

    # Study effect: bigger multipliers since daily hours are small numbers
    # Above avg -> strong boost, below avg -> weak boost
    study_effect = np.where(
        study_hours >= avg_study,
        8.0 * np.log1p(study_hours),
        3.5 * np.log1p(study_hours)
    )

    # Attendance: moderate effect
    # Below 70% -> penalty, above 85% -> small bonus
    attendance_effect = np.where(
        attendance_rate < 70,
        -5 * ((70 - attendance_rate) / 20),
        np.where(
            attendance_rate > 85,
            3 * ((attendance_rate - 85) / 15),
            0
        )
    )

    # Extracurricular: very light effect
    extra_effect = 1.5 * extracurricular

    # Synergy: high coursework (40+) AND above-avg study -> small extra boost
    synergy = (
        0.10
        * np.clip(coursework_score - 40, 0, None)
        * np.clip(study_hours - avg_study, 0, None)
    )

    # Isolation penalty: above-avg study BUT attendance < 65%
    isolation_penalty = np.where(
        (study_hours > avg_study) & (attendance_rate < 65), -4, 0
    )

    # Realistic noise
    noise = np.random.normal(0, 3.5, num_samples)

    final_exam_score = np.clip(
        base_final + study_effect + attendance_effect + extra_effect
        + synergy + isolation_penalty + noise,
        0, 40
    )

    # ------------------------------------------------------------------
    # 3. TOTAL SCORE & PROBABILISTIC PASS/FAIL
    # ------------------------------------------------------------------

    total_score = coursework_score + final_exam_score

    # Borderline boost: students 50-59 get a nudge (close to passing)
    borderline_boost = np.where((total_score >= 50) & (total_score < 60), 6, 0)

    logit = (total_score + borderline_boost - 60) / 7
    prob = 1 / (1 + np.exp(-logit))

    # Stochastic label: prevents trivial 100% accuracy
    passed = (np.random.random(num_samples) < prob).astype(int)

    # ------------------------------------------------------------------
    # 4. ASSEMBLE DATAFRAME
    # ------------------------------------------------------------------

    df = pd.DataFrame({
        'study_hours':                np.round(study_hours, 2),
        'attendance_rate':            np.round(attendance_rate, 2),
        'coursework_score':           np.round(coursework_score, 2),
        'extracurricular_activities': extracurricular,
        'passed':                     passed
    })

    # ------------------------------------------------------------------
    # 5. SANITY CHECKS
    # ------------------------------------------------------------------

    above_avg_mask = study_hours >= avg_study
    below_avg_mask = ~above_avg_mask

    print("=" * 60)
    print(f"  Synthetic dataset -- {num_samples} rows")
    print("=" * 60)
    print(f"  Overall pass rate          : {df['passed'].mean():.1%}")
    print(f"  Pass rate (above-avg study): {df.loc[above_avg_mask, 'passed'].mean():.1%}  |  fail: {1 - df.loc[above_avg_mask, 'passed'].mean():.1%}")
    print(f"  Pass rate (below-avg study): {df.loc[below_avg_mask, 'passed'].mean():.1%}  |  fail: {1 - df.loc[below_avg_mask, 'passed'].mean():.1%}")
    print(f"  Avg study hours (daily)    : {study_hours.mean():.1f} h  (threshold={avg_study:.1f})")
    print(f"  Study hours distribution   : min={study_hours.min():.1f}  median={np.median(study_hours):.1f}  max={study_hours.max():.1f}")
    print(f"  Avg coursework (0-60)      : {df['coursework_score'].mean():.1f}")
    print(f"  Avg attendance             : {df['attendance_rate'].mean():.1f} %")
    print(f"  Avg final exam (0-40)      : {final_exam_score.mean():.1f}")
    print(f"  Avg total (0-100)          : {total_score.mean():.1f}")
    borderline = ((total_score >= 50) & (total_score < 60)).sum()
    print(f"  Borderline students (50-59): {borderline}  -> given pass boost")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 6. SAVE
    # ------------------------------------------------------------------

    output_path = os.path.join('data', 'data.csv')
    os.makedirs('data', exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"  Saved -> {output_path}")

    return df


if __name__ == "__main__":
    generate_synthetic_data(200)