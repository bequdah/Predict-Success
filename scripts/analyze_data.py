import pandas as pd
import os

def analyze_data():
    data_path = os.path.join('data', 'data.csv')
    
    if not os.path.exists(data_path):
        print(f"Error: {data_path} not found!")
        return

    # Load data
    df = pd.read_csv(data_path)
    
    print("=" * 60)
    print("         DATA ANALYSIS REPORT (Current data.csv)")
    print("=" * 60)
    
    # 1. Basic Stats
    print(f"\n[1] Dataset Size: {len(df)} rows")
    
    # 2. Pass/Fail Distribution
    pass_counts = df['passed'].value_counts()
    pass_rate = (pass_counts.get(1, 0) / len(df)) * 100
    print(f"\n[2] Distribution:")
    print(f"    - Passed: {pass_counts.get(1, 0)} students ({pass_rate:.1f}%)")
    print(f"    - Failed: {pass_counts.get(0, 0)} students ({100 - pass_rate:.1f}%)")
    
    # 3. Comparing Groups (Averages)
    print("\n[3] Average Metrics by Status:")
    comparison = df.groupby('passed').mean()
    # Formatting for better display
    display_df = comparison.rename(index={0: 'Failed Students', 1: 'Passed Students'})
    print(display_df.round(2))
    
    # 4. Correlation with 'passed'
    print("\n[4] Feature Correlation with Success:")
    correlations = df.corr()['passed'].sort_values(ascending=False)
    print(correlations.drop('passed'))
    
    # 5. Quick Insights
    print("\n[5] Insights:")
    top_studier = df['study_hours'].max()
    print(f"    - Highest study hours: {top_studier} h/day")
    
    low_attendance_pass = df[(df['attendance_rate'] < 70) & (df['passed'] == 1)]
    print(f"    - Students who passed with low attendance (<70%): {len(low_attendance_pass)}")
    
    print("=" * 60)

if __name__ == "__main__":
    analyze_data()
