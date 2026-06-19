from pathlib import Path
import pandas as pd
p = Path('data/raw/')
for f in sorted(p.glob('*.csv')):
    try:
        df = pd.read_csv(f)
        print(f.name, len(df))
    except Exception as e:
        print(f.name, 'ERR', e)
