import json

keys = [
    'hybrid_static_columns','RandomForestClassifier','XGBClassifier','LogisticRegression',
    'PageHinkley','CRPI','linucb','customer_unique_id','behavior_drift_score',
    'gru_model','hybrid_meta_model','final_churn_probability','retention_priority',
    'customer_lifetime_days','avg_review_score','total_orders'
]

out = []
for nb_path in ['notebooks/02_preprocessing.ipynb', 'notebooks/04_sequential_encoder.ipynb']:
    out.append(f'\n### {nb_path}')
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    for i, cell in enumerate(nb.get('cells', [])):
        src = ''.join(cell.get('source', []))
        hits = [key for key in keys if key.lower() in src.lower()]
        if hits:
            out.append(f'CELL {i} HITS {hits[:8]}')
            for line in src.splitlines()[:80]:
                if any(k.lower() in line.lower() for k in keys):
                    out.append(line[:250])
            out.append('---')

with open('pipeline_snippets.txt', 'w', encoding='utf-8') as f:
    f.write('\n'.join(out))

print('wrote pipeline_snippets.txt')
