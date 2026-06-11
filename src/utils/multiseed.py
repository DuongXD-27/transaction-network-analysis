import numpy as np

def run_multiseed(run_single_seed_fn, exp_name="Experiment", seeds=[42, 0, 123]):
    results = []
    for seed in seeds:
        print(f"\n{exp_name} - Seed: {seed}")
        metrics = run_single_seed_fn(seed)
        metrics['seed'] = seed
        results.append(metrics)
        
    print(f"\n {exp_name} multi-seed results")
    
    metric_keys = [k for k in results[0].keys() if k != 'seed']
    
    for key in metric_keys:
        values = [res[key] for res in results]
        mean_val = np.mean(values)
        std_val = np.std(values)
        
        if key == 'f1_ill':
            display_key = 'F1(illicit)'
        elif key == 'precision':
            display_key = 'Precision'
        elif key == 'recall':
            display_key = 'Recall'
        elif key == 'auc_pr':
            display_key = 'AUC-PR'
        else:
            display_key = key.capitalize()
        print(f"{display_key:<11}: {mean_val:.4f} ± {std_val:.4f}")
        
    return results
