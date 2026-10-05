import json,sys,numpy as np
rows=[json.loads(l) for l in open('sample/sample.jsonl')]
R=np.load('out/cache/text_fp32_ref.npy')
o={r['id']:r for r in map(json.loads,open(sys.argv[1]))}
c=np.array([np.array(o[r['id']]['text_embedding'])@R[i]/np.linalg.norm(o[r['id']]['text_embedding']) for i,r in enumerate(rows)])
print(sys.argv[2], f'text vs fp32 ref mean={c.mean():.4f} min={c.min():.4f} bad={(c<0.9).sum()} idx={list(np.where(c<0.9)[0])}', flush=True)
