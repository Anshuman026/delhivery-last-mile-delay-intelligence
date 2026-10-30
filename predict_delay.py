import os
import pandas as pd, numpy as np, joblib
from src.shipment_prep import clean_shipments, build_features, FEATURES
os.makedirs('outputs',exist_ok=True)
raw=pd.read_csv('data/new_shipments.csv'); hubs=pd.read_csv('data/hubs.csv')
med=pd.read_csv('models/impute_values.csv'); medians=dict(zip(med.column,med['median']))
clean,_=clean_shipments(raw,medians=medians)
feat=build_features(clean,hubs)
order=pd.read_csv('models/feature_order.csv').feature.tolist()
model=joblib.load('models/delay_model.pkl'); threshold=float(pd.read_csv('models/delay_threshold.csv').threshold.iloc[0])
feat['delay_probability']=np.round(model.predict_proba(feat[order])[:,1],4)
feat['flagged']=(feat.delay_probability>=threshold).astype(int)
segments=pd.read_csv('outputs/hub_segments.csv')[['hub_id','segment']]
feat=feat.merge(segments,on='hub_id',how='left',validate='many_to_one')
def action(r):
 if not r.flagged: return 'No action'
 if r.segment=='Escalate': return 'Escalate to hub manager'
 return 'Proactive customer call'
feat['action']=feat.apply(action,axis=1)
out=feat[['shipment_id','hub_id','segment','delay_probability','flagged','action']]
out.to_csv('outputs/new_shipment_actions.csv',index=False,float_format='%.4f')
print('Wrote actions for',len(out),'new shipments')
