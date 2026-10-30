import os, json
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from src.shipment_prep import FEATURES, NOISE_COLUMNS
os.makedirs('outputs',exist_ok=True)
df=pd.read_csv('data/modelling_table.csv'); X=df[FEATURES]; y=df.is_delayed
X_train,X_temp,y_train,y_temp=train_test_split(X,y,test_size=.3,random_state=42,stratify=y)
X_val,X_test,y_val,y_test=train_test_split(X_temp,y_temp,test_size=.5,random_state=42,stratify=y_temp)
def metric_row(name, yt, pred):
 tn,fp,fn,tp=confusion_matrix(yt,pred,labels=[0,1]).ravel()
 return {'model':name,'TN':tn,'FP':fp,'FN':fn,'TP':tp,'accuracy':accuracy_score(yt,pred),'precision':precision_score(yt,pred,zero_division=0),'recall':recall_score(yt,pred,zero_division=0),'f1':f1_score(yt,pred,zero_division=0)}
rows=[metric_row('blind_all_on_time',y_val,np.zeros(len(y_val),dtype=int))]
log=Pipeline([('scaler',StandardScaler()),('model',LogisticRegression())]).fit(X_train,y_train)
rows.append(metric_row('logistic_regression',y_val,log.predict(X_val)))
tree_rows=[]; trees={}
for d in [2,3,4,5,6,8,10]:
 m=DecisionTreeClassifier(criterion='gini',max_depth=d,random_state=42).fit(X_train,y_train); trees[d]=m
 r=metric_row(f'decision_tree_depth_{d}',y_val,m.predict(X_val)); tree_rows.append((d,r))
best_d=sorted(tree_rows,key=lambda z:(-z[1]['f1'],z[0]))[0][0]
rows.append(next(r for d,r in tree_rows if d==best_d))
rf17=RandomForestClassifier(n_estimators=100,random_state=42).fit(X_train,y_train)
rows.append(metric_row('random_forest_default',y_val,rf17.predict(X_val)))
pd.DataFrame(rows).to_csv('outputs/part_c_models.csv',index=False,float_format='%.4f')
# Feature signal diagnostics
selector_all=SelectKBest(f_classif,k='all').fit(X_train,y_train)
fscores=selector_all.scores_; importances=rf17.feature_importances_
fs=pd.DataFrame({'feature':FEATURES,'f_score':fscores,'rf_importance':importances})
fs['f_rank']=fs['f_score'].rank(method='min',ascending=False).astype(int); fs['is_noise']=fs.feature.isin(NOISE_COLUMNS)
fs=fs[['feature','f_score','f_rank','rf_importance','is_noise']].sort_values('f_rank')
fs.to_csv('outputs/part_c_feature_scores.csv',index=False,float_format='%.4f')
select_rows=[]
for k in [5,7,10,12,17]:
 m=Pipeline([('selector',SelectKBest(f_classif,k=k)),('model',RandomForestClassifier(n_estimators=100,random_state=42))]).fit(X_train,y_train)
 select_rows.append({'k':k,'val_f1':f1_score(y_val,m.predict(X_val),zero_division=0)})
pd.DataFrame(select_rows).to_csv('outputs/part_c_selectk.csv',index=False,float_format='%.4f')
# Fit selector on training rows only; tune forest over selected seven columns
selector=SelectKBest(f_classif,k=7).fit(X_train,y_train)
Xt=selector.transform(X_train); Xv=selector.transform(X_val)
param={'n_estimators':[50,100,150],'max_depth':[4,6,8],'min_samples_leaf':[1,5,10]}
grid=GridSearchCV(RandomForestClassifier(random_state=42),param_grid=param,cv=5).fit(Xt,y_train)
rand=RandomizedSearchCV(RandomForestClassifier(random_state=42),param_distributions=param,n_iter=9,cv=5,random_state=42).fit(Xt,y_train)
def tuning_row(name, candidates, folds, fits, params, cvacc, pred):
 return {'search':name,'candidates':candidates,'cv_folds':folds,'total_fits':fits,'best_params':json.dumps(params,sort_keys=True),'best_cv_accuracy':cvacc,'val_f1':f1_score(y_val,pred,zero_division=0)}
tune=[tuning_row('grid_search',27,5,135,grid.best_params_,grid.best_score_,grid.best_estimator_.predict(Xv)),tuning_row('randomized_search',9,5,45,rand.best_params_,rand.best_score_,rand.best_estimator_.predict(Xv))]
deforest=RandomForestClassifier(n_estimators=100,random_state=42).fit(Xt,y_train)
tune.append(tuning_row('default_forest_same_7_features',1,0,1,{'n_estimators':100,'max_depth':None,'min_samples_leaf':1},'',deforest.predict(Xv)))
pd.DataFrame(tune).to_csv('outputs/part_c_tuning.csv',index=False,float_format='%.4f')
# highest validation F1; ties default, grid, randomized
priority={'default_forest_same_7_features':0,'grid_search':1,'randomized_search':2}
chosen=sorted(tune,key=lambda r:(-r['val_f1'],priority[r['search']]))[0]
if chosen['search']=='grid_search': params=grid.best_params_
elif chosen['search']=='randomized_search': params=rand.best_params_
else: params={'n_estimators':100,'max_depth':None,'min_samples_leaf':1}
final=Pipeline([('selector',SelectKBest(f_classif,k=7)),('model',RandomForestClassifier(random_state=42,**params))]).fit(X_train,y_train)
probs=final.predict_proba(X_val)[:,1]
thresholds=[]
for t in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]: thresholds.append({'threshold':t,**{k:v for k,v in metric_row('x',y_val,(probs>=t).astype(int)).items() if k not in ('model',)}})
pd.DataFrame(thresholds).to_csv('outputs/part_c_thresholds.csv',index=False,float_format='%.4f')
valid=[r for r in thresholds if r['recall']>=.8]
threshold=max([r['threshold'] for r in valid],default=.1)
test_probs=final.predict_proba(X_test)[:,1]
testrow={'threshold':threshold,**{k:v for k,v in metric_row('x',y_test,(test_probs>=threshold).astype(int)).items() if k not in ('model',)}}
pd.DataFrame([testrow]).to_csv('outputs/part_c_final_test.csv',index=False,float_format='%.4f')
pd.DataFrame([{'selector_k':7,'n_estimators':params['n_estimators'],'max_depth':'unlimited' if params.get('max_depth') is None else params.get('max_depth'),'min_samples_leaf':params['min_samples_leaf'],'threshold':threshold,'chosen_from':chosen['search']}]).to_csv('outputs/part_c_final_config.csv',index=False)
print('Part C complete; chosen:',chosen['search'],'threshold:',threshold)
