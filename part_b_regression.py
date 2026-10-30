import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from src.shipment_prep import FEATURES
os.makedirs('outputs',exist_ok=True)
df=pd.read_csv('data/modelling_table.csv')
X=df[FEATURES]; y=df['delivery_hours']
X_train,X_temp,y_train,y_temp=train_test_split(X,y,test_size=.3,random_state=42)
X_val,X_test,y_val,y_test=train_test_split(X_temp,y_temp,test_size=.5,random_state=42)
sc=StandardScaler().fit(X_train)
checks=[('train_rows',len(X_train)),('val_rows',len(X_val)),('test_rows',len(X_test)),('train_scaled_max_abs_mean',np.abs(sc.transform(X_train).mean(axis=0)).max()),('test_scaled_max_abs_mean',np.abs(sc.transform(X_test).mean(axis=0)).max())]
pd.DataFrame(checks,columns=['metric','value']).to_csv('outputs/part_b_checks.csv',index=False,float_format='%.4f')

def score(name, split, yt, yp):
    mse=mean_squared_error(yt,yp)
    return {'model':name,'split':split,'MAE':mean_absolute_error(yt,yp),'MSE':mse,'RMSE':np.sqrt(mse),'R2':r2_score(yt,yp)}
rows=[score('mean_baseline','val',y_val,np.repeat(y_train.mean(),len(y_val)))]
depths=[1,2,3,4,6,8,12,None]; depth_rows=[]; trees={}
for d in depths:
    m=DecisionTreeRegressor(max_depth=d,random_state=42).fit(X_train,y_train); trees[d]=m
    depth_rows.append({'max_depth':'unlimited' if d is None else d,'train_rmse':np.sqrt(mean_squared_error(y_train,m.predict(X_train))),'val_rmse':np.sqrt(mean_squared_error(y_val,m.predict(X_val)))})
pd.DataFrame(depth_rows).to_csv('outputs/part_b_tree_depth.csv',index=False,float_format='%.4f')
best_depth=min(depth_rows,key=lambda r:(r['val_rmse'],999 if r['max_depth']=='unlimited' else r['max_depth']))['max_depth']
best_depth=None if best_depth=='unlimited' else int(best_depth)
models={
 'linear_regression':Pipeline([('scaler',StandardScaler()),('model',LinearRegression())]),
 'decision_tree':trees[best_depth],
 'random_forest':RandomForestRegressor(n_estimators=100,random_state=42),
 'gradient_boosting':GradientBoostingRegressor(n_estimators=100,learning_rate=.1,random_state=42),
}
for name,m in models.items():
    if name!='decision_tree': m.fit(X_train,y_train)
    rows.append(score(name,'val',y_val,m.predict(X_val)))
pd.DataFrame(rows).to_csv('outputs/part_b_metrics.csv',index=False,float_format='%.4f')
lrrows=[]
for lr in [.01,.1,1.0]:
 m=GradientBoostingRegressor(n_estimators=100,learning_rate=lr,random_state=42).fit(X_train,y_train)
 lrrows.append({'learning_rate':lr,'val_rmse':np.sqrt(mean_squared_error(y_val,m.predict(X_val))),'val_r2':r2_score(y_val,m.predict(X_val))})
pd.DataFrame(lrrows).to_csv('outputs/part_b_gb_learning_rate.csv',index=False,float_format='%.4f')
rank=sorted(models,key=lambda n: mean_squared_error(y_val,models[n].predict(X_val)))
cvrows=[]; cv=KFold(n_splits=5,shuffle=True,random_state=42)
for name in rank[:2]:
 vals=cross_val_score(models[name],pd.concat([X_train,X_val]),pd.concat([y_train,y_val]),cv=cv)
 cvrows.append({'model':name,'mean_r2':vals.mean(),'std_r2':vals.std()})
pd.DataFrame(cvrows).to_csv('outputs/part_b_cv.csv',index=False,float_format='%.4f')
win=rank[0]
rows.append(score(win,'test',y_test,models[win].predict(X_test)))
pd.DataFrame(rows).to_csv('outputs/part_b_metrics.csv',index=False,float_format='%.4f')
print('Part B complete; winner:',win,'depth:',best_depth)
