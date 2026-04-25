import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

df=pd.read_csv("dataset5g_with_ris.csv")
df.columns=df.columns.str.strip()

features=[
    "T-R Separation Distance (m)",
    "Frequency",
    "Time Delay (ns)",
    "RMS Delay Spread (ns)",
    "Azimuth AoD (degree)",
    "Elevation AoD (degree)",
    "Azimuth AoA (degree)",
    "Elevation AoA (degree)",
    "Received Power (dBm)",
    "Phase (rad)",
    "RIS Gain (dB)"
]

X=df[features]
y_without=df["Path Loss (dB)"]
y_with=df["RIS Path Loss (dB)"]

X_train,X_test,y_without_train,y_without_test,y_with_train,y_with_test=train_test_split(
    X,y_without,y_with,test_size=0.2,random_state=42
)

results=[]

def metric(y_train,y_train_pred,y_test,y_test_pred,model_name,target):
    train_mae=mean_absolute_error(y_train,y_train_pred)
    train_rmse=np.sqrt(mean_squared_error(y_train,y_train_pred))
    train_r2=r2_score(y_train,y_train_pred)

    test_mae=mean_absolute_error(y_test,y_test_pred)
    test_rmse=np.sqrt(mean_squared_error(y_test,y_test_pred))
    test_r2=r2_score(y_test,y_test_pred)

    print("\nTraining Performance:")
    print("MAE:",train_mae)
    print("RMSE:",train_rmse)
    print("R2:",train_r2)

    print("\nTesting Performance:")
    print("MAE:",test_mae)
    print("RMSE:",test_rmse)
    print("R2:",test_r2)

    results.append([model_name,target,train_mae,train_rmse,train_r2,test_mae,test_rmse,test_r2])

def make_model(name):
    if name=="Linear Regression":
        return LinearRegression()
    elif name=="Decision Tree":
        return DecisionTreeRegressor(random_state=42)
    else:
        return RandomForestRegressor(n_estimators=200,max_depth=15,random_state=42)

def evaluate(name):
    print("\n==========================")
    print("Model:",name)
    print("==========================")

    model_without=make_model(name)
    model_without.fit(X_train,y_without_train)

    pred_train=model_without.predict(X_train)
    pred_test=model_without.predict(X_test)

    print("\n----- Path Loss WITHOUT RIS -----")
    metric(y_without_train,pred_train,y_without_test,pred_test,name,"Without RIS")

    model_with=make_model(name)
    model_with.fit(X_train,y_with_train)

    pred_train=model_with.predict(X_train)
    pred_test=model_with.predict(X_test)

    print("\n----- Path Loss WITH RIS -----")
    metric(y_with_train,pred_train,y_with_test,pred_test,name,"With RIS")

    return model_without,model_with

evaluate("Linear Regression")
evaluate("Decision Tree")
best_without,best_with=evaluate("Random Forest")

result_df=pd.DataFrame(results,columns=[
    "Model","Target","Train MAE","Train RMSE","Train R2","Test MAE","Test RMSE","Test R2"
])

result_df.to_csv("model_metrics_results.csv",index=False)

# -------------------------
# GRAPH 1
# -------------------------
idx=np.argsort(X_test["T-R Separation Distance (m)"])
dist=X_test.iloc[idx]["T-R Separation Distance (m)"]

pred_without_sorted=best_without.predict(X_test.iloc[idx])
pred_with_sorted=best_with.predict(X_test.iloc[idx])

plt.figure()
plt.plot(dist,pred_without_sorted,label="Without RIS")
plt.plot(dist,pred_with_sorted,label="With RIS")
plt.xlabel("Distance (m)")
plt.ylabel("Path Loss (dB)")
plt.title("Path Loss vs Distance")
plt.legend()
plt.grid()
plt.savefig("graph_1_path_loss_vs_distance.png",dpi=300,bbox_inches="tight")
plt.close()

# -------------------------
# GRAPH 2
# -------------------------
pred_without=best_without.predict(X_test)

plt.figure()
plt.scatter(y_without_test,pred_without)
plt.xlabel("Actual Path Loss Without RIS")
plt.ylabel("Predicted Path Loss Without RIS")
plt.title("Actual vs Predicted Without RIS")
plt.grid()
plt.savefig("graph_2_actual_vs_predicted_without_ris.png",dpi=300,bbox_inches="tight")
plt.close()

# -------------------------
# GRAPH 3
# -------------------------
pred_with=best_with.predict(X_test)

plt.figure()
plt.scatter(y_with_test,pred_with)
plt.xlabel("Actual Path Loss With RIS")
plt.ylabel("Predicted Path Loss With RIS")
plt.title("Actual vs Predicted With RIS")
plt.grid()
plt.savefig("graph_3_actual_vs_predicted_with_ris.png",dpi=300,bbox_inches="tight")
plt.close()

# -------------------------
# GRAPH 4
# -------------------------
model_names=["Linear Regression","Decision Tree","Random Forest"]
r2_without=[]
r2_with=[]

for name in model_names:
    r2_without.append(result_df[(result_df["Model"]==name)&(result_df["Target"]=="Without RIS")]["Test R2"].values[0])
    r2_with.append(result_df[(result_df["Model"]==name)&(result_df["Target"]=="With RIS")]["Test R2"].values[0])

x=np.arange(len(model_names))

plt.figure()
plt.bar(x-0.2,r2_without,0.4,label="Without RIS")
plt.bar(x+0.2,r2_with,0.4,label="With RIS")
plt.xticks(x,model_names)
plt.ylabel("Testing R2 Score")
plt.title("Model Comparison using R2 Score")
plt.legend()
plt.grid()
plt.savefig("graph_4_model_comparison_r2_score.png",dpi=300,bbox_inches="tight")
plt.close()

# -------------------------
# GRAPH 5
# -------------------------
improvement=pred_without_sorted-pred_with_sorted

plt.figure()
plt.plot(dist,improvement)
plt.xlabel("Distance (m)")
plt.ylabel("Improvement due to RIS (dB)")
plt.title("RIS Improvement vs Distance")
plt.grid()
plt.savefig("graph_5_ris_improvement_vs_distance.png",dpi=300,bbox_inches="tight")
plt.close()

# -------------------------
# GRAPH 6
# -------------------------
error_without=y_without_test-pred_without

plt.figure()
plt.hist(error_without,bins=30)
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("Error Distribution Without RIS")
plt.grid()
plt.savefig("graph_6_error_distribution_without_ris.png",dpi=300,bbox_inches="tight")
plt.close()

# -------------------------
# GRAPH 7
# -------------------------
error_with=y_with_test-pred_with

plt.figure()
plt.hist(error_with,bins=30)
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.title("Error Distribution With RIS")
plt.grid()
plt.savefig("graph_7_error_distribution_with_ris.png",dpi=300,bbox_inches="tight")
plt.close()

print("\nAll graphs saved successfully.")
print("Metrics saved as model_metrics_results.csv")

# -------------------------
# CUSTOM INPUT
# -------------------------
print("\n==========================")
print("Custom Input Prediction")
print("==========================")

ch=input("\nDo you want to enter custom values? yes/no: ")

if ch.lower()=="yes":
    tmp={}

    for col in features:
        val=float(input(col+" : "))
        tmp[col]=val

    custom_df=pd.DataFrame([tmp])

    ans_without=best_without.predict(custom_df)
    ans_with=best_with.predict(custom_df)

    print("\nPredicted Output:")
    print("Path Loss WITHOUT RIS:",ans_without[0],"dB")
    print("Path Loss WITH RIS:",ans_with[0],"dB")
    print("Improvement due to RIS:",ans_without[0]-ans_with[0],"dB")
else:
    print("\nProgram finished.")