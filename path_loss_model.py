import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

# -------------------------
# LOAD DATA
# -------------------------
df=pd.read_csv("dataset5g_with_ris.csv")
df.columns=df.columns.str.strip()

# -------------------------
# FEATURES
# -------------------------
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

# -------------------------
# SPLIT 80-20
# -------------------------
X_train,X_test,y_without_train,y_without_test,y_with_train,y_with_test=train_test_split(
    X,y_without,y_with,test_size=0.2,random_state=42
)

# -------------------------
# METRICS FUNCTION
# -------------------------
def show_metrics(y_train,y_train_pred,y_test,y_test_pred):
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

# -------------------------
# FUNCTION TO TRAIN + EVAL
# -------------------------
def evaluate(model,name):
    print("\n==========================")
    print("Model:",name)
    print("==========================")

    # WITHOUT RIS
    model_without=model
    model_without.fit(X_train,y_without_train)

    y_without_train_pred=model_without.predict(X_train)
    y_without_test_pred=model_without.predict(X_test)

    print("\n----- Path Loss WITHOUT RIS -----")
    show_metrics(
        y_without_train,
        y_without_train_pred,
        y_without_test,
        y_without_test_pred
    )

    # WITH RIS
    if name=="Linear Regression":
        model_with=LinearRegression()
    elif name=="Decision Tree":
        model_with=DecisionTreeRegressor(random_state=42)
    else:
        model_with=RandomForestRegressor(
            n_estimators=200,
            max_depth=15,
            random_state=42
        )

    model_with.fit(X_train,y_with_train)

    y_with_train_pred=model_with.predict(X_train)
    y_with_test_pred=model_with.predict(X_test)

    print("\n----- Path Loss WITH RIS -----")
    show_metrics(
        y_with_train,
        y_with_train_pred,
        y_with_test,
        y_with_test_pred
    )

    return model_without,model_with

# -------------------------
# MODELS
# -------------------------
lr=LinearRegression()

dt=DecisionTreeRegressor(random_state=42)

rf=RandomForestRegressor(
    n_estimators=200,
    max_depth=15,
    random_state=42
)

# -------------------------
# RUN ALL MODELS
# -------------------------
evaluate(lr,"Linear Regression")
evaluate(dt,"Decision Tree")
best_without,best_with=evaluate(rf,"Random Forest")

# -------------------------
# CUSTOM INPUT PREDICTION
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