import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.impute import SimpleImputer, KNNImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.linear_model import HuberRegressor, RANSACRegressor, TheilSenRegressor
from scipy import stats
from scipy.stats import shapiro
from scipy.stats import kstest, norm
from scipy.stats import anderson
from scipy.stats import zscore
from statsmodels.stats.diagnostic import linear_rainbow
from scipy.stats import ks_2samp, ttest_ind
import statsmodels.api as sm

st.set_page_config(layout="wide")
st.title("🔍 Multiple Linear Regression Analysis App")

# Store models and evaluations
model_results = {}

# --- Tab 1: Upload or Generate Dataset ---
tab1, tab2, tab3 = st.tabs(["1️⃣ Upload/Generate Data", "2️⃣ Handle Missing Data", "3️⃣ Handle Outliers in Data"])

with tab1:
    st.header("📁 Upload or Generate Dataset")

    uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
    else:
        st.info("No file uploaded. Generating synthetic dataset...")
        from sklearn.datasets import make_regression
        X, y = make_regression(n_samples=200, n_features=4, noise=10)
        df = pd.DataFrame(X, columns=['Feature1', 'Feature2', 'Feature3', 'Feature4'])
        df['Target'] = y

    st.subheader("🔹 Data Preview")
    st.write(df.head())

    st.subheader("🔹 Column Names")
    st.write(df.columns.tolist())

    target = st.selectbox("🎯 Select the Target Variable", df.columns)

    st.subheader("📊 Descriptive Statistics")
    st.write(df.describe())

    st.subheader("📈 Plot Graph")
    plot_type = st.selectbox("Choose plot type", ["scatter", "box", "hist", "heatmap"])
    x_col = st.selectbox("X-axis", df.columns)
    y_col = st.selectbox("Y-axis", df.columns)

    if plot_type == "scatter":
        fig, ax = plt.subplots()
        sns.scatterplot(data=df, x=x_col, y=y_col, ax=ax)
        st.pyplot(fig)
    elif plot_type == "box":
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x=x_col, ax=ax)
        st.pyplot(fig)
    elif plot_type == "hist":
        fig, ax = plt.subplots()
        df[x_col].hist(ax=ax)
        st.pyplot(fig)
    elif plot_type == "heatmap":
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)



# --- Tab 2: Handle Missing Data ---
with tab2:
    st.header("🛠️ Handle Missing Data & Outliers")

    method = st.selectbox("Select Imputation Method", ["Mean", "Median", "KNN", "Drop Rows", "mostfrequent", "IterativeImputer"])

    df2 = df.copy()
    # Imputation
    if method == "Mean":
        imputer = SimpleImputer(strategy='mean')
    elif method == "Median":
        imputer = SimpleImputer(strategy='median')
    elif method == "KNN":
        imputer = KNNImputer(n_neighbors=3)
    elif method == "mostfrequent":
        imputer = SimpleImputer(strategy= "most_frequent")
    elif method == "IterativeImputer":
        imputer = IterativeImputer()
    elif method == "Drop Rows":
        df2 = df2.dropna()
        imputer = None

    if imputer:
        df2[df2.columns] = imputer.fit_transform(df2)

    st.write(df2.head())
    #Regression Model
    X2 = df2.drop(columns=[target])
    y2 = df2[target]
    X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.2, random_state=42)
    model = LinearRegression().fit(X_train2, y_train2)
    preds = model.predict(X_test2)
    residuals = y_test2 - preds

    st.subheader("📈 Evaluation After Cleaning")
    st.write(f"R2 Score: {r2_score(y_test2, preds):.3f}")
    st.write(f"RMSE: {mean_squared_error(y_test2, preds):.3f}")
    model_results[method] = (r2_score(y_test2, preds), mean_squared_error(y_test2, preds))

    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Residuals vs Predicted
    sns.scatterplot(x=preds, y=residuals, ax=axes[0])
    axes[0].axhline(y=0, color='red', linestyle='--')
    axes[0].set_title('Residuals vs. Predicted (Check Homoscedasticity & Linearity)')
    axes[0].set_xlabel('Predicted values')
    axes[0].set_ylabel('Residuals')

    # 2. Histogram of Residuals
    sns.histplot(residuals, kde=True, ax=axes[1], bins=20, color='skyblue')
    axes[1].set_title('Histogram of Residuals (Check Normality)')
    axes[1].set_xlabel('Residuals')

    # 3. QQ Plot
    sm.qqplot(residuals, line='45', ax=axes[2])
    axes[2].set_title('QQ Plot of Residuals (Check Normality)')

    # Display in Streamlit
    st.pyplot(fig)

    # Multicollinearity
    st.subheader("🔍 Multicollinearity (VIF)")
    vif = pd.DataFrame()
    vif["feature"] = X2.columns
    vif["VIF"] = [variance_inflation_factor(X2.values, i) for i in range(X2.shape[1])]
    st.write(vif)


    # Rainbow Test (Linearity)
    X_train_sm = sm.add_constant(X_train2)
    ols_model = sm.OLS(y_train2, X_train_sm).fit()
    stat_rb, p_rb = linear_rainbow(ols_model)

    # Anderson-Darling Test (Normality)
    result_ad = anderson(residuals)

    # Kolmogorov-Smirnov Test (Normality)
    stat_ks, p_ks = kstest(residuals, 'norm', args=(residuals.mean(), residuals.std()))

    # Shapiro-Wilk Test (Normality)
    stat_sw, p_sw = shapiro(residuals)

    # Display results
    st.subheader("1. Rainbow Test for Linearity")
    st.write(f"**Rainbow Test p-value:** `{p_rb:.4f}`")
    if p_rb >= 0.05:
        st.success("✅ Model likely exhibits linearity.")
    else:
        st.error("❌ Model may not exhibit linearity.")

    st.subheader("2. Anderson-Darling Test for Normality")
    st.write(f"**Anderson-Darling Test Statistic:** `{result_ad.statistic:.4f}`")
    st.write("Critical Values:")
    for i in range(len(result_ad.critical_values)):
        st.write(f"  {result_ad.significance_level[i]}%: {result_ad.critical_values[i]}")

    st.subheader("3. Kolmogorov-Smirnov (K-S) Test for Normality")
    st.write(f"**K-S Test p-value:** `{p_ks:.4f}`")
    if p_ks >= 0.05:
        st.success("✅ Residuals may be normally distributed.")
    else:
        st.error("❌ Residuals may not be normally distributed.")

    st.subheader("4. Shapiro-Wilk Test for Normality")
    st.write(f"**Shapiro-Wilk Test p-value:** `{p_sw:.4f}`")
    if p_sw >= 0.05:
        st.success("✅ Residuals appear normally distributed.")
    else:
        st.error("❌ Residuals do not appear normally distributed.")










# tab 3: handling Outliers-----
with tab3:
    data = st.selectbox("Select Imputation Method", ["dataset_with_missing_data", "dataset_imputed_missing"])
    if data == "dataset_with_missing_data":
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        df3 = df.copy()
    elif data == "dataset_imputed_missing":
        numeric_cols = df2.select_dtypes(include=np.number).columns.tolist()
        df3 = df2.copy()

    st.subheader("Step 2: Outlier Detection Method")
    method = st.radio("Choose Method", ["Z-score", "IQR"])

    outlier_report = {}

    if method == "Z-score":
        threshold = st.slider("Z-score Threshold", 2.0, 5.0, 3.0, 0.1)
        for col in numeric_cols:
            z_scores = np.abs(zscore(df3[col]))
            outliers = df3[z_scores > threshold]
            if not outliers.empty:
                outlier_report[col] = outliers.shape[0]
    else:  # IQR method
        for col in numeric_cols:
            Q1 = df3[col].quantile(0.25)
            Q3 = df3[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outliers = df3[(df3[col] < lower) | (df3[col] > upper)]
            if not outliers.empty:
                outlier_report[col] = outliers.shape[0]

    # Show results
    if outlier_report:
        st.write("**🔍 Outliers Detected in the Following Columns:**")
        st.write(pd.DataFrame.from_dict(outlier_report, orient='index', columns=['Count of Outliers']))
    else:
        st.success("✅ No Outliers Detected")

    # Step 3: Boxplot Visualization
    st.subheader("Step 3: Boxplot of Columns")
    selected_col = st.selectbox("Select column to visualize", numeric_cols)
    fig, ax = plt.subplots()
    sns.boxplot(data=df3, y=selected_col, ax=ax)
    st.pyplot(fig)

#----------------------------------------------------------------------------------------------
    # Step 4: Handling Method
    st.subheader("Step 4: Select Outlier Handling Method")
    handle_method = st.selectbox("Choose method to handle outliers", [
        "Remove", "Cap or Winsorize", "Transformation (Log)", "Imputation", "Robust Regression Models"
    ])

    df_handled = df3.copy()

    if handle_method == "Robust Regression Models":
        st.markdown("### Select Robust Model")
        robust_model = st.selectbox("Choose Robust Model", ["Huber Regressor", "RANSAC", "Theil-Sen Regressor"])
        target_column = st.selectbox("Select Target Column", numeric_cols)
        features = st.multiselect("Select Feature Columns", [col for col in numeric_cols if col != target_column])

        if features and target_column:
            X = df3[features]
            y = df3[target_column]

            if robust_model == "Huber Regressor":
                model = HuberRegressor()
            elif robust_model == "RANSAC":
                model = RANSACRegressor()
            else:
                model = TheilSenRegressor()

            model.fit(X, y)
            preds = model.predict(X)
            residuals = y - preds

            st.write(f"Robust Regression Model ({robust_model}) Fit Complete.")
            st.write("R² Score:", model.score(X, y))

            # Create subplots
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))

            # 1. Residuals vs Predicted
            sns.scatterplot(x=preds, y=residuals, ax=axes[0])
            axes[0].axhline(y=0, color='red', linestyle='--')
            axes[0].set_title('Residuals vs. Predicted (Check Homoscedasticity & Linearity)')
            axes[0].set_xlabel('Predicted values')
            axes[0].set_ylabel('Residuals')

            # 2. Histogram of Residuals
            sns.histplot(residuals, kde=True, ax=axes[1], bins=20, color='skyblue')
            axes[1].set_title('Histogram of Residuals (Check Normality)')
            axes[1].set_xlabel('Residuals')

            # 3. QQ Plot
            sm.qqplot(residuals, line='45', ax=axes[2])
            axes[2].set_title('QQ Plot of Residuals (Check Normality)')

            # Display in Streamlit
            st.pyplot(fig)



    elif handle_method == "Remove":
        for col in outlier_report:
            if method == "Z-score":
                df_handled = df_handled[np.abs(zscore(df_handled[col])) <= threshold]
            else:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower = Q1 - 1.5 * IQR
                upper = Q3 + 1.5 * IQR
                df_handled = df_handled[(df_handled[col] >= lower) & (df_handled[col] <= upper)]
        st.success("Outliers removed from selected columns.")

    elif handle_method == "Cap or Winsorize":
        for col in outlier_report:
            Q1 = df3[col].quantile(0.25)
            Q3 = df3[col].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            df_handled[col] = np.where(df3[col] < lower, lower, df3[col])
            df_handled[col] = np.where(df3[col] > upper, upper, df_handled[col])
        st.success("Outliers winsorized (capped) in selected columns.")

    elif handle_method == "Transformation (Log)":
        for col in outlier_report:
            df_handled[col] = np.log1p(df3[col])
        st.success("Log transformation applied to selected columns.")

    elif handle_method == "Imputation":
        imputer = SimpleImputer(strategy='median')
        df_handled[numeric_cols] = imputer.fit_transform(df3[numeric_cols])
        st.success("Missing/outlier values imputed using median.")

    st.subheader("📄 Final Data Snapshot")
    st.dataframe(df_handled.head())
    
    if handle_method != "Robust Regression Models":
        #Regression Model
        X3 = df_handled.drop(columns=[target])
        y3 = df_handled[target]
        X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, y3, test_size=0.2, random_state=42)
        model3 = LinearRegression().fit(X_train3, y_train3)
        preds3 = model.predict(X_test3)
        residuals3 = y_test3 - preds3

        st.subheader("📈 Evaluation After Cleaning")
        st.write(f"R2 Score: {r2_score(y_test3, preds3):.3f}")
        st.write(f"RMSE: {mean_squared_error(y_test3, preds3, squared=False):.3f}")
        model_results[handle_method] = (r2_score(y_test3, preds3), mean_squared_error(y_test3, preds3, squared=False))

        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1. Residuals vs Predicted
        sns.scatterplot(x=preds3, y=residuals3, ax=axes[0])
        axes[0].axhline(y=0, color='red', linestyle='--')
        axes[0].set_title('Residuals vs. Predicted (Check Homoscedasticity & Linearity)')
        axes[0].set_xlabel('Predicted values')
        axes[0].set_ylabel('Residuals')

        # 2. Histogram of Residuals
        sns.histplot(residuals3, kde=True, ax=axes[1], bins=20, color='skyblue')
        axes[1].set_title('Histogram of Residuals (Check Normality)')
        axes[1].set_xlabel('Residuals')

        # 3. QQ Plot
        sm.qqplot(residuals3, line='45', ax=axes[2])
        axes[2].set_title('QQ Plot of Residuals (Check Normality)')

        # Display in Streamlit
        st.pyplot(fig)

        # Multicollinearity
        st.subheader("🔍 Multicollinearity (VIF)")
        vif = pd.DataFrame()
        vif["feature"] = X3.columns
        vif["VIF"] = [variance_inflation_factor(X2.values, i) for i in range(X2.shape[1])]
        st.write(vif)


        # Rainbow Test (Linearity)
        X_train_sm3 = sm.add_constant(X_train3)
        ols_model3 = sm.OLS(y_train3, X_train_sm3).fit()
        stat_rb, p_rb = linear_rainbow(ols_model3)

        # Anderson-Darling Test (Normality)
        result_ad = anderson(residuals3)

        # Kolmogorov-Smirnov Test (Normality)
        stat_ks, p_ks = kstest(residuals3, 'norm', args=(residuals3.mean(), residuals3.std()))

        # Shapiro-Wilk Test (Normality)
        stat_sw, p_sw = shapiro(residuals3)

        # Display results
        st.subheader("1. Rainbow Test for Linearity")
        st.write(f"**Rainbow Test p-value:** `{p_rb:.4f}`")
        if p_rb >= 0.05:
            st.success("✅ Model likely exhibits linearity.")
        else:
            st.error("❌ Model may not exhibit linearity.")

        st.subheader("2. Anderson-Darling Test for Normality")
        st.write(f"**Anderson-Darling Test Statistic:** `{result_ad.statistic:.4f}`")
        st.write("Critical Values:")
        for i in range(len(result_ad.critical_values)):
            st.write(f"{result_ad.significance_level[i]}%: {result_ad.critical_values[i]}")

        st.subheader("3. Kolmogorov-Smirnov (K-S) Test for Normality")
        st.write(f"**K-S Test p-value:** `{p_ks:.4f}`")
        if p_ks >= 0.05:
            st.success("✅ Residuals may be normally distributed.")
        else:
            st.error("❌ Residuals may not be normally distributed.")

        st.subheader("4. Shapiro-Wilk Test for Normality")
        st.write(f"**Shapiro-Wilk Test p-value:** `{p_sw:.4f}`")
        if p_sw >= 0.05:
            st.success("✅ Residuals appear normally distributed.")
        else:
            st.error("❌ Residuals do not appear normally distributed.")

