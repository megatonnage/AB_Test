import streamlit as st
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit app
st.title('AB Testing with Z-Test')

# Set the random seed for reproducibility
np.random.seed(42)

# Sidebar inputs for sample sizes and conversion rates
st.sidebar.header('Input Parameters')
n_A = st.sidebar.number_input('Number of participants in group A', value=1000, min_value=1)
n_B = st.sidebar.number_input('Number of participants in group B', value=1000, min_value=1)
p_A = st.sidebar.slider('Conversion rate for group A', 0.0, 1.0, 0.08)
p_B = st.sidebar.slider('Conversion rate for group B', 0.0, 1.0, 0.095)

# Generate conversions (1 = conversion, 0 = no conversion)
conversions_A = np.random.binomial(1, p_A, n_A)
conversions_B = np.random.binomial(1, p_B, n_B)

# Create data frames
data_A = pd.DataFrame({'version': 'A', 'converted': conversions_A})
data_B = pd.DataFrame({'version': 'B', 'converted': conversions_B})

# Combine data frames
data = pd.concat([data_A, data_B]).reset_index(drop=True)

# Calculate the number of conversions and total observations for each version
summary = data.groupby('version')['converted'].agg(['sum', 'count'])
summary.columns = ['conversions', 'total']

# Display summary table
st.write('### Summary of Conversions')
st.dataframe(summary)

# Perform the Z-test
conversions = [summary.loc['A', 'conversions'], summary.loc['B', 'conversions']]
nobs = [summary.loc['A', 'total'], summary.loc['B', 'total']]
z_stat, p_value = proportions_ztest(conversions, nobs, alternative='larger')

# Display z-statistic and p-value
st.write(f"Z-statistic: {z_stat:.4f}")
st.write(f"P-value: {p_value:.4f}")

# Set significance level
alpha = 0.05  # Significance level

# Interpretation of results
if p_value < alpha:
    st.write("Reject the null hypothesis. Version B is better.")
else:
    st.write("Fail to reject the null hypothesis. No significant difference.")

# Display a plot of the conversion rates
st.write("### Conversion Rates")
conversion_rates = [p_A, p_B]
labels = ['Group A', 'Group B']

fig, ax = plt.subplots()
sns.barplot(x=labels, y=conversion_rates, ax=ax)
ax.set_title('Conversion Rates for Group A and Group B')
ax.set_ylabel('Conversion Rate')

st.pyplot(fig)