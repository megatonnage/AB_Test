import streamlit as st
import numpy as np
import pandas as pd
from statsmodels.stats.proportion import proportions_ztest
import matplotlib.pyplot as plt
import seaborn as sns

# Streamlit app
st.title('AB Testing with Z-Test')

# Sidebar inputs for sample sizes and conversion rates
st.sidebar.header('Input Parameters')
n_A = st.sidebar.number_input('Number of participants in group A', min_value=0, step=1)
n_B = st.sidebar.number_input('Number of participants in group B', min_value=0, step=1)
p_A = st.sidebar.slider('Conversion rate for group A', 0.0, 1.0, 0.1)
p_B = st.sidebar.slider('Conversion rate for group B', 0.0, 1.0, 0.1)

# Calculate conversion counts
conversions_A = int(p_A * n_A)
conversions_B = int(p_B * n_B)

# Create data frames, using lists to avoid the scalar value issue
data_A = pd.DataFrame({'version': ['A'], 'converted': [conversions_A], 'total': [n_A]})
data_B = pd.DataFrame({'version': ['B'], 'converted': [conversions_B], 'total': [n_B]})

# Combine data frames
data = pd.concat([data_A, data_B]).reset_index(drop=True)

# Display summary table
st.write('### Summary of Conversions')
st.dataframe(data)

# Perform the Z-test (switch order of inputs to make Group B compared as larger)
conversions = [conversions_B, conversions_A]
nobs = [n_B, n_A]
z_stat, p_value = proportions_ztest(conversions, nobs, alternative='larger')

# Display z-statistic and p-value
st.write(f"Z-statistic: {z_stat:.4f}")
st.write(f"P-value: {p_value:.4f}")

# Set significance level
alpha = 0.05  # Significance level

# Interpretation of results
if p_value < alpha:
    st.write("Reject the null hypothesis. There is a significant difference.")
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
