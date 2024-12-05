import altair as alt
import pandas as pd
import streamlit as st

# File path
file_path = '/home/genna/yc/YC_Chatbot/word_frequencies_per_code.csv'

# Title for the app
st.title("Word Frequency Visualization by Code")

# Load the CSV file directly
try:
    data = pd.read_csv(file_path)

    # Show raw data
    st.subheader("Raw Data")
    st.dataframe(data)

    # Group data by 'Code'
    codes = data['Code'].unique()

    # Generate charts with individual sliders and tables
    st.subheader("Word Frequency Charts and Tables")
    for code in codes:
        st.write(f"### {code}")

        # Filter data for the specific code
        code_data = data[data['Code'] == code]

        # Add a slider to filter by frequency for this code
        min_freq, max_freq = int(code_data['Frequency'].min()), int(code_data['Frequency'].max())
        freq_threshold = st.slider(
            f"Set Minimum Frequency for {code}",
            min_value=min_freq,
            max_value=max_freq,
            value=min_freq,
            key=f"{code}_slider"
        )

        # Filter the data for this code based on the slider value
        filtered_code_data = code_data[code_data['Frequency'] >= freq_threshold].sort_values(
            by='Frequency', ascending=False
        )

        if filtered_code_data.empty:
            st.warning(f"No data matches the frequency filter for {code}.")
        else:
            # Create a bar chart for the specific code
            chart = alt.Chart(filtered_code_data).mark_bar().encode(
                x=alt.X('Word:N', sort='-y', title='Word'),
                y=alt.Y('Frequency:Q', title='Frequency'),
                color=alt.value('#4C78A8'),  # Set a single color for consistency
                tooltip=['Word', 'Frequency']
            ).properties(
                width=700,
                height=400,
                title=f"Word Frequencies for {code}"
            )

            st.altair_chart(chart)

            # Display a table for this code
            st.write(f"#### Data Table for {code}")
            st.dataframe(filtered_code_data.reset_index(drop=True))

    # Optional: Display a summary
    st.subheader("Overall Summary")
    st.write(f"Total Unique Codes: {len(codes)}")
    st.write(f"Total Words: {len(data)}")
    st.write(f"Total Frequency: {data['Frequency'].sum()}")

except FileNotFoundError:
    st.error(f"File not found at {file_path}. Please check the path and try again.")
