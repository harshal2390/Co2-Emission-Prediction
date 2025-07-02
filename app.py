# Importing libraries-----------------------------------------------------------------------------------------
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import pickle
from scipy import stats
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
# Creating Sidebar-------------------------------------------------------------------------------------------
with st.sidebar:
    st.markdown("# CO2 Emissions by Vehicle")
    user_input = st.selectbox('Please select',('Visulization','Model'))

# Load the vehicle dataset
df = pd.read_csv('co2 Emissions.csv')

# Drop rows with natural gas as fuel type
fuel_type_mapping = {"Z": "Premium Gasoline","X": "Regular Gasoline","D": "Diesel","E": "Ethanol(E85)","N": "Natural Gas"}
df["Fuel Type"] = df["Fuel Type"].map(fuel_type_mapping)
df_natural = df[~df["Fuel Type"].str.contains("Natural Gas")].reset_index(drop=True)

# Remove outliers from the data
df_new = df_natural[['Engine Size(L)', 'Cylinders', 'Fuel Consumption Comb (L/100 km)', 'CO2 Emissions(g/km)']]
df_new_model = df_new[(np.abs(stats.zscore(df_new)) < 1.9).all(axis=1)]

# Visulization-------------------------------------------------------------------------------------------------
if user_input == 'Visulization':



    # Showing Dataset------------------------------------------------------------------------------------------
    st.title('CO2 Emissions by Vehicle')
    st.header("Data We collected from the source")
    st.write(df)

    # Brands of Cars-------------------------------------------------------------------------------------------
    st.subheader('Brands of Cars')
    df_brand = df['Make'].value_counts().reset_index().rename(columns={'count': 'Count'})
    fig1, ax1 = plt.subplots(figsize=(15, 6))
    sns.barplot(data=df_brand, x="Make", y="Count", ax=ax1)
    ax1.tick_params(axis='x', labelrotation=75)
    ax1.set_title("All Car Companies and their Cars")
    ax1.set_xlabel("Companies")
    ax1.set_ylabel("Cars")
    ax1.bar_label(ax1.containers[0], fontsize=7)
    st.pyplot(fig1)
    st.write(df_brand)


    # Top 25 Models of Cars------------------------------------------------------------------------------------
    st.subheader('Top 25 Models of Cars')
    df_model = df['Model'].value_counts().reset_index().rename(columns={'count':'Count'})
    fig2 = plt.figure(figsize=(20, 6))  # assign the figure itself to fig2
    sns.barplot(data=df_model[:25], x="Model", y="Count")
    plt.xticks(rotation=75)
    plt.title("Top 25 Car Models")
    plt.xlabel("Models")
    plt.ylabel("Cars")
    plt.bar_label(plt.gca().containers[0])  # use plt.gca() since fig2 is a Figure, not Axes
    st.pyplot(fig2)  # pass the fig here
    st.write(df_model)


    # Vehicle Class--------------------------------------------------------------------------------------------
    st.subheader('Vehicle Class')
    df_vehicle_class = df['Vehicle Class'].value_counts().reset_index().rename(columns={'count': 'Count'})
    fig3, ax3 = plt.subplots(figsize=(20, 5))
    sns.barplot(data=df_vehicle_class, x="Vehicle Class", y="Count", ax=ax3)
    ax3.tick_params(axis='x', labelrotation=75)
    ax3.set_title("All Vehicle Class")
    ax3.set_xlabel("Vehicle Class")
    ax3.set_ylabel("Cars")
    ax3.bar_label(ax3.containers[0])
    st.pyplot(fig3)
    st.write(df_vehicle_class)


    # Engine Sizes of Cars-------------------------------------------------------------------------------------
    st.subheader('Engine Sizes of Cars')
    df_engine_size = df['Engine Size(L)'].value_counts().reset_index().rename(columns={'count': 'Count'})
    fig4, ax4 = plt.subplots(figsize=(20, 6))
    sns.barplot(data=df_engine_size, x="Engine Size(L)", y="Count", ax=ax4)
    ax4.tick_params(axis='x', labelrotation=90)
    ax4.set_title("All Engine Sizes")
    ax4.set_xlabel("Engine Size(L)")
    ax4.set_ylabel("Cars")
    ax4.bar_label(ax4.containers[0])
    st.pyplot(fig4)
    st.write(df_engine_size)


    # Cylinders-----------------------------------------------------------------------------------------------
    st.subheader('Cylinders')
    df_cylinders = df['Cylinders'].value_counts().reset_index().rename(columns={'count': 'Count'})
    fig5, ax5 = plt.subplots(figsize=(20, 6))
    sns.barplot(data=df_cylinders, x="Cylinders", y="Count", ax=ax5)
    ax5.tick_params(axis='x', labelrotation=90)
    ax5.set_title("All Cylinders")
    ax5.set_xlabel("Cylinders")
    ax5.set_ylabel("Cars")
    ax5.bar_label(ax5.containers[0])
    st.pyplot(fig5)
    st.write(df_cylinders)


    # Transmission of Cars------------------------------------------------------------------------------------
    transmission_mapping = {
    "A4": "Automatic", "A5": "Automatic", "A6": "Automatic", "A7": "Automatic", "A8": "Automatic", "A9": "Automatic", "A10": "Automatic",
    "AM5": "Automated Manual", "AM6": "Automated Manual", "AM7": "Automated Manual", "AM8": "Automated Manual", "AM9": "Automated Manual",
    "AS4": "Automatic with Select Shift", "AS5": "Automatic with Select Shift", "AS6": "Automatic with Select Shift", "AS7": "Automatic with Select Shift", "AS8": "Automatic with Select Shift", "AS9": "Automatic with Select Shift", "AS10": "Automatic with Select Shift",
    "AV": "Continuously Variable", "AV6": "Continuously Variable", "AV7": "Continuously Variable", "AV8": "Continuously Variable", "AV10": "Continuously Variable",
    "M5": "Manual", "M6": "Manual", "M7": "Manual"}
    df["Transmission"] = df["Transmission"].map(transmission_mapping)
    st.subheader('Transmission')
    df_transmission = df['Transmission'].value_counts().reset_index().rename(columns={'count': 'Count'})
    fig6, ax6 = plt.subplots(figsize=(20, 5))
    sns.barplot(data=df_transmission, x="Transmission", y="Count", ax=ax6)
    ax6.set_title("All Transmissions")
    ax6.set_xlabel("Transmissions")
    ax6.set_ylabel("Cars")
    ax6.bar_label(ax6.containers[0])
    st.pyplot(fig6)
    st.write(df_transmission)


    # Fuel Type of Cars--------------------------------------------------------------------------------------
    st.subheader('Fuel Type')
    df_fuel_type = df['Fuel Type'].value_counts().reset_index().rename(columns={'count': 'Count'})
    fig7, ax7 = plt.subplots(figsize=(20, 5))
    sns.barplot(data=df_fuel_type, x="Fuel Type", y="Count", ax=ax7)
    ax7.set_title("All Fuel Types")
    ax7.set_xlabel("Fuel Types")
    ax7.set_ylabel("Cars")
    ax7.bar_label(ax7.containers[0])
    st.pyplot(fig7)
    st.text("We have only one data on natural gas. So we cannot predict anything using only one data. That's why we have to drop this row.")
    st.write(df_fuel_type)


    # Removing Natural Gas-----------------------------------------------------------------------------------
    st.subheader('After removing Natural Gas data')
    df_ftype = df_natural['Fuel Type'].value_counts().reset_index().rename(columns={'count': 'Count'})
    fig8, ax8 = plt.subplots(figsize=(20, 5))
    sns.barplot(data=df_ftype, x="Fuel Type", y="Count", ax=ax8)
    ax8.set_title("All Fuel Types")
    ax8.set_xlabel("Fuel Types")
    ax8.set_ylabel("Cars")
    ax8.bar_label(ax8.containers[0])
    st.pyplot(fig8)
    st.write(df_ftype)


    # CO2 Emission variation with Brand----------------------------------------------------------------------
    st.header('Variation in CO2 emissions with different features')
    st.subheader('CO2 Emission with Brand ')
    df_co2_make = df.groupby(['Make'])['CO2 Emissions(g/km)'].mean().sort_values().reset_index()
    fig8, ax8 = plt.subplots(figsize=(20, 5))
    sns.barplot(data=df_co2_make, x="Make", y="CO2 Emissions(g/km)", ax=ax8)
    ax3.tick_params(axis='x', labelrotation=90)
    ax8.set_title("CO2 Emissions variation with Brand")
    ax8.set_xlabel("Brands")
    ax8.set_ylabel("CO2 Emissions(g/km)")
    ax8.bar_label(ax8.containers[0], fontsize=8, fmt='%.1f')
    st.pyplot(fig8)

    def plot_bar(data, x_label, y_label, title):
        fig, ax = plt.subplots(figsize=(23, 5))
        sns.barplot(data=data, x=x_label, y=y_label, ax=ax)
        ax.tick_params(axis='x', labelrotation=90)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.bar_label(ax.containers[0], fontsize=9)
        return fig


    st.subheader('CO2 Emissions variation with Vehicle Class')
    df_co2_vehicle_class = df.groupby(['Vehicle Class'])['CO2 Emissions(g/km)'].mean().sort_values().reset_index()
    fig_vclass = plot_bar(df_co2_vehicle_class, "Vehicle Class", "CO2 Emissions(g/km)", "CO2 Emissions variation with Vehicle Class")
    st.pyplot(fig_vclass)

    st.subheader('CO2 Emission variation with Transmission')
    df_co2_transmission = df.groupby(['Transmission'])['CO2 Emissions(g/km)'].mean().sort_values().reset_index()
    fig_trans = plot_bar(df_co2_transmission, "Transmission", "CO2 Emissions(g/km)", "CO2 Emission variation with Transmission")
    st.pyplot(fig_trans)

    st.subheader('CO2 Emissions variation with Fuel Type')
    df_co2_fuel_type = df.groupby(['Fuel Type'])['CO2 Emissions(g/km)'].mean().sort_values().reset_index()
    fig_fuel = plot_bar(df_co2_fuel_type, "Fuel Type", "CO2 Emissions(g/km)", "CO2 Emissions variation with Fuel Type")
    st.pyplot(fig_fuel)


    # Box Plots-------------------------------------------------------------------------------------------
    st.header("Box Plots")
    fig_box1, axs1 = plt.subplots(2, 2, figsize=(20, 10))
    features = ['Engine Size(L)', 'Cylinders', 'Fuel Consumption Comb (L/100 km)', 'CO2 Emissions(g/km)']
    for i, feature in enumerate(features):
        row, col = divmod(i, 2)
        axs1[row, col].boxplot(df_new[feature])
        axs1[row, col].set_title(feature)
    st.pyplot(fig_box1)

    st.text("As we can see there are some outliers present in our Dataset")
    st.subheader("After removing outliers")
    st.write("Before removing outliers we have", len(df), "data")
    st.write("After removing outliers we have", len(df_new_model), "data")

    st.subheader("Boxplot after removing outliers")
    fig_box2, axs2 = plt.subplots(2, 2, figsize=(20, 10))
    for i, feature in enumerate(features):
        row, col = divmod(i, 2)
        axs2[row, col].boxplot(df_new_model[feature])
        axs2[row, col].set_title(feature)
    st.pyplot(fig_box2)



else:
    
    X = df_new_model[['Engine Size(L)', 'Cylinders', 'Fuel Consumption Comb (L/100 km)']]
    y = df_new_model['CO2 Emissions(g/km)']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    st.title('CO2 Emission Prediction')
    st.write('Enter the vehicle specifications to predict CO2 emissions.')

    if st.checkbox("Show model evaluation metrics"):
        st.subheader("Model Evaluation:")
        st.write(f"R² Score: {r2_score(y_test, y_pred):.2f}")
        st.write(f"Mean Absolute Error (MAE): {mean_absolute_error(y_test, y_pred):.2f}")
        st.write(f"Mean Squared Error (MSE): {mean_squared_error(y_test, y_pred):.2f}")

    engine_size = st.number_input('Engine Size(L)', step=0.1, format="%.1f")
    cylinders = st.number_input('Cylinders', min_value=2, max_value=16, step=1)
    fuel_consumption = st.number_input('Fuel Consumption Comb (L/100 km)', step=0.1, format="%.1f")

    input_data = pd.DataFrame([[engine_size, cylinders, fuel_consumption]],
                              columns=['Engine Size(L)', 'Cylinders', 'Fuel Consumption Comb (L/100 km)'])

    predicted_co2 = model.predict(input_data)

    st.subheader("Prediction Result:")
    st.write(f'Predicted CO2 Emissions: {predicted_co2[0]:.2f} g/km')
