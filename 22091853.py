import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from scipy.optimize import curve_fit


def exponential_growth(x, a, b, c):
    """
    Exponential growth model function.

    Parameters:
    - x (array): Input array representing the independent variable (e.g., time).
    - a (float): Coefficient determining the amplitude of the exponential growth.
    - b (float): Coefficient determining the growth rate.
    - c (float): Coefficient determining the vertical shift of the curve.

    Returns:
    - array: Output array representing the values of the exponential growth model.
    """
    return a * np.exp(b * x) + c


def visualize_clusters_and_growth(df):
    """
    Perform clustering and visualize clustered data, as well as fit an exponential growth model.

    Parameters:
    - df (DataFrame): Input DataFrame containing relevant data columns.

    Returns:
    - None: Displays visualizations using Matplotlib and Seaborn.
    """
    # Select relevant columns for clustering
    selected_columns = ['Annual CO₂ emissions (per capita)', 'GDP per capita']

    # Drop rows with missing values in selected columns
    df_selected = df.dropna(subset=selected_columns)

    # Normalize the data
    scaler = StandardScaler()
    df_normalized = scaler.fit_transform(df_selected[selected_columns])

    # Apply K-means clustering
    num_clusters = 3  # Adjust the number of clusters based on your requirements
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df_selected['cluster'] = kmeans.fit_predict(df_normalized)

    # Visualize the clustered data
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    axes[0].scatter(df_selected[selected_columns[0]],
                    df_selected[selected_columns[1]], c=df_selected['cluster'], cmap='viridis')
    axes[0].set_title('Annual CO₂ emissions vs. GDP per capita')
    axes[0].set_xlabel(selected_columns[0])
    axes[0].set_ylabel(selected_columns[1])

    # ... (similar code for the other two plots)

    plt.show()

    # Seaborn plots
    plt.scatter(df_selected['Annual CO₂ emissions (per capita)'],
                df_selected['GDP per capita'], c=df_selected['cluster'], cmap='viridis')
    plt.xlabel('Annual CO₂ emissions (per capita)')
    plt.ylabel('GDP per capita')
    plt.title('Clustered Data: CO₂ Emissions vs. GDP per Capita')
    plt.colorbar(label='Cluster')
    plt.show()

    # Exponential Growth Model Fitting
    x_data = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    y_data = np.array([2, 4, 8, 20, 50, 100, 150, 250, 400, 600])

    params, covariance = curve_fit(exponential_growth, x_data, y_data)
    a, b, c = params

    future_years = np.arange(11, 21)
    predicted_values = exponential_growth(future_years, a, b, c)

    param_errors = np.sqrt(np.diag(covariance))
    confidence_interval = 1.96

    lower_bound = np.array([exponential_growth(
        x, a - param_errors[0], b - param_errors[1], c - param_errors[2]) for x in future_years])
    upper_bound = np.array([exponential_growth(
        x, a + param_errors[0], b + param_errors[1], c + param_errors[2]) for x in future_years])

    plt.scatter(x_data, y_data, label='Actual Data')
    plt.plot(future_years, predicted_values,
             label='Predicted Values', color='green')
    plt.fill_between(future_years, lower_bound, upper_bound,
                     color='lightgray', label='Confidence Interval', alpha=0.5)
    plt.xlabel('Years')
    plt.ylabel('Y Values')
    plt.title('Exponential Growth Model Fitting with Confidence Interval')
    plt.legend()
    plt.show()

    # Continuing with Further Clustering and Time-Series Analysis

    # Select relevant columns for clustering
    selected_columns = [
        'Annual CO₂ emissions (per capita)', 'GDP per capita', 'Population (historical estimates)']

    # Drop rows with missing values in the selected columns
    df = df.dropna(subset=selected_columns)

    # Normalize only numeric columns
    scaler = StandardScaler()
    df[selected_columns] = scaler.fit_transform(df[selected_columns])

    # Check if 'cluster' column exists before dropping it
    if 'cluster' in df.columns:
        df = df.drop('cluster', axis=1)

    # Apply K-means clustering
    num_clusters = 3  # Adjust the number of clusters based on your requirements
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    df['cluster'] = kmeans.fit_predict(df[selected_columns])

    # Verify that the 'cluster' column is added correctly
    print(df.head())

    # Select one country from each cluster for comparison
    countries_per_cluster = df.groupby('cluster').head(1)

    # Extract relevant indicators for time-series analysis
    selected_columns = [
        'Year', 'Annual CO₂ emissions (per capita)', 'GDP per capita', 'Entity', 'cluster']
    countries_data = df[df['Entity'].isin(
        countries_per_cluster['Entity'])][selected_columns]

    # Plot trends within clusters for CO₂ emissions
    for cluster, cluster_data in countries_data.groupby('cluster'):
        plt.figure(figsize=(10, 6))
        for country, country_data in cluster_data.groupby('Entity'):
            plt.plot(
                country_data['Year'], country_data['Annual CO₂ emissions (per capita)'], label=country)

        plt.title(f'Trends in Cluster {cluster} - CO₂ Emissions')
        plt.xlabel('Year')
        plt.ylabel('CO₂ Emissions (per capita)')
        plt.legend()
        plt.show()

    # Plot trends within clusters for GDP per capita
    for cluster, cluster_data in countries_data.groupby('cluster'):
        plt.figure(figsize=(10, 6))
        for country, country_data in cluster_data.groupby('Entity'):
            plt.plot(country_data['Year'],
                     country_data['GDP per capita'], label=country)

        plt.title(f'Trends in Cluster {cluster} - GDP per Capita')
        plt.xlabel('Year')
        plt.ylabel('GDP per Capita')
        plt.legend()
        plt.show()


# Load dataset or create a sample one for demonstration
data = {
    'Entity': ['Abkhazia', 'Afghanistan', 'Afghanistan', 'Afghanistan', 'Afghanistan'],
    'Code': ['OWID_ABK', 'AFG', 'AFG', 'AFG', 'AFG'],
    'Year': [2015, 1949, 1950, 1951, 1952],
    'Annual CO₂ emissions (per capita)': [None, 0.001992, 0.011266, 0.012098, 0.011946],
    'GDP per capita': [None, None, 1156.0, 1170.0, 1189.0],
    '417485-annotations': [None, None, None, None, None],
    'Population (historical estimates)': [None, 7356890.0, 7480464.0, 7571542.0, 7667534.0],
    'Continent': ['Asia', None, None, None, None]
}

df = pd.DataFrame(data)

# Call the function with the DataFrame
visualize_clusters_and_growth(df)
