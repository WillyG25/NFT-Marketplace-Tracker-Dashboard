import streamlit as st
from PIL import Image
import pandas as pd
import altair as alt
from user_manager import UserManager
from database_manager import DatabaseManager
from sqlalchemy import create_engine
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from datetime import datetime
import mysql.connector
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
import base64

class DataAnalysis:
    def __init__(self, db):
        self.db = db

    def descriptive_analysis(self):
        st.subheader("Descriptive Analysis for Marketplaces")

        # Fetch and display summary statistics for each numeric column in the Marketplaces table
        marketplaces_query = "SELECT * FROM Marketplaces"
        marketplaces_data = self.db.fetch_all(marketplaces_query)
        marketplaces_df = pd.DataFrame(marketplaces_data, columns=["Marketplace_ID", "Marketplace", "Collection_Name", "Total_Supply", "Total_Volume", "Market_Cap", "ThirtyDayvolume", "Update_Date", "chain_name"])

        # Fill or replace None values with 0
        marketplaces_df.fillna(0, inplace=True)

        # Convert relevant columns to integers
        columns_to_convert = ["Total_Volume", "Market_Cap", "ThirtyDayvolume"]
        marketplaces_df[columns_to_convert] = marketplaces_df[columns_to_convert].astype(int)

        st.write("Summary Statistics for Marketplaces:")
        st.write(marketplaces_df.describe())
        
        # Display summary statistics including count for values equal to 0
        summary_stats = marketplaces_df.describe()
        count_zeros = marketplaces_df[["Total_Volume", "Market_Cap", "ThirtyDayvolume"]].eq(0).sum()
        summary_stats.loc['count_zeros'] = count_zeros
        st.write(summary_stats)


        # Use histograms to visualize the distribution of numeric columns
        st.write("Histograms for Numeric Columns:")
        numeric_columns = ["Total_Volume", "ThirtyDayvolume", "Market_Cap"]

        for column in numeric_columns:
            # Plot histogram for values equal to 0
            fig, ax = plt.subplots(figsize=(8, 6))
            n, bins, patches = ax.hist(marketplaces_df[column][marketplaces_df[column] == 0], bins=20, color='skyblue', edgecolor='black', label='= 0')
            ax.set_title(f'Histogram for {column} (= 0)')
            ax.set_xlabel(column)
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True)

            # Annotate each bin with its frequency
            for bin, patch in zip(bins, patches):
                height = patch.get_height()
                ax.text(bin + 0.5, height, str(int(height)), ha='center', va='bottom')

            st.pyplot(fig)

            # Plot histogram for values above 0 and less than or equal to 100
            fig, ax = plt.subplots(figsize=(8, 6))
            n, bins, patches = ax.hist(marketplaces_df[column][(marketplaces_df[column] > 0) & (marketplaces_df[column] <= 100)], bins=20, color='green', edgecolor='black', label='0 < x <= 100')
            ax.set_title(f'Histogram for {column} (0 < x <= 100)')
            ax.set_xlabel(column)
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True)

            # Annotate each bin with its frequency
            for bin, patch in zip(bins, patches):
                height = patch.get_height()
                ax.text(bin + 2, height, str(int(height)), ha='center', va='bottom')

            st.pyplot(fig)

            # Plot histogram for values above 100 and less than or equal to 1000
            fig, ax = plt.subplots(figsize=(8, 6))
            n, bins, patches = ax.hist(marketplaces_df[column][(marketplaces_df[column] > 100) & (marketplaces_df[column] <= 1000)], bins=20, color='blue', edgecolor='black', label='100 < x <= 1000')
            ax.set_title(f'Histogram for {column} (100 < x <= 1000)')
            ax.set_xlabel(column)
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True)

            # Annotate each bin with its frequency
            for bin, patch in zip(bins, patches):
                height = patch.get_height()
                ax.text(bin + 20, height, str(int(height)), ha='center', va='bottom')

            st.pyplot(fig)

            # Plot histogram for values above 1000
            fig, ax = plt.subplots(figsize=(8, 6))
            n, bins, patches = ax.hist(marketplaces_df[column][marketplaces_df[column] > 1000], bins=20, color='orange', edgecolor='black', label='> 1000')
            ax.set_title(f'Histogram for {column} (> 1000)')
            ax.set_xlabel(column)
            ax.set_ylabel('Frequency')
            ax.legend()
            ax.grid(True)

            # Annotate each bin with its frequency
            for bin, patch in zip(bins, patches):
                height = patch.get_height()
                ax.text(bin + 200, height, str(int(height)), ha='center', va='bottom')

            st.pyplot(fig)


    def plot_heatmap(self, correlation_matrix):
        import seaborn as sns
        import matplotlib.pyplot as plt

        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
        plt.title('Correlation Matrix')
        st.pyplot(plt)

    def correlation_analysis(self):
        st.subheader("Correlation Analysis")

        # Fetch data relevant to blockchain diversity and marketplace performance
        correlation_query = "SELECT B.Blockchain_ID, M.Total_Volume, M.ThirtyDayvolume FROM Marketplaces M JOIN Marketplace_Blockchain_Association A ON M.Marketplace_ID = A.Marketplace_ID JOIN Blockchains B ON A.Blockchain_ID = B.Blockchain_ID"
        correlation_data = self.db.fetch_all(correlation_query)
        correlation_df = pd.DataFrame(correlation_data, columns=["Blockchain_ID", "Total_Volume", "ThirtyDayvolume"])

        # Calculate the correlation matrix
        correlation_matrix = correlation_df.corr()

        # Display a heatmap to visualize the correlation matrix
        st.write("Correlation Matrix:")
        self.plot_heatmap(correlation_matrix)
        
    def ranking_analysis(self):
        st.subheader("Ranking Analysis")

        # Fetch and display the top 10 collections based on total volume
        top_collections_query = "SELECT Marketplace, Collection_Name, Total_Volume FROM Marketplaces ORDER BY Total_Volume DESC LIMIT 10"
        top_collections_data = self.db.fetch_all(top_collections_query)
        top_collections_df = pd.DataFrame(top_collections_data, columns=["Marketplace", "Collection_Name", "Total_Volume"])

        st.write("Top 10 Collections Based on Total Volume:")
        st.write(top_collections_df)

        # Fetch and display the top 10 collections with the lowest total volume
        lowest_total_volume_query = "SELECT Marketplace, Collection_Name, Total_Volume FROM Marketplaces ORDER BY Total_Volume ASC LIMIT 10"
        lowest_total_volume_data = self.db.fetch_all(lowest_total_volume_query)
        lowest_total_volume_df = pd.DataFrame(lowest_total_volume_data, columns=["Marketplace", "Collection_Name", "Total_Volume"])

        st.write("Top 10 Collections with the Lowest Total Volume:")
        st.write(lowest_total_volume_df)

        # Fetch and display the top 10 collections with the highest market capitalization
        top_market_cap_query = "SELECT Marketplace, Collection_Name, Market_Cap FROM Marketplaces ORDER BY Market_Cap DESC LIMIT 10"
        top_market_cap_data = self.db.fetch_all(top_market_cap_query)
        top_market_cap_df = pd.DataFrame(top_market_cap_data, columns=["Marketplace", "Collection_Name", "Market_Cap"])

        st.write("Top 10 Collections Based on Market Capitalization:")
        st.write(top_market_cap_df)


        # Fetch and display the top 10 collections with the highest thirty-day volume
        top_thirty_day_volume_query = "SELECT Marketplace, Collection_Name, ThirtyDayvolume FROM Marketplaces ORDER BY ThirtyDayvolume DESC LIMIT 10"
        top_thirty_day_volume_data = self.db.fetch_all(top_thirty_day_volume_query)
        top_thirty_day_volume_df = pd.DataFrame(top_thirty_day_volume_data, columns=["Marketplace", "Collection_Name", "ThirtyDayvolume"])

        st.write("Top 10 Collections Based on Thirty-Day Volume:")
        st.write(top_thirty_day_volume_df)

        # Fetch and display the top 10 collections with the highest price
        top_price_query = """
        SELECT M.Marketplace, C.Name, MAX(P.Value) AS MaxPrice
        FROM Prices P
        JOIN Collections C ON P.NFT_Address = C.Contract_Address
        JOIN Marketplaces M ON C.Name = M.Collection_Name
        GROUP BY M.Marketplace, C.Name
        ORDER BY MaxPrice DESC
        LIMIT 10
        """
        top_price_data = self.db.fetch_all(top_price_query)
        top_price_df = pd.DataFrame(top_price_data, columns=["Marketplace", "Name", "Value"])

        st.write("Top 10 Collections Based on Price:")
        st.write(top_price_df)



    def categorical_analysis(self):
        st.subheader("Categorical Analysis")

        # Fetch and display the distribution of collections across different marketplaces
        categorical_query = "SELECT Marketplace, COUNT(DISTINCT Collection_Name) AS Unique_Collections FROM Marketplaces GROUP BY Marketplace"
        categorical_data = self.db.fetch_all(categorical_query)
        categorical_df = pd.DataFrame(categorical_data, columns=["Marketplace", "Unique_Collections"])

        st.write("Distribution of Collections Across Marketplaces:")
        st.write(categorical_df)

        # Query to find how many marketplaces are linked to each chain_name
        chain_marketplace_query = """
            SELECT DISTINCT m.Marketplace, m.chain_name,
                   CASE WHEN b.name IS NOT NULL THEN 'Yes' ELSE 'No' END AS Linked
            FROM Marketplaces m
            LEFT JOIN Blockchains b ON m.chain_name = b.name
        """
        chain_marketplace_data = self.db.fetch_all(chain_marketplace_query)
        chain_marketplace_df = pd.DataFrame(chain_marketplace_data, columns=["Marketplace", "chain_name", "Linked"])

        st.write("Marketplaces Linked to Each Chain:")
        st.write(chain_marketplace_df)
        
    def performance_metrics_analysis(self):
        st.subheader("Performance Metrics Analysis")

        # Fetch and display the top 10 collections based on thirty-day volume
        top_thirty_day_volume_query = """
        SELECT Collection_Name, SUM(ThirtyDayvolume) AS ThirtyDayVolume
        FROM Marketplaces
        GROUP BY Collection_Name
        ORDER BY ThirtyDayVolume DESC
        LIMIT 10
        """
        top_thirty_day_volume_data = self.db.fetch_all(top_thirty_day_volume_query)
        top_thirty_day_volume_df = pd.DataFrame(top_thirty_day_volume_data, columns=["Collection_Name", "ThirtyDayVolume"])

        st.write("Top 10 Collections Based on Thirty-Day Volume:")
        self.plot_bar_chart(top_thirty_day_volume_df, "Collection_Name", "ThirtyDayVolume")

        # Fetch and display the top 10 collections based on total volume
        top_total_volume_query = """
        SELECT Collection_Name, SUM(Total_Volume) AS TotalVolume
        FROM Marketplaces
        GROUP BY Collection_Name
        ORDER BY TotalVolume DESC
        LIMIT 10
        """
        top_total_volume_data = self.db.fetch_all(top_total_volume_query)
        top_total_volume_df = pd.DataFrame(top_total_volume_data, columns=["Collection_Name", "TotalVolume"])

        st.write("Top 10 Collections Based on Total Volume:")
        self.plot_bar_chart(top_total_volume_df, "Collection_Name", "TotalVolume")

        # Fetch and display all collections in a scatter plot based on total volume and thirty-day volume
        all_metrics_query = "SELECT Total_Volume, ThirtyDayvolume FROM Marketplaces"
        all_metrics_data = self.db.fetch_all(all_metrics_query)
        all_metrics_df = pd.DataFrame(all_metrics_data, columns=["Total_Volume", "ThirtyDayvolume"])

        st.write("All Collections Scatter Plot based on Total Volume and Thirty-Day Volume:")
        self.plot_scatter_chart(all_metrics_df, "Total_Volume", "ThirtyDayvolume")

    @staticmethod
    def plot_bar_chart(df, x_column, y_column):
        # Add your code for plotting bar charts here
        # You can use libraries like Matplotlib, Seaborn, or Altair for plotting
        # For example, using Matplotlib:
        import matplotlib.pyplot as plt

        plt.bar(df[x_column], df[y_column], color='skyblue')
        plt.title(f'Bar Chart for {y_column} by {x_column}')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.xticks(rotation=45, ha='right')
        st.pyplot(plt)

    @staticmethod
    def plot_bar_chart(df, x_column, y_column):
        # Add your code for plotting bar charts here
        # You can use libraries like Matplotlib, Seaborn, or Altair for plotting
        # For example, using Matplotlib:
        import matplotlib.pyplot as plt

        plt.bar(df[x_column], df[y_column], color='red')
        plt.title(f'Bar Chart for {y_column} by {x_column}')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.xticks(rotation=45, ha='right')
        st.pyplot(plt)

        
    @staticmethod
    def plot_scatter_chart(df, x_column, y_column):
        plt.figure(figsize=(8, 6))
        plt.scatter(df[x_column], df[y_column], color='green', alpha=0.5)
        plt.title(f'Scatter Plot for {y_column} by {x_column}')
        plt.xlabel(x_column)
        plt.ylabel(y_column)
        plt.xticks(rotation=45, ha='right')
        st.pyplot(plt)


class App:
    def __init__(self, session_state):
        self.db = DatabaseManager(
            host="localhost",
            user="root",
            password="Wirryjayee1",
            database="nftmarketplacetracker"
        )
        self.user_manager = UserManager(self.db, session_state)
        self.data_analysis = DataAnalysis(self.db)  # Instantiate DataAnalysis class

    def login(self):

        st.markdown('<p style="color: black; font-size: 24px; font-weight: bold;">Welcome to the NFT Marketplace Tracker Dashboard</p>', unsafe_allow_html=True)

            
        username_input = st.text_input("Username", key='username_input', value="")
        password_input = st.text_input("Password", key='password_input', type='password', value="")

        if st.button("Login"):
            self.user_manager.authenticate(username_input, password_input)

        st.write('\n\n\n') 
        signup_button = st.button('Sign Up')
        if signup_button:
             st.session_state.page = 'signup'

      
            
    def signup_page(self):
        st.title('Sign Up')

        # User Information
        username = st.text_input('Username', key='username_input')
        email = st.text_input('Email', key='email_input')
        password = st.text_input('Password', type='password', key='password_input')

        # NFT Information
        st.header('NFT Information')
        NFT_address = st.text_input('Contract Address', key='contract_address_input')
        token_name = st.text_input('Token Name', key='token_name_input')
        address = st.text_input('Address', key='address_input')
        quantity = st.number_input('Quantity', min_value=1, key='quantity_input')
        token_id = st.number_input('Token ID', key='token_id_input')
        image = st.text_input('Image URL', key='image_input')
        chain_name = st.text_input('Chain Name', key='chain_name_input')

        # Collection Information
        st.header('Collection Information')
        collection_name = st.text_input('Collection Name', key='collection_name_input')
        total_supply = st.number_input('Total Supply', min_value=1, key='total_supply_input')

        signup_button = st.button('Sign Up', key='signup_button')

        if signup_button:
            if not username or not email or not password or not NFT_address or not token_name or not address or not collection_name:
                st.error('Please fill out all required fields.')
            else:
                if self.is_valid_email(email):
                    # Step 1: Generate NFT data
                    nft_data = {
                        "Contract_Address": NFT_address,
                        "Token_Name": token_name,
                        "Total_Current_Owners": 1,
                        "Address": address,
                        "Quantity": quantity,
                        "Token_ID": token_id,
                        "Image": image,
                        "chain_name": chain_name
                    }

                    # Step 2: Generate Collection data
                    collection_data = {
                        "Contract_Address": NFT_address,
                        "Name": collection_name,
                        "Total_Supply": total_supply
                    }

                    # Step 3: Sign up the user
                    self.user_manager.signup(username, email, password, nft_data, collection_data)
                    # Redirect the user to the login page or any other desired page
                    st.session_state.page = 'login'
                    
    def get_all_collections(self):
        # Fetch all collection names from the database
        collections_query = "SELECT DISTINCT Collection_Name FROM Marketplaces"
        collections = self.db.fetch_all(collections_query)

        if collections:
            return [row[0] for row in collections]
        else:
            return []
        
    def main_app(self):
        # Custom CSS for sidebar and title
        st.markdown(
            """
            <style>
                .sidebar {
                    background-color: #4CAF50; /* Green */
                    color: white;
                    padding: 15px;
                }
                .sidebar .sidebar-content {
                    color: white;
                }
                .sidebar .sidebar-title {
                    color: white;
                }
                footer {
                    text-align: center;
                    padding: 10px;
                    position: fixed;
                    bottom: 0;
                    width: 100%;
                    background-color: black; /* Change to black */
                    color: white;
                }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.sidebar.title("Tracker Dashboard")

        options = ["Home", "Compare Collections", "Price Predictor", "Blockchain Hub", "Statistical Analyser"]
        choice = st.sidebar.selectbox("Select Option", options)

        # Main content based on the selected option
        if choice == "Home":
            st.title('NFT Marketplace Tracker Dashboard')
            st.write('Browse information from Opensea and LooksRare!')
            selected_marketplace = st.selectbox('Select a marketplace:', ['Opensea', 'Looksrare'])

            if selected_marketplace:
                image_path = f'C:/Users/Wilfred Amoo-G/Desktop/Masters Project/{selected_marketplace.lower()}.png'
                image = Image.open(image_path)
                st.image(image, use_column_width=True)

                selected_collection = self.show_collection_dropdown(selected_marketplace)

                if selected_collection and st.button(f"Show Stats for {selected_collection}"):
                    self.show_collection_stats(selected_collection)

                st.write("---")  # Add a separator between entries
            else:
                st.write("No data found for the selected marketplace.")


            # Price Trends Over Time Analysis
            st.header("Price Trends Over Time")

            # Fetch distinct collection names for the dropdown
            collection_names_query = "SELECT DISTINCT Name FROM Collections"
            collection_names = self.db.fetch_all(collection_names_query)

            if collection_names:
                # Extract the collection names from the result
                collection_names = [row[0] for row in collection_names]

                # Dropdown to select an NFT collection
                selected_collection = st.selectbox("Select NFT Collection", collection_names)

                # Fetch prices data with associated collection names from the Prices and Collections tables
                prices_query = f"""
                    SELECT P.Date, P.Value
                    FROM Prices AS P
                    JOIN Collections AS C ON P.NFT_Address = C.Contract_Address
                    WHERE C.Name = '{selected_collection}'
                """
                prices_data = self.db.fetch_all(prices_query)

                if prices_data:
                    # Create a DataFrame from the fetched data
                    prices_df = pd.DataFrame(prices_data, columns=["Date", "Value"])

                    # Convert 'Date' column to datetime
                    prices_df['Date'] = pd.to_datetime(prices_df['Date'])

                    # Sort the DataFrame by the 'Date' column
                    prices_df = prices_df.sort_values(by='Date')

                    # Display the data (without plotting the chart)
                    st.write("Price data for", selected_collection)
                    st.write(prices_df)

                    # Remove commas and convert "Value" column to numeric
                    prices_df['Value'] = prices_df['Value'].replace('[\$,]', '', regex=True).astype(float)

                    # Plot the price trends using Altair
                    chart = alt.Chart(prices_df).mark_line().encode(
                        x='Date:T',
                        y='Value:Q',
                        tooltip=['Value:Q', 'Date:T']
                    ).properties(width=800, height=500)

                    # Display the chart
                    st.altair_chart(chart, use_container_width=True)
                else:
                    st.warning(f"No prices data found for {selected_collection}.")

                st.write("---")  # Add a separator between entries

            else:
                st.warning("No collection names found.")


            # Marketplace Performance Analysis
            st.header("Marketplace Performance")

            # Fetch data for the selected collection and marketplace
            selected_marketplace = st.selectbox('Select a marketplace:', ['Opensea', 'Looksrare'], key="marketplace_selector")
            selected_collection = st.selectbox('Select a Collection:', collection_names, index=0, key="collection_selector")

            # Dropdown to select which chart to plot
            selected_chart = st.selectbox('Select a chart:', ['Total Volume', 'Thirty Day Volume'], key="chart_selector")

            # Fetch data for the selected collection and marketplace
            marketplaces_query = f"""
                SELECT M.Marketplace, M.Collection_Name, M.Total_Volume, M.ThirtyDayvolume
                FROM Marketplaces AS M
                WHERE M.Marketplace = '{selected_marketplace}' AND M.Collection_Name = '{selected_collection}'
            """
            marketplaces_data = self.db.fetch_all(marketplaces_query)

            if marketplaces_data:
                # Create a DataFrame from the fetched data
                marketplaces_df = pd.DataFrame(marketplaces_data, columns=["Marketplace", "Collection_Name", "Total_Volume", "ThirtyDayvolume"])

                # Display the data before plotting the chart
                st.write("Marketplace Data:", marketplaces_df)

                # Convert "Total Volume" to numeric format (remove commas)
                marketplaces_df['Total_Volume'] = pd.to_numeric(marketplaces_df['Total_Volume'].astype(str).str.replace(',', ''), errors='coerce')

                # Plot the marketplace performance using Altair
                if selected_chart == 'Total Volume':
                    chart_marketplace = alt.Chart(marketplaces_df).mark_bar().encode(
                        x='Collection_Name:N',
                        y='Total_Volume:Q',
                        color='Marketplace:N',
                        tooltip=['Marketplace:N', 'Collection_Name:N', 'Total_Volume:Q']
                    ).properties(width=800, height=500)
                elif selected_chart == 'Thirty Day Volume':
                    chart_marketplace = alt.Chart(marketplaces_df).mark_bar().encode(
                        x='Collection_Name:N',
                        y='ThirtyDayvolume:Q',
                        color='Marketplace:N',
                        tooltip=['Marketplace:N', 'Collection_Name:N', 'ThirtyDayvolume:Q']
                    ).properties(width=800, height=500)

                # Display the chart
                st.altair_chart(chart_marketplace, use_container_width=True)
            else:
                st.warning(f"No data found for {selected_collection} in {selected_marketplace}.")

            st.write("---")  # Add a separator between entries

            

        elif choice == "Compare Collections":
            st.title("Compare Collections")

            # Allow the user to select multiple collections
            selected_collections = st.multiselect("Select Collections to Compare", self.get_all_collections())

            # Allow the user to select the metric for comparison
            metrics_dict = {"Total_Supply": "Total Supply", "Market_Cap": "Market Cap", "ThirtyDayvolume": "Thirty Day Volume", "Total_Volume": "Total Volume"}
            selected_metric = st.selectbox("Select Metric for Comparison", list(metrics_dict.values()))

            # Display bar charts for the selected metric
            if selected_collections:
                data = []

                for collection in selected_collections:
                    # Fetch and display the metric values for each collection
                    metric_key = next(key for key, value in metrics_dict.items() if value == selected_metric)
                    metric_values = self.db.fetch_one(f"SELECT {metric_key} FROM Marketplaces WHERE Collection_Name = '{collection}'")

                    if metric_values and metric_values[0] is not None:
                        data.append({"Collection": collection, "Value": float(metric_values[0])})  # Convert to float
                    else:
                        st.warning(f"No data found for {selected_metric} in {collection}")

                # Convert data to a dataframe
                df = pd.DataFrame(data)

                # Plot the bar chart with logarithmic y-axis scale for values greater than zero
                chart = alt.Chart(df).mark_bar().encode(
                    x='Collection',
                    y=alt.Y('Value', scale=alt.Scale(type='log'), axis=alt.Axis(title=selected_metric)),
                ).transform_filter(alt.datum.Value > 0).properties(width=600, height=400)

                # Display the chart 
                st.altair_chart(chart, use_container_width=True)

        elif choice == "Price Predictor":
            st.title("Price Predictor")
            st.write("Welcome to the Price Predictor! Enter details and get predictions.")
            # Database configuration
            db_config = {
                "host": "localhost",
                "user": "root",
                "password": "Wirryjayee1",
                "database": "nftmarketplacetracker"
            }

            # Load data from the Prices table in the database
            @st.cache_data(show_spinner="Fetching Prices data ...")
            def load_prices_data():
                engine = create_engine(f"mysql+mysqlconnector://{db_config['user']}:{db_config['password']}@{db_config['host']}/{db_config['database']}")
                query_prices = "SELECT * FROM Prices"
                return pd.read_sql(query_prices, engine)

            df_prices = load_prices_data()

            # Load data from the Collections table in the database
            @st.cache_data(show_spinner="Fetching Collections data ...")
            def load_collections_data():
                engine = create_engine(f"mysql+mysqlconnector://{db_config['user']}:{db_config['password']}@{db_config['host']}/{db_config['database']}")
                query_collections = "SELECT * FROM Collections"
                df_collections = pd.read_sql(query_collections, engine)

                # Filter Collections to include only those with entries in Prices
                valid_collections = df_collections[df_collections['Contract_Address'].isin(df_prices['NFT_Address'].unique())]

                return valid_collections

            valid_collections = load_collections_data()

            # Function to preprocess and train the model
            @st.cache_data(show_spinner="Loading Price Predictor ...")
            def preprocess_and_train_model(df):
                # Convert the 'Date' column to datetime format
                df['Date'] = pd.to_datetime(df['Date'])

                # Extract temporal features from the 'Date' column
                df['Year'] = df['Date'].dt.year
                df['Month'] = df['Date'].dt.month
                df['Day'] = df['Date'].dt.day

                # Features (X) and target variable (y)
                X = df[['NFT_Address', 'Year', 'Month', 'Day']]
                y = df['Native_Price']

                # Train-test split
                X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

                # Preprocessing using ColumnTransformer
                preprocessor = ColumnTransformer(
                    transformers=[
                        ('num', StandardScaler(), ['Year', 'Month', 'Day']),
                        ('cat', OneHotEncoder(), ['NFT_Address'])
                    ])

                # Random Forest Model with Pipeline
                model = Pipeline(steps=[
                    ('preprocessor', preprocessor),
                    ('regressor', RandomForestRegressor(random_state=42))
                ])

                # Fit the model
                model.fit(X_train, y_train)

                return model

            # Load the model
            model = preprocess_and_train_model(df_prices)


            # Dropdown for selecting NFT collection using names from Collections table
            selected_collection_name = st.selectbox("Select NFT Collection", valid_collections['Name'])

            # User input for prediction date
            future_date_input = st.date_input("Select future date", datetime.today())

            # Button to trigger prediction
            if st.button("Predict Future Price"):
                # Retrieve the corresponding NFT Address from the Collections table
                selected_nft_address = valid_collections.loc[valid_collections['Name'] == selected_collection_name, 'Contract_Address'].values[0]

                # Prepare input data for prediction
                input_data = pd.DataFrame({
                    'NFT_Address': [selected_nft_address],
                    'Year': [future_date_input.year],
                    'Month': [future_date_input.month],
                    'Day': [future_date_input.day]
                })

                # Make prediction
                predicted_value = model.predict(input_data)

                # Display the result
                st.success(f"Predicted Value for {selected_collection_name} on {future_date_input}: {predicted_value[0]}")


        elif choice == "Blockchain Hub":
            st.title("Marketplaces ")
            st.write("Find out which blockchains host Marketplaces!")

            # Fetch available blockchains from Marketplace_Blockchain_Association
            blockchains_query = "SELECT DISTINCT B.Blockchain_ID, B.Name FROM Marketplace_Blockchain_Association AS A JOIN Blockchains AS B ON A.Blockchain_ID = B.Blockchain_ID"
            blockchains = self.db.fetch_all(blockchains_query)

            if blockchains:
                blockchain_names = [row[1] for row in blockchains]
                selected_blockchain = st.selectbox("Select Blockchain", blockchain_names)

                # Fetch associated marketplaces for the selected blockchain
                marketplaces_query = f"""
                    SELECT DISTINCT M.Marketplace
                    FROM Marketplace_Blockchain_Association AS A
                    JOIN Marketplaces AS M ON A.Marketplace_ID = M.Marketplace_ID
                    JOIN Blockchains AS B ON A.Blockchain_ID = B.Blockchain_ID
                    WHERE B.Name = '{selected_blockchain}'
                """
                associated_marketplaces = self.db.fetch_all(marketplaces_query)

                if associated_marketplaces:
                    st.write(f"Marketplaces under {selected_blockchain}:")
                    for row in associated_marketplaces:
                        st.write(row[0])
                else:
                    st.warning(f"No marketplaces found for {selected_blockchain}")
            else:
                st.warning("No blockchains found.")

            st.title("NFTs ")
            st.write("Find out which blockchains host NFTs!")
            # Fetch available blockchains from NFT_Blockchain_Association
            blockchains_query_nft = "SELECT DISTINCT B.Blockchain_ID, B.Name FROM NFT_Blockchain_Association AS A JOIN Blockchains AS B ON A.Blockchain_ID = B.Blockchain_ID"
            blockchains_nft = self.db.fetch_all(blockchains_query_nft)

            if blockchains_nft:
                blockchain_names_nft = [row[1] for row in blockchains_nft]
                selected_blockchain_nft = st.selectbox("Select Blockchain", blockchain_names_nft, key='blockchain_nft')

                # Fetch associated NFTs from NFT_Blockchain_Association
                nfts_query = f"""
                    SELECT DISTINCT A.NFT_ID
                    FROM NFT_Blockchain_Association AS A
                    JOIN Blockchains AS B ON A.Blockchain_ID = B.Blockchain_ID
                    WHERE B.Name = '{selected_blockchain_nft}'
                """
                associated_nfts = self.db.fetch_all(nfts_query)

                if associated_nfts:
                    # Extract NFT_IDs from the result
                    nft_ids = [row[0] for row in associated_nfts]

                    # Fetch DISTINCT Contract_Address for each NFT_ID from NFTs table
                    contract_addresses_query = f"""
                        SELECT DISTINCT N.Contract_Address
                        FROM NFTs AS N
                        WHERE N.NFT_ID IN ({', '.join(map(str, nft_ids))})
                    """
                    contract_addresses = self.db.fetch_all(contract_addresses_query)

                    if contract_addresses:
                        # Extract DISTINCT Contract_Addresses from the result
                        contract_addresses = [row[0] for row in contract_addresses]

                        # Display the contract addresses
                        st.write(f"Contract Addresses associated with {selected_blockchain_nft}:")
                        for address in contract_addresses:
                            st.write(f"{address}")
                    else:
                        st.warning(f"No Contract Addresses found for {selected_blockchain_nft}")
                else:
                    st.warning(f"No NFTs found for {selected_blockchain_nft}")
            else:
                    st.warning("No blockchains found.")


        elif choice == "Statistical Analyser":
            st.title("Statistical Analyser")
            

            # Add buttons for each type of statistical analysis
            analysis_options = ["Descriptive Analysis", "Correlation Analysis", "Ranking Analysis", "Categorical Analysis", "Performance Metrics Analysis"]
            selected_analysis = st.radio("Select Analysis Type:", analysis_options)

            # Perform the selected statistical analysis when the corresponding button is clicked
            if st.button("Perform Analysis"):
                if selected_analysis == "Descriptive Analysis":
                    self.data_analysis.descriptive_analysis()
                elif selected_analysis == "Correlation Analysis":
                    self.data_analysis.correlation_analysis()
                elif selected_analysis == "Ranking Analysis":
                    self.data_analysis.ranking_analysis()
                elif selected_analysis == "Categorical Analysis":
                    self.data_analysis.categorical_analysis()
                elif selected_analysis == "Performance Metrics Analysis":
                    self.data_analysis.performance_metrics_analysis()


        st.markdown(
            """
            <style>
                footer {
                    text-align: center;
                    padding: 10px;
                    position: fixed;
                    bottom: 0;
                    width: 100%;
                    background-color: black; /* Change to black */
                    color: white;
                }
            </style>
            <footer>
                <p>© 2023 WAG</p>
            </footer>
            """,
            unsafe_allow_html=True,
        )

    def show_collection_stats(self, collection_name):
        # Fetch and display detailed statistics for the selected collection
        collection_stats = self.db.fetch_one(f"SELECT * FROM Marketplaces WHERE Collection_Name = '{collection_name}'")

        if collection_stats:
            st.write(f"Collection Name: {collection_stats[2]}")
            st.write(f"Total Volume: {collection_stats[3]}")
            st.write(f"Market Cap: {collection_stats[4]}")
            st.write(f"Total Supply: {collection_stats[5]}")
            st.write(f"Thirty-Day Volume: {collection_stats[6]}")
            st.write(f"Chain Name: {collection_stats[8]}")
            st.write(f"Update Date: {collection_stats[7]}")
        else:
            st.write(f"No data found for {collection_name}")

    def show_collection_dropdown(self, selected_marketplace=None):
        if selected_marketplace:
            # Fetch distinct collection names for the selected marketplace using NFT_Marketplace_Listing
            collection_names_query = f"""
                SELECT DISTINCT C.Name
                FROM NFT_Marketplace_Listing AS N
                JOIN Collections AS C ON N.Collection_ID = C.Collection_ID
                JOIN Marketplaces AS M ON N.Marketplace_ID = M.Marketplace_ID
                WHERE M.Marketplace = '{selected_marketplace}'
            """
            collection_names = self.db.fetch_all(collection_names_query)

            if collection_names:
                # Extract the collection names from the result
                collection_names = [row[0] for row in collection_names]

                # Display the collection dropdown
                selected_collection = st.selectbox('Select a Collection:', collection_names, index=0)

                return selected_collection
            else:
                st.warning(f"No collections found for {selected_marketplace}")
                return None
        else:
            st.warning("No marketplace selected.")
            return None
        
    def is_valid_email(self, email):
        # Add your email validation logic here
        st.session_state.page = 'login'
        return  # or return some_value if needed

        # st.experimental_rerun()
    def run(self):
        if 'page' not in st.session_state:
            st.session_state.page = 'login'

        if st.session_state.page == 'login':
            self.login()
        elif st.session_state.page == 'signup':
            self.signup_page()
        elif st.session_state.page == 'main_app':
            self.main_app()

if __name__ == '__main__':
    app = App(st.session_state)
    app.run()
