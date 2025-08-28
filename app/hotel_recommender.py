import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import requests
from io import StringIO
import csv

# Set page configuration
st.set_page_config(
    page_title="Addis Ababa Hotel Recommender",
    page_icon="🏨",
    layout="wide"
)

# App title and description
st.title("🏨 Addis Ababa Hotel Recommender")
st.markdown("Discover hotels similar to your preferences or find the perfect stay based on your needs.")

# Load data from online source with error handling
@st.cache_data
def load_data():
    # URL to your CSV file (replace with your actual URL)
    csv_url = "https://github.com/Fuadsadik21/hotel-recommender-app/raw/refs/heads/main/data/Hotels_Data.csv"
    
    try:
        # Download the CSV file
        response = requests.get(csv_url)
        response.raise_for_status()  # Check for HTTP errors
        
        # Read the CSV data with more flexible parsing
        data = response.text
        
        # Use Python's CSV reader to handle potential formatting issues
        csv_reader = csv.reader(StringIO(data))
        rows = list(csv_reader)
        
        # Get header and data rows
        header = rows[0]
        data_rows = rows[1:]
        
        # Create DataFrame
        df = pd.DataFrame(data_rows, columns=header)
        
        # Clean up column names by stripping whitespace and newlines
        df.columns = df.columns.str.strip()
        
        # Convert numeric columns
        numeric_columns = ['Rating', 'Price', 'Free parking', 'Free Wi-Fi', 'Restaurant', 
                          'Free breakfast', 'Pool', 'Air-Conditioned', 'Spa',
                          'Is_5_star', 'Is_4_star', 'Is_3_star', 'Is_2_star', 'Is_1_star',
                          'Is_Guest_House', 'Is_Bed_Breakfast', 'Is_hotel', 'Is_hostel', 'Is_lodge',
                          'Rating_Normalized', 'Price_Normalized']
        
        for col in numeric_columns:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Create feature matrix for similarity calculation
        feature_columns = [
            'Rating_Normalized', 'Price_Normalized',
            'Free parking', 'Free Wi-Fi', 'Restaurant', 'Free breakfast', 'Pool', 
            'Air-Conditioned', 'Spa',
            'Is_5_star', 'Is_4_star', 'Is_3_star', 'Is_2_star', 'Is_1_star',
            'Is_Guest_House', 'Is_Bed_Breakfast', 'Is_hotel', 'Is_hostel', 'Is_lodge'
        ]
        
        # Check if all required columns exist
        missing_columns = [col for col in feature_columns if col not in df.columns]
        if missing_columns:
            st.error(f"Missing columns in dataset: {missing_columns}")
            return None, None, None
        
        # Fill any NaN values with 0 for the feature matrix
        feature_df = df[feature_columns].fillna(0)
        
        # Calculate cosine similarity matrix
        cosine_sim = cosine_similarity(feature_df, feature_df)
        
        # Create a reverse mapping of hotel names to indices
        # Use the first column as hotel names if 'Hotel' column doesn't exist
        hotel_column = 'Hotel' if 'Hotel' in df.columns else df.columns[0]
        hotel_indices = pd.Series(df.index, index=df[hotel_column]).drop_duplicates()
        
        return df, cosine_sim, hotel_indices
        
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None, None, None

# Load the data
df, cosine_sim, hotel_indices = load_data()

# Check if data loaded successfully
if df is None or cosine_sim is None or hotel_indices is None:
    st.error("""
    Failed to load data. This is usually due to formatting issues in the CSV file.
    
    Common issues and solutions:
    1. Check line 38 of your CSV for extra commas or formatting issues
    2. Ensure all rows have the same number of columns
    3. Make sure text fields with commas are properly quoted
    4. Verify that numeric columns don't contain text values
    
    You can download and check your CSV file using a text editor to identify the issue.
    """)
    st.stop()

# Recommendation function
def get_recommendations(hotel_name, cosine_sim_matrix=cosine_sim, num_recommendations=5):
    try:
        # Get the index of the hotel that matches the name
        idx = hotel_indices[hotel_name]

        # Get the pairwise similarity scores of all hotels with that hotel
        sim_scores = list(enumerate(cosine_sim_matrix[idx]))

        # Sort the hotels based on the similarity scores
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

        # Get the scores of the top N most similar hotels
        sim_scores = sim_scores[1:num_recommendations+1]

        # Get the hotel indices
        hotel_indices_list = [i[0] for i in sim_scores]

        # Return the top N most similar hotels
        return df.iloc[hotel_indices_list]
    except KeyError:
        return None

# Create two main tabs
tab1, tab2 = st.tabs(["🏨 Hotel Similarity Recommender", "🔍 Advanced Filter Search"])

with tab1:
    st.header("Find Similar Hotels")
    st.write("Select a hotel you like to find similar options.")
    
    # Hotel selection dropdown - use the first column if 'Hotel' doesn't exist
    hotel_column = 'Hotel' if 'Hotel' in df.columns else df.columns[0]
    hotel_list = df[hotel_column].tolist()
    selected_hotel = st.selectbox("Select a hotel:", hotel_list)
    
    # Number of recommendations slider
    num_recommendations = st.slider("Number of recommendations:", 3, 10, 5)
    
    # Get recommendations when hotel is selected
    if selected_hotel:
        recommendations = get_recommendations(selected_hotel, num_recommendations=num_recommendations)
        
        if recommendations is not None and len(recommendations) > 0:
            st.subheader(f"Hotels similar to {selected_hotel}:")
            
            # Display recommendations in a nice format
            for i, (_, hotel) in enumerate(recommendations.iterrows(), 1):
                # Safely get category information
                category = hotel.get('Category', 'N/A')
                
                with st.expander(f"{i}. {hotel[hotel_column]} ({category})"):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write(f"**Rating:** {hotel.get('Rating', 'N/A')}")
                        st.write(f"**Price:** {hotel.get('Price', 'N/A')}")
                        
                        # Display amenities
                        amenities = []
                        if hotel.get('Free parking', 0) == 1:
                            amenities.append("Free Parking")
                        if hotel.get('Free Wi-Fi', 0) == 1:
                            amenities.append("Free Wi-Fi")
                        if hotel.get('Restaurant', 0) == 1:
                            amenities.append("Restaurant")
                        if hotel.get('Free breakfast', 0) == 1:
                            amenities.append("Free Breakfast")
                        if hotel.get('Pool', 0) == 1:
                            amenities.append("Pool")
                        if hotel.get('Air-Conditioned', 0) == 1:
                            amenities.append("Air Conditioning")
                        if hotel.get('Spa', 0) == 1:
                            amenities.append("Spa")
                            
                        st.write(f"**Amenities:** {', '.join(amenities)}")
                    
                    with col2:
                        # You could add hotel images here if you had them
                        st.write("")  # Placeholder for image
        else:
            st.warning("No recommendations found. Please try another hotel.")

with tab2:
    st.header("Find Hotels by Criteria")
    st.write("Filter hotels based on your specific requirements.")
    
    # Create filters
    col1, col2 = st.columns(2)
    
    with col1:
        min_rating = st.slider("Minimum Rating", 0.0, 5.0, 3.0, 0.1)
        max_price = st.slider("Maximum Price", float(df['Price'].min()), float(df['Price'].max()), 
                             float(df['Price'].max()), 100.0)
        
    with col2:
        # Safely get category options
        category_options = df['Category'].unique() if 'Category' in df.columns else ['N/A']
        category_filter = st.multiselect("Hotel Category", options=category_options)
        amenities_filter = st.multiselect("Amenities", 
                                         ["Free parking", "Free Wi-Fi", "Restaurant", "Free breakfast", "Pool", "Air-Conditioned", "Spa"])
    
    # Filter the dataframe based on selections
    filtered_df = df.copy()
    
    # Apply rating filter
    filtered_df = filtered_df[filtered_df['Rating'] >= min_rating]
    
    # Apply price filter
    filtered_df = filtered_df[filtered_df['Price'] <= max_price]
    
    # Apply category filter if category column exists
    if 'Category' in df.columns and category_filter:
        filtered_df = filtered_df[filtered_df['Category'].isin(category_filter)]
    
    # Apply amenities filters
    if "Free parking" in amenities_filter:
        filtered_df = filtered_df[filtered_df['Free parking'] == 1]
    if "Free Wi-Fi" in amenities_filter:
        filtered_df = filtered_df[filtered_df['Free Wi-Fi'] == 1]
    if "Restaurant" in amenities_filter:
        filtered_df = filtered_df[filtered_df['Restaurant'] == 1]
    if "Free breakfast" in amenities_filter:
        filtered_df = filtered_df[filtered_df['Free breakfast'] == 1]
    if "Pool" in amenities_filter:
        filtered_df = filtered_df[filtered_df['Pool'] == 1]
    if "Air-Conditioned" in amenities_filter:
        filtered_df = filtered_df[filtered_df['Air-Conditioned'] == 1]
    if "Spa" in amenities_filter:
        filtered_df = filtered_df[filtered_df['Spa'] == 1]
    
    # Display results
    st.subheader(f"Found {len(filtered_df)} hotels matching your criteria")
    
    if len(filtered_df) > 0:
        for i, (_, hotel) in enumerate(filtered_df.iterrows(), 1):
            # Safely get category information
            category = hotel.get('Category', 'N/A')
            
            with st.expander(f"{i}. {hotel[hotel_column]} ({category})"):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.write(f"**Rating:** {hotel.get('Rating', 'N/A')}")
                    st.write(f"**Price:** {hotel.get('Price', 'N/A')}")
                    
                    # Display amenities
                    amenities = []
                    if hotel.get('Free parking', 0) == 1:
                        amenities.append("Free Parking")
                    if hotel.get('Free Wi-Fi', 0) == 1:
                        amenities.append("Free Wi-Fi")
                    if hotel.get('Restaurant', 0) == 1:
                        amenities.append("Restaurant")
                    if hotel.get('Free breakfast', 0) == 1:
                        amenities.append("Free Breakfast")
                    if hotel.get('Pool', 0) == 1:
                        amenities.append("Pool")
                    if hotel.get('Air-Conditioned', 0) == 1:
                        amenities.append("Air Conditioning")
                    if hotel.get('Spa', 0) == 1:
                        amenities.append("Spa")
                        
                    st.write(f"**Amenities:** {', '.join(amenities)}")
                
                with col2:
                    # Placeholder for image
                    st.write("")
    else:
        st.info("No hotels match your criteria. Try adjusting your filters.")

# Add a footer
st.markdown("---")
st.markdown("### About this app")
st.markdown("This hotel recommendation system uses content-based filtering to suggest similar hotels based on their features, ratings, and amenities.")

# Display column names for debugging
st.sidebar.markdown("### Debug Info")

st.sidebar.write("Columns in dataset:", df.columns.tolist())


