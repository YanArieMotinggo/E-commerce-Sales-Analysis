import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob

# Set up Streamlit page
st.set_page_config(page_title="E-commerce Sales Analysis", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("amazon.csv")
    
    # Check for column existence before dropping NA
    required_columns = ['product_id', 'product_name', 'discounted_price', 
                        'actual_price', 'rating', 'rating_count']
    optional_columns = ['category', 'review_content']
    
    missing_columns = [col for col in required_columns + optional_columns if col not in df.columns]
    if missing_columns:
        st.warning(f"Warning: The following columns are missing from the dataset: {', '.join(missing_columns)}")
    
    # Process required columns
    critical_columns = [col for col in required_columns if col in df.columns]
    df = df.dropna(subset=critical_columns)
    
    # Remove currency symbols and convert to numeric
    if 'discounted_price' in df.columns:
        df['discounted_price'] = df['discounted_price'].replace('[\₹\$,]', '', regex=True).astype(float)
    if 'actual_price' in df.columns:
        df['actual_price'] = df['actual_price'].replace('[\₹\$,]', '', regex=True).astype(float)
    if 'rating' in df.columns:
        df['rating'] = pd.to_numeric(df['rating'], errors='coerce')  
        df = df.dropna(subset=['rating'])
    if 'rating_count' in df.columns:
        df['rating_count'] = pd.to_numeric(df['rating_count'], errors='coerce').fillna(0).astype(int)
    
    # Add calculated columns
    if 'actual_price' in df.columns and 'discounted_price' in df.columns:
        df['discount_percentage'] = ((df['actual_price'] - df['discounted_price']) / df['actual_price']) * 100
    if 'rating' in df.columns and 'rating_count' in df.columns:
        df['sales_volume'] = df['rating_count'] * df['rating']
    
    # Sentiment column handling: If it doesn't exist, analyze the sentiment of the reviews
    if 'review_sentiment' not in df.columns:
        if 'review_content' in df.columns:
            df['sentiment_score'] = df['review_content'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
            
            # Classify sentiment based on the polarity score
            df['review_sentiment'] = df['sentiment_score'].apply(
                lambda score: 'Positive' if score > 0 else ('Negative' if score < 0 else 'Neutral')
            )
        else:
            st.warning("No 'review_text' column found. A default 'Neutral' sentiment will be assigned to all rows.")
            df['review_sentiment'] = 'Neutral'  # Fallback if no review text exists
    
    return df

# Load the dataset
df = load_data()

# Section: Four Key Analyses in a Row
st.title("E-commerce Sales Analysis")
st.markdown("Explore the key metrics that impact sales performance. Filter the data by category to gain tailored insights.")

# Create five columns for the visualizations and category selection
col1, col2, col3, col4, col5 = st.columns(5)

# Category Selection in the First Column
with col1:
    st.subheader("Category Selection")
    if 'category' in df.columns:
        categories = df['category'].unique()
        selected_category = st.selectbox(
            "Select a Category",
            ["All Categories"] + list(categories),
            key="category_select",
        )
        if selected_category != "All Categories":
            df = df[df['category'] == selected_category]
    st.markdown("Filter the data by product categories to analyze specific trends.")

# 1. Discount Percentage vs. Sales Volume
with col2:
    fig1, ax1 = plt.subplots(figsize=(5, 5))
    sns.scatterplot(data=df, x='discount_percentage', y='sales_volume', alpha=0.6, ax=ax1)
    ax1.set_title("Discount % vs. Sales Volume")
    ax1.set_xlabel("Discount Percentage")
    ax1.set_ylabel("Sales Volume")
    st.pyplot(fig1)
    st.subheader("Discount Percentage vs. Sales Volume")

# 2. Actual Price vs. Discounted Price
with col3:
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    sns.scatterplot(data=df, x='actual_price', y='discounted_price', hue='sales_volume', palette='viridis', alpha=0.6, ax=ax2)
    ax2.set_title("Actual vs. Discounted Price")
    ax2.set_xlabel("Actual Price")
    ax2.set_ylabel("Discounted Price")
    st.pyplot(fig2)
    st.subheader("Actual Price vs. Discounted Price")

# 3. Correlation Matrix: Prices, Ratings, and Review Counts
with col4:
    correlation_matrix = df[['actual_price', 'discounted_price', 'rating', 'rating_count']].corr()
    fig3, ax3 = plt.subplots(figsize=(5, 5))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax3)
    ax3.set_title("Correlation Matrix")
    st.pyplot(fig3)
    st.subheader("Correlation Matrix")

# 4. Distribution of Ratings
with col5:
    fig4, ax4 = plt.subplots(figsize=(5, 5))
    sns.histplot(data=df, x='rating', bins=10, kde=True, color='purple', ax=ax4)
    ax4.set_title("Distribution of Ratings")
    ax4.set_xlabel("Rating")
    ax4.set_ylabel("Count")
    st.pyplot(fig4)
    st.subheader("Distribution of Ratings")

# Sentiment Analysis Section in a Grid
st.header("Review Sentiment Analysis")
col6, col7 = st.columns(2)

# Sentiment Distribution
with col6:
    sentiment_counts = df['review_sentiment'].value_counts()
    fig5, ax5 = plt.subplots(figsize=(5, 5))
    sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="viridis", ax=ax5)
    ax5.set_title("Distribution of Review Sentiments")
    ax5.set_xlabel("Sentiment")
    ax5.set_ylabel("Count")
    st.pyplot(fig5)

# Impact of Sentiment on Sales
with col7:
    sentiment_sales = df.groupby('review_sentiment')['sales_volume'].mean().reset_index()
    fig6, ax6 = plt.subplots(figsize=(5, 5))
    sns.barplot(data=sentiment_sales, x='review_sentiment', y='sales_volume', palette="viridis", ax=ax6)
    ax6.set_title("Average Sales Volume by Review Sentiment")
    ax6.set_xlabel("Sentiment")
    ax6.set_ylabel("Average Sales Volume")
    st.pyplot(fig6)
