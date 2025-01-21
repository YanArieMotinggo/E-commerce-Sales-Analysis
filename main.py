import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from textblob import TextBlob

# Set up Streamlit page
st.set_page_config(page_title="E-commerce Sales Analysis", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("amazon.csv")
    
    # Check for column existence before dropping NA
    required_columns = ['product_id', 'product_name', 'discounted_price', 
                        'actual_price', 'rating', 'rating_count']
    optional_columns = ['category', 'date', 'review_text']
    
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
    
    # Process 'category' to only include the first two levels
    if 'category' in df.columns:
        df['category'] = df['category'].str.split('|').str[:2].str.join('|')
    
    # Convert 'date' to datetime if it exists
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    return df


# Sentiment analysis function
def analyze_sentiment(text):
    analysis = TextBlob(text)
    if analysis.sentiment.polarity > 0:
        return "Positive"
    elif analysis.sentiment.polarity < 0:
        return "Negative"
    else:
        return "Neutral"

# Load data
df = load_data()

# Perform sentiment analysis
df['review_sentiment'] = df['review_content'].apply(lambda x: analyze_sentiment(x) if pd.notnull(x) else "Neutral")


# Sidebar filters
st.sidebar.title("Filters")
st.sidebar.header("Filter Dataset")

categories = df['category'].unique()
selected_category = st.sidebar.selectbox("Select Category", ["All Categories"] + list(categories))

if selected_category != "All Categories":
    df = df[df['category'] == selected_category]

min_rating, max_rating = st.sidebar.slider("Filter by Rating", 0.0, 5.0, (0.0, 5.0), key="rating_slider")
df = df[(df['rating'] >= min_rating) & (df['rating'] <= max_rating)]

min_price, max_price = st.sidebar.slider("Filter by Discounted Price", 0.0, float(df['discounted_price'].max()), (0.0, float(df['discounted_price'].max())), key="price_slider")
df = df[(df['discounted_price'] >= min_price) & (df['discounted_price'] <= max_price)]

st.title("E-commerce Sales Analysis")

st.markdown("---")

# Section 1: Discount Analysis
st.header("Discount Analysis")
if 'category' in df.columns:
    st.subheader("Average Discount Percentage by Category")
    discount_by_category = df.groupby('category')['discount_percentage'].mean().sort_values(ascending=False)
    st.bar_chart(discount_by_category)

st.subheader("Discount Percentage vs. Sales Volume")
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=df, x='discount_percentage', y='sales_volume', alpha=0.5, ax=ax)
ax.set_title('Discount Percentage vs. Sales Volume')
st.pyplot(fig)

# Section 2: Pricing Analysis
st.header("Pricing Analysis")
st.subheader("Actual Price vs. Discounted Price")
fig, ax = plt.subplots(figsize=(10, 6))
sns.scatterplot(data=df, x='actual_price', y='discounted_price', hue='sales_volume', palette='viridis', alpha=0.6, ax=ax)
ax.set_title('Actual Price vs. Discounted Price with Sales Volume')
st.pyplot(fig)

st.subheader("Correlation Matrix: Prices, Ratings, and Review Counts")
correlation_matrix = df[['actual_price', 'rating', 'rating_count']].corr()
fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
st.pyplot(fig)

# Section 3: Ratings & Review Analysis
st.header("Ratings & Review Analysis")
st.subheader("Distribution of Ratings")
fig, ax = plt.subplots(figsize=(8, 6))
sns.histplot(data=df, x='rating', bins=10, kde=True, color='purple', ax=ax)
ax.set_title('Distribution of Ratings')
st.pyplot(fig)

st.subheader("Products with High Ratings but Low Sales")
high_ratings_low_sales = df[(df['rating'] >= 4.5) & (df['sales_volume'] <= 50)]
st.dataframe(high_ratings_low_sales[['product_name', 'rating', 'sales_volume']])

st.subheader("Products with Low Ratings but High Sales")
low_ratings_high_sales = df[(df['rating'] < 4.5) & (df['sales_volume'] > 50)]
st.dataframe(low_ratings_high_sales[['product_name', 'rating', 'sales_volume']])

# Sentiment Analysis Section
st.header("Review Sentiment Analysis")
sentiment_counts = df['review_sentiment'].value_counts()
st.subheader("Sentiment Distribution")
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(x=sentiment_counts.index, y=sentiment_counts.values, palette="viridis", ax=ax)
ax.set_title("Distribution of Review Sentiments")
ax.set_xlabel("Sentiment")
ax.set_ylabel("Count")
st.pyplot(fig)

st.subheader("Impact of Sentiment on Sales")
sentiment_sales = df.groupby('review_sentiment')['sales_volume'].mean().reset_index()
fig, ax = plt.subplots(figsize=(8, 6))
sns.barplot(data=sentiment_sales, x='review_sentiment', y='sales_volume', palette="viridis", ax=ax)
ax.set_title("Average Sales Volume by Review Sentiment")
ax.set_xlabel("Sentiment")
ax.set_ylabel("Average Sales Volume")
st.pyplot(fig)

st.subheader("Comparison of Discounts and Sentiment")
df['discount_group'] = pd.cut(df['discount_percentage'], bins=[0, 20, 50, 100], labels=["Low", "Medium", "High"])
discount_sentiment = df.groupby(['discount_group', 'review_sentiment'])['sales_volume'].mean().unstack()
st.write("Average Sales Volume by Discount Group and Sentiment")
st.dataframe(discount_sentiment)