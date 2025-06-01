import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("bookrec.csv")

df = load_data()

# Title
st.title("📚 Books Recommender")

# Sidebar options
option = st.sidebar.radio("Recommend by", ["Genre", "Book Title"])

# --- Genre-based Recommendation ---
if option == "Genre":
    genre_input = st.text_input("Enter a genre (e.g., Romance, Thriller)", "")
    
    if genre_input:
        filtered = df[df['Genre'].str.contains(genre_input, case=False, na=False)]
        
        if not filtered.empty:
            st.subheader(f"Books in Genre: {genre_input}")
            for _, row in filtered.sample(min(10, len(filtered))).iterrows():
                st.markdown(f"**{row['Title']}** by {row['Author']}")
        else:
            st.warning("No books found for that genre.")

# --- Book-based Recommendation ---
else:
    title_input = st.text_input("Enter a book title ", "")

    if title_input:
        matching_titles = df[df['Title'].str.contains(title_input, case=False, na=False)]
        
        if not matching_titles.empty:
            selected_title = matching_titles.iloc[0]['Title']  # pick first match
            
            tfidf = TfidfVectorizer()
            tfidf_matrix = tfidf.fit_transform(df['Title'] + " " + df['Genre'])
            cosine_sim = cosine_similarity(tfidf_matrix)
            
            idx = df[df['Title'] == selected_title].index[0]
            scores = list(enumerate(cosine_sim[idx]))
            scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]
            
            st.subheader(f"Books similar to: {selected_title}")
            for i, _ in scores:
                st.markdown(f"**{df.iloc[i]['Title']}** by {df.iloc[i]['Author']} ({df.iloc[i]['Genre']})")
        else:
            st.warning("No matching book titles found.")
