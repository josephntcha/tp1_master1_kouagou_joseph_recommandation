import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# ─────────────────────────────────────────
# CHARGEMENT DES DONNÉES
# ─────────────────────────────────────────
@st.cache_data
def load_data():
    movies  = pd.read_csv("ml-latest-small/movies.csv")
    ratings = pd.read_csv("ml-latest-small/ratings.csv")
    return movies, ratings

@st.cache_data
def build_model():
    movies, ratings = load_data()

    # Matrice utilisateur-item
    user_item = ratings.pivot_table(
        index="userId",
        columns="movieId",
        values="rating"
    ).fillna(0)

    # Similarité cosinus Item-Item
    item_sim = cosine_similarity(user_item.T)
    item_sim_df = pd.DataFrame(
        item_sim,
        index=user_item.columns,
        columns=user_item.columns
    )
    return movies, item_sim_df

# ─────────────────────────────────────────
# FONCTION DE RECOMMANDATION TOP-N
# ─────────────────────────────────────────
def recommend(movie_id, item_sim_df, movies, n=10):
    if movie_id not in item_sim_df.columns:
        return pd.DataFrame()
    similar = item_sim_df[movie_id].sort_values(ascending=False)
    similar = similar.drop(movie_id)
    top_ids = similar.head(n).index.tolist()
    top_scores = similar.head(n).values.tolist()
    result = movies[movies["movieId"].isin(top_ids)].copy()
    score_map = dict(zip(top_ids, top_scores))
    result["similarite"] = result["movieId"].map(score_map)
    result = result.sort_values("similarite", ascending=False)
    return result[["title", "genres", "similarite"]]

# ─────────────────────────────────────────
# INTERFACE STREAMLIT
# ─────────────────────────────────────────
st.set_page_config(
    page_title="Système de Recommandation",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Système de Recommandation de Films")
st.subheader("Collaborative Filtering — Item-Item Top-N")
st.markdown("---")

# Chargement
with st.spinner("Chargement des données et calcul des similarités..."):
    movies, item_sim_df = build_model()

st.success(f"✅ {len(movies)} films chargés — Matrice de similarité construite !")
st.markdown("---")

# Colonnes
col1, col2 = st.columns([2, 1])

with col1:
    selected_movie = st.selectbox(
        "🎥 Choisissez un film :",
        movies["title"].sort_values().tolist()
    )

with col2:
    n = st.slider("🔢 Nombre de recommandations :", 5, 20, 10)

# Bouton recommander
if st.button("🚀 Recommander", type="primary"):
    movie_id = movies[movies["title"] == selected_movie]["movieId"].values[0]

    st.markdown(f"### Films similaires à **{selected_movie}** :")

    recs = recommend(movie_id, item_sim_df, movies, n)

    if recs.empty:
        st.warning("Aucune recommandation trouvée pour ce film.")
    else:
        for i, row in recs.iterrows():
            score = row["similarite"]
            col_a, col_b, col_c = st.columns([3, 2, 1])
            with col_a:
                st.write(f"🎬 **{row['title']}**")
            with col_b:
                st.write(f"🏷️ {row['genres']}")
            with col_c:
                st.metric("Score", f"{score:.3f}")

st.markdown("---")
st.markdown("### 📊 Statistiques du dataset")
col3, col4, col5 = st.columns(3)
movies_local, ratings_local = load_data()
with col3:
    st.metric("🎬 Films", f"{len(movies_local):,}")
with col4:
    st.metric("👥 Utilisateurs", f"{ratings_local['userId'].nunique():,}")
with col5:
    st.metric("⭐ Notes", f"{len(ratings_local):,}")