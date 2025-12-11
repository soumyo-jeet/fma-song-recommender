import streamlit as st
from assets import embed_song, algorithm

st.set_page_config(
    page_title="FMA Music Recommender",
    page_icon="🎵",
    layout="centered"
)

# STREAMLIT UI
st.title("🎵 FMA Music Recommender")
st.write("Upload an audio file and get similar songs.")

uploaded_audio = st.file_uploader("Upload audio (.mp3, .wav)", type=["mp3", "wav"])

lyrics_input = st.text_area("Paste the song lyrics here...", height=150)

if st.button("Get Recommendations"):
    if uploaded_audio is None and lyrics_input.strip() == "":
        st.error("Please upload an audio file or enter lyrics.")
    
    else:
        try:
            st.info("Extracting features...")
            combined_vec = embed_song(lyrics_input, uploaded_audio)

            st.success("Features extracted successfully!")
            print("Combined Feature Vector:", combined_vec)

            st.info("Generating recommendations...")
            recommended_songs = algorithm(combined_vec, k=5)
            st.success("Recommendations generated!")
            st.subheader("Recommended Songs:")
            st.table(recommended_songs.reset_index(drop=True))

            if lyrics_input.strip():
                st.write("Lyrics Provided:")
                st.code(lyrics_input)

        except Exception as e:
            st.error(f"Error processing audio: {e}")
