import numpy as np
import pandas as pd
import librosa
from scipy import stats
import joblib
def extract_audio_features(file_path):
    feature_sizes = dict(
        chroma_stft=12,
        chroma_cqt=12,
        chroma_cens=12,
        tonnetz=6,
        mfcc=20,
        rmse=1,
        zcr=1,
        spectral_centroid=1,
        spectral_bandwidth=1,
        spectral_contrast=7,
        spectral_rolloff=1
    )
    moments = ("mean", "std", "skew", "kurtosis", "median", "min", "max")

    # Prepare output dict for all raw features
    raw_feats = {}

    # 2. Load audio
    y, sr = librosa.load(file_path, sr=None, mono=True)

    # Helper to push all statistical moments
    def store(name, values):
        raw_feats[name] = {
            "mean": np.mean(values, axis=1),
            "std": np.std(values, axis=1),
            "skew": stats.skew(values, axis=1),
            "kurtosis": stats.kurtosis(values, axis=1),
            "median": np.median(values, axis=1),
            "min": np.min(values, axis=1),
            "max": np.max(values, axis=1),
        }

    # 3. Compute features (same code as FMA)

    # zcr
    f = librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=512)
    store("zcr", f)

    # CQT → chroma_cqt, chroma_cens, tonnetz
    cqt = np.abs(librosa.cqt(
        y, sr=sr, hop_length=512, bins_per_octave=12, n_bins=7 * 12
    ))

    store("chroma_cqt", librosa.feature.chroma_cqt(C=cqt, n_chroma=12))
    store("chroma_cens", librosa.feature.chroma_cens(C=cqt, n_chroma=12))
    tonnetz = librosa.feature.tonnetz(chroma=librosa.feature.chroma_cqt(C=cqt))
    store("tonnetz", tonnetz)

    # STFT → stft-based features
    stft = np.abs(librosa.stft(y, n_fft=2048, hop_length=512))

    store("chroma_stft", librosa.feature.chroma_stft(S=stft**2, n_chroma=12))
    store("rmse", librosa.feature.rms(S=stft))
    store("spectral_centroid", librosa.feature.spectral_centroid(S=stft))
    store("spectral_bandwidth", librosa.feature.spectral_bandwidth(S=stft))
    store("spectral_contrast", librosa.feature.spectral_contrast(S=stft, n_bands=6))
    store("spectral_rolloff", librosa.feature.spectral_rolloff(S=stft))

    # MFCC
    mel = librosa.feature.melspectrogram(sr=sr, S=stft ** 2)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel), n_mfcc=20)
    store("mfcc", mfcc)

    # 4. Flatten into one feature vector (same ordering)
    final_vector = []

    for feat, size in feature_sizes.items():
        for moment in moments:
            final_vector.extend(raw_feats[feat][moment])

    final_vector = np.array(final_vector, dtype=np.float32)  # 518 dims originally

    # 5. DROP CORRELATED FEATURES → produce exactly 336 dims
    remove_groups = [
        "spectral_rolloff",
        "spectral_bandwidth",
        "chroma_cens",
        "chroma_stft",
    ]

    idx_to_keep = []
    start = 0
    for feat, size in feature_sizes.items():
        block_len = size * len(moments)  # 7 moments
        end = start + block_len

        if feat not in remove_groups:
            idx_to_keep.extend(list(range(start, end)))

        start = end

    cleaned_vector = final_vector[idx_to_keep]  # final 336-dim vector

    scaler = joblib.load("../models/audio_feature_scaler.pkl")
    cleaned_vector = scaler.transform([cleaned_vector])[0]

    return cleaned_vector



import joblib
import re
import spacy
nlp = spacy.load("en_core_web_sm")
def modify_lyrics(lyrics):
    # Remove text within square brackets
    lyrics = re.sub(r'\[.*?\]', '', lyrics)
    # Convert to lowercase
    lyrics = lyrics.lower()
    # Remove extra whitespace
    lyrics = re.sub(r'\s+', ' ', lyrics).strip()
    # Lemmatization and stopword removal
    doc = nlp(lyrics)
    lyrics = ' '.join([token.text for token in doc if not token.is_stop and not token.is_punct])
    return lyrics

def extract_lyrics_vector(lyrics):
    vectorizer = joblib.load("../models/tfidf_vectorizer.pkl")
    lyrics = modify_lyrics(lyrics)
    return vectorizer.transform([lyrics]).toarray()[0]



def embed_song(lyrics, audio_file):
    audio_vec = extract_audio_features(audio_file)
    lyrics_vec = extract_lyrics_vector(lyrics)
    print(lyrics_vec.shape)
    print(audio_vec.shape)
    combined_vec =np.concatenate([audio_vec * 0.6, lyrics_vec * 0.4])
    return combined_vec



from sklearn.metrics.pairwise import cosine_similarity
import pyarrow.feather as feather
def load_dataset():
    table = feather.read_table("../dataset/featured_data_with_clusters.feather")
    data = table.to_pandas()
    return data

def recc_top_k (train_data, target_song, k):
    # get all song embeddings in 2D array
    all_song_embeddings = np.vstack(train_data.loc[:, 'song_embedding'].values)

    # compute cosine similarity scores
    scores = cosine_similarity(all_song_embeddings, [target_song]).flatten()
    # getting top k indices
    idx = np.argsort(scores)[-k:]
    
    # getting songs corresponding to the indices
    return train_data.iloc[idx]['title']
    
def algorithm(target_song_embedding, k):
    # loading model
    km_model = joblib.load('../models/kmeans_model.pkl')
    train_data = load_dataset()

    # predicting cluster for the target song
    cluster_label = km_model.predict([target_song_embedding])[0]

    # getting songs from the same cluster
    cluster_songs = train_data[train_data['cluster'] == cluster_label]
    return recc_top_k(cluster_songs, target_song_embedding, k)
    