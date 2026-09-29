import re
from collections import Counter
from typing import List, Dict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
import numpy as np

YOUTUBE_STOP_WORDS = {
    "a", "an", "the", "and", "or", "but", "for", "from", "in", "into", "of", "on", "to", 
    "with", "by", "at", "as", "is", "are", "be", "this", "that", "these", "those", "your", 
    "you", "my", "we", "it", "how", "what", "video", "videos", "channel", "subscribe", 
    "sub", "subscribing", "like", "comment", "share", "watch", "official", "hd", "4k", 
    "com", "https", "http", "www", "youtube", "link", "follow", "twitter", "instagram", 
    "facebook", "discord", "patreon", "spotify", "music", "full", "part", "episode"
}

def extract_keywords(video: dict, stop_words: set = YOUTUBE_STOP_WORDS)->set:
    """Extracts normalized, unique keywords from a video's title, description, and tags.
    Handles missing tags (null or empty) safely."""
    
    title = video.get("title", "")
    description = video.get("description", "")
    raw_tags = video.get("tags")
    tags_text = " ".join(raw_tags) if raw_tags and isinstance(raw_tags, list) else ""
    
    text = f"{title} {title} {tags_text} {tags_text} {description}"
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {word for word in words if word not in stop_words and len(word) > 2}

def jaccard_similarity(set1, set2):
    """Calculates Jaccard similarity between two sets.
    J(A, B) = |A ∩ B| / |A ∪ B|"""
    if not set1 and not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union

def score_candidate_pairwise(candidate: dict, playlist_videos: list, top_n: int = 3):
    """Compares a candidate video against each video in the playlist individually. Returns the average Jaccard score of the top-N closest matches
    along with matched details."""
    cand_keywords = extract_keywords(candidate)
    pairwise_matches = []
    for p_vid in playlist_videos:
        p_keywords = extract_keywords(p_vid)
        score = jaccard_similarity(cand_keywords, p_keywords)
        pairwise_matches.append({
            "playlist_video_id" : p_vid.get("video_id"),
            "playlist_title" : p_vid.get("title"),
            "score" : score,
            "shared_words" : cand_keywords.intersection(p_keywords)})
    pairwise_matches.sort(key=lambda x: x["score"], reverse=True)
    top_matches = pairwise_matches[:top_n]
    
    avg_score = sum(m["score"] for m in top_matches) / len(top_matches) if top_matches else 0.0
    
    return {
        "candidate_id": candidate.get("video_id"),
        "title": candidate.get("title"),
        "avg_top_score": avg_score,
        "max_score": top_matches[0]["score"] if top_matches else 0.0,
        "best_matches": top_matches
    }
    
def group_playlist_by_graph_jaccard(playlist_videos: List[Dict], similarity_threshold: float=0.08)->Dict[str, List]:
    """Dynamically clusters playlist videos into topic groups using a graph based Jaccard similarity threshold
    and connected components. Auto generates cluster labels based on the most frequent shared keywords."""
    if not playlist_videos:
        return {}
    
    # Extract keyword sets for every video
    video_keywords={
        v.get("video_id"): extract_keywords(v)
        for v in playlist_videos
    }
    video_map = {v.get("video_id"): v for v in playlist_videos}
    video_ids = list(video_map.keys())
    
    # Build adjacency list for connected components
    adj_list = {v_id: set() for v_id in video_ids}
    for i in range(len(video_ids)):
        for j in range(i+1, len(video_ids)):
            id1, id2 = video_ids[i], video_ids[j]
            sim = jaccard_similarity(video_keywords[id1], video_keywords[id2])
            if sim >= similarity_threshold:
                adj_list[id1].add(id2)
                adj_list[id2].add(id1)
        
    # Find connected components using BFS       
    visited = set()
    clusters = []
    for v_id in video_ids:
        if v_id not in visited:
            component = []
            queue = [v_id]
            visited.add(v_id)
            while queue:
                curr = queue.pop(0)
                component.append(video_map[curr])
                for neighbor in adj_list[curr]:
                    if neighbor not in visited:
                        visited.add(neighbor)
                        queue.append(neighbor)
            clusters.append(component)

    # Generate group labels from top cluster keywords     
    topic_groups = {}
    for idx, cluster_vids in enumerate(clusters):
        word_counts = Counter()
        for v in cluster_vids:
            word_counts.update(video_keywords[v.get("video_id")])
        
        top_terms = [word for word, _ in word_counts.most_common(2)]
        auto_label = "_".join(top_terms) if top_terms else f"topic_{idx+1}"
        topic_groups[auto_label] = cluster_vids
    return topic_groups
    
def group_playlist_by_tfidf(playlist_videos: List[Dict], distance_threshold: float=0.7)->Dict[str, List[Dict]]:
    """Clusters playlist videos using TF-IDF feature extraction and agglomerative cosine distance clustering."""
    if not playlist_videos:
        return {}
    
    # Prepare raw text corpus for each video
    corpus = []
    for v in playlist_videos:
        title = v.get("title", "")
        desc = v.get("description", "")
        tags = " ".join(v.get("tags") or [])
        full_text = f"{title} {desc} {tags}".strip()
        corpus.append(full_text)
        
    if not any(corpus):
        return {"all_videos": playlist_videos}
        
    # Extract TF-IDF matrix
    vectorizer = TfidfVectorizer(stop_words='english', max_features=500)
    tfidf_sparse = vectorizer.fit_transform(corpus)
    tfidf_matrix = np.asarray(tfidf_sparse.todense())
    feature_names = np.array(vectorizer.get_feature_names_out())
    
    # Perform agglomerative clustering with cosine distance
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric='cosine',
        linkage='average',
        distance_threshold=distance_threshold
    )
    labels = clustering.fit_predict(tfidf_matrix)
    
    clusters = {}
    for idx, label in enumerate(labels):
        clusters.setdefault(label, []).append((playlist_videos[idx], tfidf_matrix[idx]))
    topic_groups = {}
    for label, cluster_items in clusters.items():
        vids = [item[0] for item in cluster_items]
        vectors = np.array([item[1] for item in cluster_items])
        mean_vector = vectors.mean(axis=0)
        top_indices = mean_vector.argsort()[-2:][::-1]
        top_words = feature_names[top_indices]
        auto_label = '_'.join(top_words) if len(top_words) > 0 else f"cluster_{label}"
        topic_groups[auto_label] = vids
    return topic_groups

def group_playlist_by_topic(playlist_videos: list)->dict:
    """Groups playlist videos based on simple keyword/tag heuristics.
    Falls back to 'unknown' if no domain keywords are present.
    """
    topic_groups = {}
    DOMAIN_KEYWORDS = {
        "gaming": {"minecraft", "game", "gaming", "survival", "redstone", "crafting", "build"},
        "music": {"lofi", "beats", "music", "chillhop", "mix", "instrumental", "focus"},
        "programming": {"python", "programming", "code", "json", "data", "lists", "dictionaries"}
    }
    
    for video in playlist_videos:
        keywords = extract_keywords(video)
        assigned_topic = "unknown"
        max_overlap = 0
        for domain, domain_words in DOMAIN_KEYWORDS.items():
            overlap = len(keywords.intersection(domain_words))
            if overlap > max_overlap:
                max_overlap = overlap
                assigned_topic = domain
        topic_groups.setdefault(assigned_topic, []).append(video)
        
    return topic_groups