from typing import List, Dict
from topic_analysis import extract_keywords
from collections import Counter
import re

def filter_existing_playlist_videos(candidate_videos: List[Dict], playlist_videos: List[Dict])->List[Dict]:
    """Removes candidate videos whose video_id is already in the playlist.
    Prevents recommending exact duplicates."""
    playlist_video_ids = {video["video_id"] for video in playlist_videos if "video_id" in video}
    seen_ids = set(playlist_video_ids)
    seen_titles = set()
    unique_candidates = []
    for cand in candidate_videos:
        v_id = cand.get("video_id")
        norm_title = re.sub(r"[^\w\s]", "", cand.get("title", "").lower().strip())
        if v_id and v_id not in seen_ids and norm_title not in seen_titles:
            seen_ids.add(v_id)
            seen_titles.add(norm_title)
            unique_candidates.append(cand)
    return unique_candidates

def generate_topic_search_queries(grouped_playlists: Dict[str, List[Dict]], 
    max_terms: int = 3
)->Dict[str, List[Dict]]:
    """Generates high-value search query strings for each topic group.
    Used when fetching live candidate videos via YouTube Data API.
    Extracts the most frequent non-stopword keywords within each topic cluster."""
    
    topic_queries = {}
    for topic, videos in grouped_playlists.items():
        if topic == "unknown" or not videos:
            continue
        word_counts = Counter()
        for video in videos:
            keywords = extract_keywords(video)
            word_counts.update(keywords)
        top_terms = [word for word, _ in word_counts.most_common(max_terms)]
        if top_terms:
            query = " ".join(top_terms)
            topic_queries[topic] = [query]
    return topic_queries

def retrieve_candidates_from_local_data(all_candidate_data: List[Dict], playlist_videos: List[Dict])->List[Dict]:
    """
    Validates candidates by filtering out duplicates already in the user's playlist.
    (For testing sample JSON data)
    """
    valid_candidates = filter_existing_playlist_videos(all_candidate_data, playlist_videos)
    return valid_candidates
    
    
    