from typing import List, Dict
from topic_analysis import extract_keywords

def filter_existing_playlist_videos(candidate_videos: List[Dict], playlist_videos: List[Dict])->List[Dict]:
    """Removes candidate videos whose video_id is already in the playlist.
    Prevents recommending exact duplicates."""
    playlist_video_ids = {video["video_id"] for video in playlist_videos if "video_id" in video}
    return [candidate for candidate in candidate_videos if candidate.get("video_id") not in playlist_video_ids]

def generate_topic_search_queries(grouped_playlists: Dict[str, List[Dict]], 
    max_queries_per_topic: int = 1
)->Dict[str, List[Dict]]:
    """Generates high-value search query strings for each topic group.
    Used when fetching live candidate videos via YouTube Data API.
    Extracts the most frequent non-stopword keywords within each topic cluster."""
    
    topic_queries = {}
    for topic, videos in grouped_playlists.items():
        if topic == "unknown" or not videos:
            continue
        word_counts = {}
        for video in videos:
            keywords = extract_keywords(video)
            for word in keywords:
                word_counts[word] = word_counts.get(word, 0) + 1
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        top_terms = [word for word, count in sorted_words[:3]]
        
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
    
    
    