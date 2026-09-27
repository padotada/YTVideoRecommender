from typing import List, Dict
from topic_analysis import score_candidate_pairwise, extract_keywords

def calculate_engagement_ratio(video: dict)->float:
    """Calculates a basic engagement ratio (likes / views).
    Returns 0.0 if views are missing or zero."""
    
    views = video.get("view_count", 0)
    likes = video.get("like_count", 0)
    if not views or views <= 0:
        return 0.0
    return likes / views

def score_candidate(candidate: dict, playlist_videos: list, channel_counts: dict)->dict:
    """Scores a single candidate video using a weighted combination of:
       - Text similarity (Jaccard with top pairwise playlist matches)
       - frequent channel name in playlist
       - engagement ratio (likes to view ratio)
    Returns a score breakdown dict   
    """
    pairwise_info = score_candidate_pairwise(candidate, playlist_videos, top_n=3)
    text_score = pairwise_info["avg_top_score"]
    if text_score < 0.02:
        return {
            "candidate_id": candidate.get("video_id"),
            "title": candidate.get("title"),
            "final_score": text_score, 
            "text_score": text_score, 
            "channel_bonus": 0.0,
            "engagement_score": 0.0,
            "best_match": pairwise_info["best_matches"][0] if pairwise_info["best_matches"] else None
        }
        
    cand_channel = candidate.get("channel_title")
    total_playlist_vids = len(playlist_videos)
    channel_freq = channel_counts.get(cand_channel, 0) / total_playlist_vids if total_playlist_vids > 0 else 0
    channel_bonus = min(channel_freq * 0.15, 0.15)
    engagement = calculate_engagement_ratio(candidate)
    engagement_score = min(engagement * 0.5, 0.05)
    final_score = text_score + channel_bonus + engagement_score
    
    return {
        "candidate_id": candidate.get("video_id"),
        "title": candidate.get("title"),
        "channel_title": cand_channel,
        "final_score": final_score,
        "text_score": text_score,
        "channel_bonus": channel_bonus,
        "engagement_score": engagement_score,
        "best_match": pairwise_info["best_matches"][0] if pairwise_info["best_matches"] else None
    }
    
def rank_candidates(candidates: List[Dict], playlist_videos: List[Dict], max_recommendations: int = 10,
                    max_per_channel: int = 3)->List[Dict]:
    """Ranks eligible candidates, applies channel repetition caps, and returns
    up to max_recommendation results with transparent scoring."""
    
    channel_counts = {}
    for video in playlist_videos:
        ch = video.get("channel_title")
        if ch:
            channel_counts[ch] = channel_counts.get(ch, 0) + 1
    scored_candidates = [score_candidate(cand, playlist_videos, channel_counts) for cand in candidates]
    scored_candidates.sort(key=lambda x: x["final_score"], reverse=True)
    
    final_recommendations = []
    channel_selection_counts = {}
    
    for candidate in scored_candidates:
        ch = candidate.get("channel_title")
        current_count = channel_selection_counts.get(ch, 0)
        if current_count < max_per_channel:
            final_recommendations.append(candidate)
            channel_selection_counts[ch] = current_count + 1
        if len(final_recommendations) >= max_recommendations:
            break
    return final_recommendations