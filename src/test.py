from youtube_client import YouTubeClient
from topic_analysis import group_playlist_by_graph_jaccard
from candidate_retrieval import generate_topic_search_queries, filter_existing_playlist_videos
from ranking import rank_candidates
import pandas as pd

def run_recommendation_pipeline(playlist_url_or_id: str, api_key = None):
    client = YouTubeClient(api_key=api_key)
    print("Fetching playlist videos...")
    playlist_videos = client.fetch_playlist_videos(playlist_url_or_id)
    print(f"Loaded {len(playlist_videos)} videos from playlist.")
    
    grouped_playlist = group_playlist_by_graph_jaccard(playlist_videos)
    search_queries_dict = generate_topic_search_queries(grouped_playlist)
    queries = [q for query_list in search_queries_dict.values() for q in query_list]
    
    print(f"Generated search queries: {queries}")
    
    raw_candidates = client.search_candidates_by_queries(queries, max_per_query=10)
    print(f"Retrieved {len(raw_candidates)} candidate videos from search.")
    
    valid_candidates = filter_existing_playlist_videos(raw_candidates, playlist_videos)
    
    recommendations = rank_candidates(valid_candidates, playlist_videos, max_recommendations=10)
    return recommendations

def get_recommendation_dataframe(recommendations):
    rows = []
    for rank, rec in enumerate(recommendations, 1):
        rows.append({
            "Rank": rank,
            "Candidate Title": rec['title'],
            "Channel": rec['channel_title'],
            "Final Score": round(rec['final_score'], 4),
            "Text Score": round(rec['text_score'], 4),
            "Channel Bonus": round(rec['channel_bonus'], 4),
            "Engagement": round(rec['engagement_score'], 4),
            "Best Matched Video": rec['best_match']['playlist_title'] if rec.get('best_match') else ""
        })
    return pd.DataFrame(rows)
if __name__ == '__main__':
    playlist_url = 'https://www.youtube.com/playlist?list=PLYSxFvzslwht7pWZ4KtBVwIZDmJ3x9aoU'
    second_url = "https://www.youtube.com/watch?v=2xcFM9CBiOE&list=PLIdGxYqxOZEXyzTG_WXp9cujA3_xgp5Ps"
    recommendations = run_recommendation_pipeline(playlist_url)
    df = get_recommendation_dataframe(recommendations)
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):
        print(df)
    
    