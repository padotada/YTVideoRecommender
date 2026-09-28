import os
import re
from typing import List, Dict
from googleapiclient.discovery import build

class YouTubeClient:
    def __init__(self, api_key = None):
        self.api_key = api_key or os.getenv("API_KEY")
        if not self.api_key:
            raise ValueError("API Key is required. Set YOUTUBE_API_KEY environment variable or pass api_key.")
        self.youtube = build("youtube", "v3", developerKey=self.api_key)
        
    @staticmethod
    def extract_playlist_id(url_or_id: str)->str:
        """Extracts playlist ID from full YouTube URL or returns the ID directly."""
        match = re.search(r"list=([a-zA-z0-9_-]+)", url_or_id)
        if match:
            return match.group(1)
        return url_or_id
    
    def fetch_playlist_videos(self, playlist_url_or_id: str, max_results: int = 50):
        """Fetches full metadata for videos inside a public YouTube playlist."""
        playlist_id = self.extract_playlist_id(playlist_url_or_id)
        playlist_request = self.youtube.playlistItems().list(
            part="contentDetails",
            playlistId=playlist_id,
            maxResults=max_results
        )
        playlist_response = playlist_request.execute()
        video_ids = [item["contentDetails"]["videoId"] for item in playlist_response.get("items", [])]
        if not video_ids:
            return []
        return self.fetch_videos_by_ids(video_ids)
    
    def fetch_videos_by_ids(self, video_ids: List[str])->List[Dict]:
        """Fetches detailed metadata for a list of video IDs and converts them to a standard internal scheme."""
        if not video_ids:
            return []
        videos_request = self.youtube.videos().list(
            part="snippet,statistics",
            id=','.join(video_ids[:50]) # API limit is 50 per batch
        )
        response = videos_request.execute()
        formatted_videos = []
        for item in response.get("items", []):
            snippet = item.get("snippet", {})
            statistics = item.get("statistics", {})

            formatted_videos.append({
                "video_id": item.get("id"),
                "title": snippet.get("title", ""),
                "description": snippet.get("description", ""),
                "channel_id": snippet.get("channelId", ""),
                "channel_title": snippet.get("channelTitle", ""),
                "tags": snippet.get("tags", []),  # Handled safely if missing/empty
                "published_at": snippet.get("publishedAt", ""),
                "view_count": int(statistics.get("viewCount", 0)),
                "like_count": int(statistics.get("likeCount", 0)),
            })

        return formatted_videos
    
    def search_candidates_by_queries(self, queries: List, max_per_query: int = 10) -> List[Dict]:
        """Searches YouTube for candidate videos based on query strings"""
        candidate_ids = set()
        for q in queries:
            search_request = self.youtube.search().list(
                part="id",
                q=q,
                type="video",
                maxResults=max_per_query
            )
            search_response = search_request.execute()
            
            for item in search_response.get("items", []):
                v_id = item.get("id", {}).get("videoId")
                if v_id:
                    candidate_ids.add(v_id)
        return self.fetch_videos_by_ids(list(candidate_ids))
        