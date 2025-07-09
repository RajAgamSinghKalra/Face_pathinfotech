export interface SearchResult {
  id: string
  thumb_url: string
  full_url: string
  similarity: number
  metadata: {
    file_path: string
    file_name?: string
    timestamp: string
    hash: string
    bbox?: number[]
  }
}
