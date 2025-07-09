"use client"

import { useState } from "react"
import { motion } from "framer-motion"
import HeroSection from "@/components/hero-section"
import UploadProgress from "@/components/upload-progress"
import ResultsGallery from "@/components/results-gallery"
import type { SearchResult } from "@/types"

export default function Home() {
  const [searchId, setSearchId] = useState<string | null>(null)
  const [isUploading, setIsUploading] = useState(false)
  const [results, setResults] = useState<SearchResult[]>([])
  const [isComplete, setIsComplete] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleFileUpload = async (file: File) => {
    setIsUploading(true)
    setError(null)
    setResults([])
    setIsComplete(false)

    const formData = new FormData()
    formData.append("file", file)

    try {
      // Call the real backend API
      const apiUrl = `http://${window.location.hostname}:8000/api/search`
      const res = await fetch(apiUrl, {
        method: "POST",
        body: formData,
      })

      let data
      try {
        data = await res.json()
      } catch (err) {
        throw new Error("Could not parse response from server.")
      }

      if (!res.ok) {
        throw new Error(data.error || data.message || "Unknown error")
      }

      // Map backend response to SearchResult[]
      const seen = new Set()
      const mapped: SearchResult[] = (data.results || []).flatMap((face: any) => {
        return (face.matches || []).map((m: any) => {
          const uniqueKey = `${m.id}_${m.cropped_face_path}`
          if (seen.has(uniqueKey)) return null
          seen.add(uniqueKey)

          // Use the backend's relative path directly
          let thumb_url = m.cropped_face_path || "/placeholder-user.jpg"
          let full_url = m.original_image || m.cropped_face_path || ""
          
          // Use Next.js proxy route for images
          if (thumb_url && !thumb_url.startsWith("http")) {
            thumb_url = `/api/image-proxy?path=${encodeURIComponent(thumb_url)}`
          }
          if (full_url && !full_url.startsWith("http")) {
            full_url = `/api/image-proxy?path=${encodeURIComponent(full_url)}`
          }

          // Calculate similarity from distance
          let similarity = 0
          if (typeof m.dist === "number") {
            similarity = Math.max(0, Math.min(100, ((1 - m.dist) * 100)))
            similarity = Math.round(similarity * 10) / 10
          }

          // Extract file name from cropped_face_path
          let file_name = m.cropped_face_path ? m.cropped_face_path.split(/[\\/]/).pop() : ""

          return {
            id: m.id?.toString() || Math.random().toString(),
            thumb_url,
            full_url,
            similarity,
            metadata: {
              file_path: m.original_image || "",
              file_name,
              timestamp: m.metadata?.timestamp || new Date().toISOString(),
              hash: m.hash || "",
            },
          }
        }).filter(Boolean)
      })

      setResults(mapped)
      setIsComplete(true)
      setSearchId(`search_${Date.now()}`)
    } catch (error: any) {
      console.error("Upload failed:", error)
      setError(error.message || "Failed to connect to backend API. Make sure the backend server is running on port 8000.")
    } finally {
      setIsUploading(false)
    }
  }

  const handleReset = () => {
    setSearchId(null)
    setIsUploading(false)
    setResults([])
    setIsComplete(false)
    setError(null)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#0D1B2A] to-[#1B263B]">
      {!searchId && !isUploading && <HeroSection onFileUpload={handleFileUpload} />}

      {(isUploading || searchId) && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="container mx-auto px-4 py-8"
        >
          {error && (
            <div className="bg-red-500 text-white p-4 rounded-lg mb-4">
              {error}
            </div>
          )}
          {isUploading && <UploadProgress />}
          {results.length > 0 && <ResultsGallery results={results} isComplete={isComplete} onReset={handleReset} />}
        </motion.div>
      )}
    </div>
  )
}
