"use client"

import { useState } from "react"
import { motion } from "framer-motion"
import HeroSection from "@/components/hero-section"
import UploadProgress from "@/components/upload-progress"
import ResultsGallery from "@/components/results-gallery"
import type { SearchResult } from "@/types"

export default function Home() {
  const [isUploading, setIsUploading] = useState(false)
  const [isComplete, setIsComplete] = useState(false)
  const [results, setResults] = useState<SearchResult[]>([])
  const [error, setError] = useState<string | null>(null)

  const handleFileUpload = async (file: File) => {
    setIsUploading(true)
    setError(null)
    setResults([])
    setIsComplete(false)

    try {
      const formData = new FormData()
      formData.append("file", file)
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
      if (!res.ok) throw new Error(data.error || data.message || "Unknown error")

      const seen = new Set()
      const mapped: SearchResult[] = (data.results || []).flatMap((face: any) => {
        return (face.matches || []).map((m: any) => {
          const uniqueKey = `${m.id}_${m.cropped_face_path}`
          if (seen.has(uniqueKey)) return null
          seen.add(uniqueKey)

          let thumb_url = m.cropped_face_path || "/placeholder-user.jpg"
          let full_url = m.original_image || ""
          // Fall back to metadata field if original_image not present
          if (!full_url && m.metadata) {
            try {
              const meta = typeof m.metadata === "string" ? JSON.parse(m.metadata) : m.metadata
              if (meta && typeof meta.original_image === "string") {
                full_url = meta.original_image
              }
            } catch {}
          }
          if (!full_url) {
            full_url = m.cropped_face_path || ""
          }

          if (thumb_url && !thumb_url.startsWith("http")) {
            thumb_url = `/api/image-proxy?path=${encodeURIComponent(thumb_url)}`
          }
          if (full_url && !full_url.startsWith("http")) {
            full_url = `/api/image-proxy?path=${encodeURIComponent(full_url)}`
          }

          let similarity = 0
          if (typeof m.dist === "number") {
            similarity = Math.max(0, Math.min(100, (1 - m.dist) * 100))
            similarity = Math.round(similarity * 10) / 10
          }

          let bbox: number[] | undefined = undefined
          if (Array.isArray(m.bbox_x1) && m.bbox_x1.length === 4) {
            bbox = m.bbox_x1
          } else if (
            typeof m.bbox_x1 === "number" &&
            typeof m.bbox_y1 === "number" &&
            typeof m.bbox_x2 === "number" &&
            typeof m.bbox_y2 === "number"
          ) {
            bbox = [m.bbox_x1, m.bbox_y1, m.bbox_x2, m.bbox_y2]
          }

          let file_name = m.cropped_face_path ? m.cropped_face_path.split(/[\\/]/).pop() : ""

          let hash = ""
          if (m.hash) {
            hash = m.hash
          } else if (m.metadata) {
            let metaObj = m.metadata
            if (typeof metaObj === "string") {
              try {
                metaObj = JSON.parse(metaObj)
              } catch {}
            }
            if (metaObj && typeof metaObj === "object" && metaObj.hash) {
              hash = metaObj.hash
            }
          }

          return {
            id: m.id?.toString() || Math.random().toString(),
            thumb_url,
            full_url,
            similarity,
            metadata: {
              file_path:
                m.original_image ||
                (() => {
                  if (m.metadata) {
                    try {
                      const meta = typeof m.metadata === "string" ? JSON.parse(m.metadata) : m.metadata
                      if (meta && typeof meta.original_image === "string") {
                        return meta.original_image
                      }
                    } catch {}
                  }
                  return ""
                })(),
              file_name,
              timestamp: m.metadata?.timestamp || new Date().toISOString(),
              hash,
              bbox,
            },
          }
        }).filter(Boolean)
      })

      const deduped = mapped.filter((_, i) => i % 2 === 0)
      setResults(deduped)
      setIsComplete(true)
    } catch (err: any) {
      setError(err.message || "Failed to connect to backend API.")
    } finally {
      setIsUploading(false)
    }
  }

  const handleReset = () => {
    setResults([])
    setIsComplete(false)
    setError(null)
  }

  return (
    <div className="min-h-screen bg-gradient-to-br from-[#0D1B2A] to-[#1B263B]">
      {!isUploading && !isComplete && <HeroSection onFileUpload={handleFileUpload} />}

      {isUploading && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="container mx-auto px-4 py-8"
        >
          {error && <div className="bg-red-500 text-white p-4 rounded-lg mb-4">{error}</div>}
          <UploadProgress />
        </motion.div>
      )}

      {!isUploading && isComplete && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="container mx-auto px-4 py-8"
        >
          {error && <div className="bg-red-500 text-white p-4 rounded-lg mb-4">{error}</div>}
          <ResultsGallery results={results} isComplete={isComplete} onReset={handleReset} />
        </motion.div>
      )}
    </div>
  )
}
