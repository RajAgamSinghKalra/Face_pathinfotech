/// <reference types="react" />
"use client"

import React, { useState, useRef } from "react"
import UploadProgress from "@/components/upload-progress"
import ResultsGallery from "@/components/results-gallery"
import { Button } from "@/components/ui/button"
import { SearchResult } from "@/types"

export default function SearchPage() {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [results, setResults] = useState<SearchResult[]>([])
  const [isLoading, setIsLoading] = useState(false)
  const [isComplete, setIsComplete] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const inputRef = useRef<HTMLInputElement>(null)

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setSelectedFile(e.target.files[0])
      setResults([])
      setError(null)
      setIsComplete(false)
    }
  }

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault()
    console.log('Submit handler called', { selectedFile, isLoading });
    if (!selectedFile) return
    setIsLoading(true)
    setError(null)
    setResults([])
    setIsComplete(false)
    try {
      const formData = new FormData()
      formData.append("file", selectedFile)
      // Use window.location.hostname for local/prod compatibility
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
      console.log("API response:", data)
      if (!res.ok) throw new Error(data.error || "Unknown error")
      // Map backend response to SearchResult[]
      const seen = new Set();
      let firstMatchLogged = false;
      const mapped: SearchResult[] = (data.results || []).flatMap((face: any) => {
        return (face.matches || []).map((m: any) => {
          console.log('Match metadata:', m.metadata);
          if (!firstMatchLogged) {
            console.log('First match object:', m);
            firstMatchLogged = true;
          }
          const uniqueKey = `${m.id}_${m.cropped_face_path}`;
          if (seen.has(uniqueKey)) return null;
          seen.add(uniqueKey);
          // Use the backend's relative path directly
          let thumb_url = m.cropped_face_path || "/placeholder-user.jpg";
          let full_url = m.original_image || m.cropped_face_path || "";
          // Instead of rewriting to /api/static, use a Next.js proxy route
          if (thumb_url && !thumb_url.startsWith("http")) {
            thumb_url = `/api/image-proxy?path=${encodeURIComponent(thumb_url)}`;
          }
          if (full_url && !full_url.startsWith("http")) {
            full_url = `/api/image-proxy?path=${encodeURIComponent(full_url)}`;
          }
          // Clamp and round similarity
          let similarity = 0;
          if (typeof m.dist === "number") {
            similarity = Math.max(0, Math.min(100, ((1 - m.dist) * 100)));
            similarity = Math.round(similarity * 10) / 10;
          }
          // Pass bbox from match if available
          let bbox = undefined;
          if (Array.isArray(m.bbox_x1) && m.bbox_x1.length === 4) {
            bbox = m.bbox_x1;
          } else if (
            typeof m.bbox_x1 === "number" &&
            typeof m.bbox_y1 === "number" &&
            typeof m.bbox_x2 === "number" &&
            typeof m.bbox_y2 === "number"
          ) {
            bbox = [m.bbox_x1, m.bbox_y1, m.bbox_x2, m.bbox_y2];
          }
          // Extract file name from cropped_face_path
          let file_name = m.cropped_face_path ? m.cropped_face_path.split(/[\\/]/).pop() : "";
          // Parse hash from metadata if it's a JSON string
          let hash = "";
          if (m.hash) {
            hash = m.hash;
          } else if (m.metadata) {
            let metaObj = m.metadata;
            if (typeof metaObj === "string") {
              try {
                metaObj = JSON.parse(metaObj);
              } catch {}
            }
            if (metaObj && typeof metaObj === "object" && metaObj.hash) {
              hash = metaObj.hash;
            }
          }
          return {
            id: m.id?.toString() || Math.random().toString(),
            thumb_url,
            full_url,
            similarity,
            metadata: {
              file_path: m.original_image || "",
              file_name,
              timestamp: m.metadata?.timestamp || new Date().toISOString(),
              hash,
              bbox,
            },
          };
        }).filter(Boolean);
      });
      // Drop every other image from the mapped results
      const deduped = mapped.filter((_, i) => i % 2 === 0);
      setResults(deduped)
      setIsComplete(true)
    } catch (err: any) {
      setError(err.message || "Failed to connect to backend API.")
    } finally {
      setIsLoading(false)
    }
  }

  const handleReset = () => {
    setSelectedFile(null)
    setResults([])
    setIsComplete(false)
    setError(null)
    if (inputRef.current) inputRef.current.value = ""
  }

  return (
    <div className="min-h-screen bg-[#0D1B2A] py-12 px-4">
      <div className="max-w-2xl mx-auto">
        <h1 className="text-3xl font-bold text-white mb-6 text-center">Face Similarity Search</h1>
        <form onSubmit={handleSubmit} className="bg-[#1B263B] rounded-xl p-8 shadow-lg flex flex-col items-center">
          <input
            type="file"
            accept="image/*"
            onChange={handleFileChange}
            ref={inputRef}
            className="mb-4 text-white"
          />
          <Button type="submit" /*disabled={!selectedFile || isLoading}*/ className="bg-[#00A6FB] hover:bg-[#00A6FB]/90">
            {isLoading ? "Uploading..." : "Upload & Search"}
          </Button>
          {error && <p className="text-red-500 mt-4">{error}</p>}
        </form>
        {isLoading && <UploadProgress />}
        {!isLoading && isComplete && (
          <ResultsGallery results={results} isComplete={isComplete} onReset={handleReset} />
        )}
        {!isLoading && !isComplete && selectedFile && (
          <div className="text-center mt-8">
            <Button onClick={handleReset} className="bg-[#00A6FB] hover:bg-[#00A6FB]/90">
              Try Another Photo
            </Button>
          </div>
        )}
      </div>
    </div>
  )
} 