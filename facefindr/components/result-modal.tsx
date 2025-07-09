"use client"

import { useState } from "react"
import { motion } from "framer-motion"
import { X, Download, ExternalLink, Calendar, Hash, Folder, ToggleLeft, ToggleRight } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import type { SearchResult } from "@/types"
import React from "react"

interface ResultModalProps {
  result: SearchResult
  onClose: () => void
}

export default function ResultModal({ result, onClose }: ResultModalProps) {
  const [showBoundingBox, setShowBoundingBox] = useState(false)
  const [showFolderModal, setShowFolderModal] = useState(false)

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4"
      onClick={onClose}
    >
      <motion.div
        initial={{ scale: 0.8, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        exit={{ scale: 0.8, opacity: 0 }}
        transition={{ type: "spring", damping: 25, stiffness: 300 }}
        className="bg-[#1B263B] rounded-xl max-w-4xl w-full max-h-[90vh] overflow-hidden shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-700">
          <div className="flex items-center space-x-4">
            <Badge
              className={`
                px-3 py-1 text-sm font-medium
                ${
                  result.similarity >= 95
                    ? "bg-green-500 text-white"
                    : result.similarity >= 85
                      ? "bg-[#00A6FB] text-white"
                      : "bg-[#FF715B] text-white"
                }
              `}
            >
              {result.similarity.toFixed(1)}% Match
            </Badge>
            <h3 className="text-lg font-semibold text-white">{result.metadata.file_path.split("/").pop()}</h3>
          </div>

          <Button variant="ghost" size="sm" onClick={onClose} className="text-gray-400 hover:text-white">
            <X className="w-5 h-5" />
          </Button>
        </div>

        <div className="flex flex-col lg:flex-row">
          {/* Image */}
          <div className="flex-1 p-6">
            <div className="relative bg-black rounded-lg overflow-hidden">
              <img
                src={result.full_url || "/placeholder.svg"}
                alt="Full resolution match"
                className="w-full h-auto max-h-96 object-contain"
                id="modal-face-image"
              />

              {/* Bounding Box Toggle */}
              {showBoundingBox && result.metadata && result.metadata.bbox && Array.isArray(result.metadata.bbox) && (
                <BoundingBoxOverlay bbox={result.metadata.bbox} />
              )}
            </div>

            {/* Controls */}
            <div className="flex items-center justify-between mt-4">
              <Button
                variant="ghost"
                size="sm"
                onClick={() => setShowBoundingBox(!showBoundingBox)}
                className="text-gray-400 hover:text-white"
              >
                {showBoundingBox ? (
                  <ToggleRight className="w-4 h-4 mr-2 text-[#00A6FB]" />
                ) : (
                  <ToggleLeft className="w-4 h-4 mr-2" />
                )}
                Bounding Box
              </Button>

              <div className="flex space-x-2">
                <Button
                  variant="outline"
                  size="sm"
                  className="border-gray-600 text-gray-300 hover:bg-[#0D1B2A] bg-transparent"
                  asChild
                >
                  <a href={result.full_url} download target="_blank" rel="noopener noreferrer">
                    <Download className="w-4 h-4 mr-2" />
                    Download
                  </a>
                </Button>
                <Button
                  variant="outline"
                  size="sm"
                  className="border-gray-600 text-gray-300 hover:bg-[#0D1B2A] bg-transparent"
                  onClick={() => {
                    const filePath = result.metadata.file_path;
                    if (filePath && (filePath.startsWith('http://') || filePath.startsWith('https://'))) {
                      const url = new URL(filePath);
                      const folderUrl = url.origin + url.pathname.substring(0, url.pathname.lastIndexOf('/'));
                      window.open(folderUrl, '_blank');
                    } else {
                      setShowFolderModal(true);
                    }
                  }}
                >
                  <Folder className="w-4 h-4 mr-2" />
                  Open Folder
                </Button>
              </div>
            </div>

            {showFolderModal && (
              <FolderPathModal
                filePath={result.metadata.file_path}
                onClose={() => setShowFolderModal(false)}
              />
            )}
          </div>

          {/* Metadata */}
          <div className="lg:w-80 p-6 border-t lg:border-t-0 lg:border-l border-gray-700">
            <h4 className="text-lg font-semibold text-white mb-4">Details</h4>

            <div className="space-y-4">
              <div>
                <label className="text-sm text-gray-400 block mb-1">File Path</label>
                <p className="text-white text-sm font-mono bg-[#0D1B2A] p-2 rounded break-all">
                  {result.metadata.file_path}
                </p>
              </div>

              <div>
                <label className="text-sm text-gray-400 block mb-1">Timestamp</label>
                <div className="flex items-center text-white">
                  <Calendar className="w-4 h-4 mr-2 text-gray-400" />
                  <span>{new Date(result.metadata.timestamp).toLocaleString()}</span>
                </div>
              </div>

              <div>
                <label className="text-sm text-gray-400 block mb-1">Similarity Score</label>
                <div className="flex items-center justify-between">
                  <div className="flex-1 bg-gray-700 rounded-full h-2 mr-3">
                    <div
                      className="bg-gradient-to-r from-[#FF715B] to-[#00A6FB] h-2 rounded-full transition-all duration-500"
                      style={{ width: `${result.similarity}%` }}
                    />
                  </div>
                  <span className="text-white font-semibold">{result.similarity.toFixed(1)}%</span>
                </div>
              </div>
            </div>

            <Button
              className="w-full mt-6 bg-[#00A6FB] hover:bg-[#00A6FB]/90"
              onClick={() => window.open(result.full_url, "_blank")}
            >
              <ExternalLink className="w-4 h-4 mr-2" />
              Open Original
            </Button>
          </div>
        </div>
      </motion.div>
    </motion.div>
  )
}

function BoundingBoxOverlay({ bbox }: { bbox: number[] }) {
  // bbox: [x1, y1, x2, y2] in pixel coordinates relative to the original image
  // We need to scale these to the displayed image size
  // We'll use a ref to the image to get its displayed size
  const [box, setBox] = useState<{left: number, top: number, width: number, height: number} | null>(null);
  React.useEffect(() => {
    const img = document.getElementById("modal-face-image") as HTMLImageElement | null;
    if (!img) return;
    const updateBox = () => {
      const { naturalWidth, naturalHeight, width: dispW, height: dispH } = img;
      if (naturalWidth && naturalHeight && dispW && dispH) {
        const scaleX = dispW / naturalWidth;
        const scaleY = dispH / naturalHeight;
        const [x1, y1, x2, y2] = bbox;
        setBox({
          left: x1 * scaleX,
          top: y1 * scaleY,
          width: (x2 - x1) * scaleX,
          height: (y2 - y1) * scaleY,
        });
      }
    };
    updateBox();
    window.addEventListener('resize', updateBox);
    return () => window.removeEventListener('resize', updateBox);
  }, [bbox]);
  if (!box) return null;
  return (
    <div
      className="absolute border-2 border-[#00A6FB] pointer-events-none z-10"
      style={{
        left: box.left,
        top: box.top,
        width: box.width,
        height: box.height,
      }}
    >
      <div className="absolute -top-6 left-0 bg-[#00A6FB] text-white px-2 py-1 rounded text-xs">
        Detected Face
      </div>
    </div>
  );
}

function FolderPathModal({ filePath, onClose }: { filePath: string, onClose: () => void }) {
  // Extract folder path
  let folderPath = filePath;
  if (folderPath && (folderPath.includes("/") || folderPath.includes("\\"))) {
    folderPath = folderPath.replace(/[\\/][^\\/]*$/, "");
  }
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60">
      <div className="bg-[#1B263B] p-6 rounded-lg shadow-lg max-w-md w-full">
        <h4 className="text-lg font-semibold text-white mb-2">Open Folder</h4>
        <p className="text-gray-300 mb-2">Copy and paste this path into your file explorer:</p>
        <div className="flex items-center mb-4">
          <input
            type="text"
            value={folderPath}
            readOnly
            className="flex-1 bg-gray-800 text-white px-2 py-1 rounded mr-2 font-mono text-xs"
            onFocus={e => e.target.select()}
          />
          <Button
            size="sm"
            className="bg-[#00A6FB] hover:bg-[#00A6FB]/90"
            onClick={() => {
              navigator.clipboard.writeText(folderPath)
            }}
          >
            Copy Path
          </Button>
        </div>
        <div className="flex justify-end">
          <Button size="sm" variant="ghost" onClick={onClose} className="text-gray-400 hover:text-white">Close</Button>
        </div>
      </div>
    </div>
  )
}
