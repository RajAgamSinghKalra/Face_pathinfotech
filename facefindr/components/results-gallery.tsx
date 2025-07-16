"use client"

import { useState } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { ArrowLeft, Filter, Download, Calendar, Hash, Folder } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Badge } from "@/components/ui/badge"
import ResultModal from "@/components/result-modal"
import FilterDrawer from "@/components/filter-drawer"
import type { SearchResult } from "@/types"

interface ResultsGalleryProps {
  results: SearchResult[]
  isComplete: boolean
  onReset: () => void
}

export default function ResultsGallery({ results, isComplete, onReset }: ResultsGalleryProps) {
  const [selectedResult, setSelectedResult] = useState<SearchResult | null>(null)
  const [showFilters, setShowFilters] = useState(false)
  const [similarityThreshold, setSimilarityThreshold] = useState(70)

  const filteredResults = results.filter((result) => result.similarity >= similarityThreshold)

  return (
    <div className="max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex items-center justify-between mb-8">
        <div className="flex items-center space-x-4">
          <Button variant="ghost" size="sm" onClick={onReset} className="text-gray-400 hover:text-white">
            <ArrowLeft className="w-4 h-4 mr-2" />
            New Search
          </Button>
          <div>
            <h2 className="text-2xl font-bold text-white">Found {filteredResults.length} matches</h2>
            <p className="text-gray-400">{isComplete ? "Search complete" : "Still searching..."}</p>
          </div>
        </div>

        <div className="flex items-center space-x-3">
          <Button
            variant="outline"
            size="sm"
            onClick={() => setShowFilters(true)}
            className="border-gray-600 text-gray-300 hover:bg-[#1B263B]"
          >
            <Filter className="w-4 h-4 mr-2" />
            Filters
          </Button>
          <Button
            variant="outline"
            size="sm"
            className="border-gray-600 text-gray-300 hover:bg-[#1B263B] bg-transparent"
          >
            <Download className="w-4 h-4 mr-2" />
            Export
          </Button>
        </div>
      </div>

      {/* Results Grid */}
      {filteredResults.length > 0 ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4">
          {filteredResults.map((result, index) => (
            <motion.div
              key={result.id}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: index * 0.04 }}
              className="group cursor-pointer"
              onClick={() => setSelectedResult(result)}
            >
              <div className="relative aspect-square bg-[#1B263B] rounded-lg overflow-hidden shadow-lg hover:shadow-xl transition-all duration-300 hover:scale-104">
                <img src={result.thumb_url || "/placeholder.svg"} alt="Match" className="w-full h-full object-cover" />

                {/* Similarity Badge */}
                <div className="absolute top-3 right-3">
                  <Badge
                    className={`
                      px-2 py-1 text-xs font-medium transition-transform duration-300 group-hover:-translate-y-1
                      ${
                        result.similarity >= 95
                          ? "bg-green-500/90 text-white"
                          : result.similarity >= 85
                            ? "bg-[#00A6FB]/90 text-white"
                            : "bg-[#FF715B]/90 text-white"
                      }
                    `}
                  >
                    {result.similarity.toFixed(1)}%
                  </Badge>
                </div>

                {/* Hover Overlay */}
                <div className="absolute inset-0 bg-black/50 opacity-0 group-hover:opacity-100 transition-opacity duration-300 flex items-center justify-center">
                  <div className="text-white text-center">
                    <Folder className="w-6 h-6 mx-auto mb-2" />
                    <p className="text-sm">View Details</p>
                  </div>
                </div>
              </div>

              {/* Metadata Preview */}
              <div className="mt-2 px-1">
                <p className="text-xs text-gray-400 truncate">{result.metadata.file_name || result.thumb_url.split("/").pop()?.split("?")[0]}</p>
                <div className="flex items-center justify-between mt-1">
                  <span className="text-xs text-gray-500 flex items-center">
                    <Calendar className="w-3 h-3 mr-1" />
                    {new Date(result.metadata.timestamp).toLocaleDateString()}
                  </span>
                  <span className="text-xs text-gray-500 flex items-center">
                    <Hash className="w-3 h-3 mr-1" />
                    {result.metadata.hash.slice(0, 6)}
                  </span>
                </div>
              </div>
            </motion.div>
          ))}
        </div>
      ) : (
        <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="text-center py-16">
          <div className="text-6xl mb-4">🔍</div>
          <h3 className="text-xl font-semibold text-white mb-2">No matches found</h3>
          <p className="text-gray-400 mb-6">Try adjusting your similarity threshold or upload a different photo</p>
          <Button onClick={onReset} className="bg-[#00A6FB] hover:bg-[#00A6FB]/90">
            Try Another Photo
          </Button>
        </motion.div>
      )}

      {/* Modal */}
      <AnimatePresence>
        {selectedResult && <ResultModal result={selectedResult} onClose={() => setSelectedResult(null)} />}
      </AnimatePresence>

      {/* Filter Drawer */}
      <FilterDrawer
        isOpen={showFilters}
        onClose={() => setShowFilters(false)}
        similarityThreshold={similarityThreshold}
        onSimilarityChange={setSimilarityThreshold}
      />
    </div>
  )
}
