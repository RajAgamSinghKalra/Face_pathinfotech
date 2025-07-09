"use client"

import { motion, AnimatePresence } from "framer-motion"
import { X, Download, FileText } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import { Label } from "@/components/ui/label"

interface FilterDrawerProps {
  isOpen: boolean
  onClose: () => void
  similarityThreshold: number
  onSimilarityChange: (value: number) => void
}

export default function FilterDrawer({ isOpen, onClose, similarityThreshold, onSimilarityChange }: FilterDrawerProps) {
  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 bg-black/50 backdrop-blur-sm z-40"
            onClick={onClose}
          />

          {/* Drawer */}
          <motion.div
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ type: "spring", damping: 25, stiffness: 300 }}
            className="fixed right-0 top-0 h-full w-80 bg-[#1B263B] shadow-2xl z-50 overflow-y-auto"
          >
            {/* Header */}
            <div className="flex items-center justify-between p-6 border-b border-gray-700">
              <h3 className="text-lg font-semibold text-white">Filters & Controls</h3>
              <Button variant="ghost" size="sm" onClick={onClose} className="text-gray-400 hover:text-white">
                <X className="w-5 h-5" />
              </Button>
            </div>

            <div className="p-6 space-y-8">
              {/* Similarity Threshold */}
              <div>
                <Label className="text-white text-sm font-medium mb-4 block">
                  Similarity Threshold: {similarityThreshold}%
                </Label>
                <Slider
                  value={[similarityThreshold]}
                  onValueChange={(value) => onSimilarityChange(value[0])}
                  min={60}
                  max={100}
                  step={1}
                  className="w-full"
                />
                <div className="flex justify-between text-xs text-gray-400 mt-2">
                  <span>60%</span>
                  <span>100%</span>
                </div>
              </div>

              {/* Date Range */}
              <div>
                <Label className="text-white text-sm font-medium mb-4 block">Date Range</Label>
                <div className="space-y-3">
                  <div>
                    <Label className="text-gray-400 text-xs">From</Label>
                    <input
                      type="date"
                      className="w-full mt-1 px-3 py-2 bg-[#0D1B2A] border border-gray-600 rounded-lg text-white text-sm focus:border-[#00A6FB] focus:outline-none"
                    />
                  </div>
                  <div>
                    <Label className="text-gray-400 text-xs">To</Label>
                    <input
                      type="date"
                      className="w-full mt-1 px-3 py-2 bg-[#0D1B2A] border border-gray-600 rounded-lg text-white text-sm focus:border-[#00A6FB] focus:outline-none"
                    />
                  </div>
                </div>
              </div>

              {/* Group by Person */}
              <div className="flex items-center justify-between">
                <div>
                  <Label className="text-white text-sm font-medium">Group by Person</Label>
                  <p className="text-gray-400 text-xs mt-1">Cluster duplicate faces</p>
                </div>
                <Switch />
              </div>

              {/* Export Options */}
              <div>
                <Label className="text-white text-sm font-medium mb-4 block">Export Options</Label>
                <div className="space-y-3">
                  <Button
                    variant="outline"
                    className="w-full justify-start border-gray-600 text-gray-300 hover:bg-[#0D1B2A] bg-transparent"
                  >
                    <FileText className="w-4 h-4 mr-2" />
                    Export CSV
                  </Button>
                  <Button
                    variant="outline"
                    className="w-full justify-start border-gray-600 text-gray-300 hover:bg-[#0D1B2A] bg-transparent"
                  >
                    <Download className="w-4 h-4 mr-2" />
                    Download ZIP
                  </Button>
                </div>
              </div>

              {/* Quick Filters */}
              <div>
                <Label className="text-white text-sm font-medium mb-4 block">Quick Filters</Label>
                <div className="grid grid-cols-2 gap-2">
                  <Button
                    variant="outline"
                    size="sm"
                    className="border-gray-600 text-gray-300 hover:bg-[#0D1B2A] bg-transparent"
                  >
                    High Match
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    className="border-gray-600 text-gray-300 hover:bg-[#0D1B2A] bg-transparent"
                  >
                    Recent
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    className="border-gray-600 text-gray-300 hover:bg-[#0D1B2A] bg-transparent"
                  >
                    This Year
                  </Button>
                  <Button
                    variant="outline"
                    size="sm"
                    className="border-gray-600 text-gray-300 hover:bg-[#0D1B2A] bg-transparent"
                  >
                    Clear All
                  </Button>
                </div>
              </div>
            </div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  )
}
