"use client"

import { motion } from "framer-motion"
import { Loader2 } from "lucide-react"

export default function UploadProgress() {
  return (
    <div className="max-w-2xl mx-auto text-center mb-12">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-[#1B263B] rounded-xl p-8 shadow-lg"
      >
        <div className="flex items-center justify-center mb-6">
          <Loader2 className="w-8 h-8 text-[#00A6FB] animate-spin mr-3" />
          <h3 className="text-xl font-semibold text-white">Processing your photo...</h3>
        </div>

        <div className="w-full bg-gray-700 rounded-full h-3 mb-4">
          <motion.div
            className="bg-gradient-to-r from-[#00A6FB] to-[#FF715B] h-3 rounded-full"
            initial={{ width: 0 }}
            animate={{ width: "100%" }}
            transition={{ duration: 2, ease: "easeInOut" }}
          />
        </div>

        <p className="text-gray-400">Analyzing facial features and searching database...</p>
      </motion.div>

      {/* Skeleton grid for upcoming results */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.5 }}
        className="grid grid-cols-2 md:grid-cols-4 gap-4 mt-8"
      >
        {[...Array(8)].map((_, i) => (
          <div key={i} className="aspect-square bg-[#1B263B] rounded-lg animate-pulse" />
        ))}
      </motion.div>
    </div>
  )
}
