"use client"

import { useCallback, useState } from "react"
import { motion } from "framer-motion"
import { useDropzone } from "react-dropzone"
import { Upload, Zap, Shield } from "lucide-react"
import { Button } from "@/components/ui/button"

interface HeroSectionProps {
  onFileUpload: (file: File) => void
}

export default function HeroSection({ onFileUpload }: HeroSectionProps) {
  const [isDragActive, setIsDragActive] = useState(false)

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (acceptedFiles.length > 0) {
        onFileUpload(acceptedFiles[0])
      }
    },
    [onFileUpload],
  )

  const { getRootProps, getInputProps } = useDropzone({
    onDrop,
    accept: {
      "image/*": [".jpeg", ".jpg", ".png", ".webp"],
    },
    multiple: false,
    onDragEnter: () => setIsDragActive(true),
    onDragLeave: () => setIsDragActive(false),
  })

  return (
    <div className="min-h-screen flex items-center justify-center px-4">
      <div className="max-w-4xl mx-auto text-center">
        <motion.div initial={{ opacity: 0, y: 40 }} animate={{ opacity: 1, y: 0 }} transition={{ duration: 0.6 }}>
          <h1 className="text-5xl md:text-7xl font-bold text-white mb-6 font-serif">
            Find every photo of that <span className="text-[#00A6FB]">face</span> — in seconds.
          </h1>

          <p className="text-xl text-gray-300 mb-12 max-w-2xl mx-auto">
            Powered by private on-prem AI; no images leave your browser. Upload a photo and instantly discover every
            match in our database.
          </p>

          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.2 }}
            className="mb-16"
          >
            <div
              {...getRootProps()}
              className={`
                relative border-2 border-dashed rounded-xl p-12 cursor-pointer transition-all duration-300
                ${
                  isDragActive
                    ? "border-[#00A6FB] bg-[#00A6FB]/10 scale-105"
                    : "border-gray-600 hover:border-[#00A6FB] hover:bg-[#00A6FB]/5"
                }
              `}
            >
              <input {...getInputProps()} />

              <motion.div
                animate={{
                  scale: isDragActive ? 1.1 : 1,
                }}
                className="flex flex-col items-center"
              >
                <div className="w-16 h-16 rounded-full bg-[#00A6FB]/20 flex items-center justify-center mb-6">
                  <Upload className="w-8 h-8 text-[#00A6FB]" />
                </div>

                <h3 className="text-2xl font-semibold text-white mb-2">Drop your photo here</h3>
                <p className="text-gray-400 mb-6">or click to browse your files</p>

                <Button
                  size="lg"
                  className="bg-[#00A6FB] hover:bg-[#00A6FB]/90 text-white px-8 py-3 rounded-lg font-medium"
                >
                  Choose Photo
                </Button>
              </motion.div>

              {/* Pulse animation */}
              <motion.div
                animate={{
                  scale: [1, 1.03, 1],
                }}
                transition={{
                  duration: 6,
                  repeat: Number.POSITIVE_INFINITY,
                  ease: "easeInOut",
                }}
                className="absolute inset-0 border-2 border-[#00A6FB]/30 rounded-xl pointer-events-none"
              />
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 40 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.6, delay: 0.4 }}
            className="grid md:grid-cols-3 gap-8 text-left"
          >
            <div className="flex items-start space-x-4">
              <div className="w-12 h-12 rounded-lg bg-[#FF715B]/20 flex items-center justify-center flex-shrink-0">
                <Zap className="w-6 h-6 text-[#FF715B]" />
              </div>
              <div>
                <h4 className="text-lg font-semibold text-white mb-2">Lightning Fast</h4>
                <p className="text-gray-400">Advanced ArcFace AI processes thousands of faces in seconds</p>
              </div>
            </div>

            <div className="flex items-start space-x-4">
              <div className="w-12 h-12 rounded-lg bg-[#00A6FB]/20 flex items-center justify-center flex-shrink-0">
                <Shield className="w-6 h-6 text-[#00A6FB]" />
              </div>
              <div>
                <h4 className="text-lg font-semibold text-white mb-2">Private & Secure</h4>
                <p className="text-gray-400">All processing happens locally - your photos never leave your device</p>
              </div>
            </div>

            <div className="flex items-start space-x-4">
              <div className="w-12 h-12 rounded-lg bg-[#FF715B]/20 flex items-center justify-center flex-shrink-0">
                <Upload className="w-6 h-6 text-[#FF715B]" />
              </div>
              <div>
                <h4 className="text-lg font-semibold text-white mb-2">Easy to Use</h4>
                <p className="text-gray-400">Simply drag, drop, and discover - no technical knowledge required</p>
              </div>
            </div>
          </motion.div>
        </motion.div>
      </div>
    </div>
  )
}
