import type React from "react"
import type { Metadata } from "next"
import "./globals.css"

// Use system fonts to avoid fetching from Google Fonts
const fontClasses = "font-sans antialiased"

export const metadata: Metadata = {
  title: "FaceFindr - Find every photo of that face",
  description: "Upload a photo and instantly discover every match in our database using advanced AI face recognition.",
    generator: 'v0.dev'
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en">
      <body className={fontClasses}>{children}</body>
    </html>
  )
}
