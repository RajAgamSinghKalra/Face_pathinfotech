import { NextRequest, NextResponse } from 'next/server'

export async function GET(req: NextRequest) {
  const { searchParams } = new URL(req.url)
  const path = searchParams.get('path')
  if (!path) {
    return new NextResponse('Missing path', { status: 400 })
  }
  // Proxy to backend FastAPI static endpoint
  const backendUrl = `http://localhost:8000/api/static?path=${encodeURIComponent(path)}`
  const resp = await fetch(backendUrl)
  if (!resp.ok) {
    const status = resp.status === 403 ? 403 : 404
    return new NextResponse('Image not found', { status })
  }
  const contentType = resp.headers.get('content-type') || 'image/jpeg'
  const arrayBuffer = await resp.arrayBuffer()
  return new NextResponse(arrayBuffer, {
    status: 200,
    headers: {
      'Content-Type': contentType,
      'Cache-Control': 'public, max-age=86400',
    },
  })
} 