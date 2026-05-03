import type { Metadata } from 'next'
import { Geist, Outfit, IBM_Plex_Mono } from 'next/font/google'
import { cn } from '@/lib/utils'
import './globals.css'

const geist = Geist({ subsets: ['latin'], variable: '--font-geist' })

const outfit = Outfit({
  variable: '--font-outfit',
  subsets: ['latin'],
  display: 'swap',
})

const ibmPlexMono = IBM_Plex_Mono({
  variable: '--font-ibm-plex-mono',
  subsets: ['latin'],
  weight: ['400', '500', '600'],
  display: 'swap',
})

export const metadata: Metadata = {
  title: 'Spellbook',
  description: 'Web interface for Spellbook entities',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html
      lang="en"
      className={cn(outfit.variable, ibmPlexMono.variable, geist.variable, 'font-sans')}
    >
      <body className="antialiased">{children}</body>
    </html>
  )
}
