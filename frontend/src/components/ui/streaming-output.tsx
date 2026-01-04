import { useEffect, useState, useRef } from "react"
import { motion, AnimatePresence } from "framer-motion"
import { cn } from "@/lib/utils"

interface StreamingOutputProps {
    content: string
    className?: string
    speed?: number
    onComplete?: () => void
}

export function StreamingOutput({
    content,
    className,
    speed = 15,
    onComplete
}: StreamingOutputProps) {
    const [displayedContent, setDisplayedContent] = useState("")
    const [isTyping, setIsTyping] = useState(false)
    const indexRef = useRef(0)
    const timerRef = useRef<NodeJS.Timeout | null>(null)

    useEffect(() => {
        // Reset state when content changes significantly or is cleared
        if (!content) {
            setDisplayedContent("")
            indexRef.current = 0
            setIsTyping(false)
            if (timerRef.current) clearInterval(timerRef.current)
            return
        }

        // If we already displayed this content (or close to it), don't re-stream
        if (displayedContent === content) {
            setIsTyping(false)
            return
        }

        setIsTyping(true)

        // Clear any existing timer
        if (timerRef.current) clearInterval(timerRef.current)

        timerRef.current = setInterval(() => {
            const currentIndex = indexRef.current

            if (currentIndex < content.length) {
                // Boost speed for long content to vary rhythm and prevents waiting too long
                const jump = content.length > 500 ? 3 : 1

                setDisplayedContent(prev => content.slice(0, currentIndex + jump))
                indexRef.current += jump
            } else {
                setDisplayedContent(content)
                setIsTyping(false)
                if (timerRef.current) clearInterval(timerRef.current)
                onComplete?.()
            }
        }, speed)

        return () => {
            if (timerRef.current) clearInterval(timerRef.current)
        }
    }, [content, speed, onComplete])

    return (
        <div className={cn("relative font-mono text-sm leading-relaxed", className)}>
            <span className="whitespace-pre-wrap">{displayedContent}</span>
            <AnimatePresence>
                {isTyping && (
                    <motion.span
                        initial={{ opacity: 0 }}
                        animate={{ opacity: 1 }}
                        exit={{ opacity: 0 }}
                        transition={{ repeat: Infinity, duration: 0.8, ease: "easeInOut" }}
                        className="inline-block w-2 h-4 ml-1 align-middle bg-primary"
                    />
                )}
            </AnimatePresence>
        </div>
    )
}
