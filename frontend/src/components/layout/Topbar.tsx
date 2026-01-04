import { useState, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Moon, Sun, Bell, Search, Wifi, WifiOff } from "lucide-react"
import { motion } from "framer-motion"

interface TopbarProps {
    hospitalName?: string
}

export function Topbar({ hospitalName = "City General Hospital" }: TopbarProps) {
    const [currentTime, setCurrentTime] = useState(new Date())
    const [isDark, setIsDark] = useState(false)
    const [isOnline, setIsOnline] = useState(navigator.onLine)

    useEffect(() => {
        const timer = setInterval(() => setCurrentTime(new Date()), 1000)
        return () => clearInterval(timer)
    }, [])

    useEffect(() => {
        // Check for dark mode preference
        const isDarkMode = document.documentElement.classList.contains("dark")
        setIsDark(isDarkMode)

        // Online/offline detection
        const handleOnline = () => setIsOnline(true)
        const handleOffline = () => setIsOnline(false)
        window.addEventListener("online", handleOnline)
        window.addEventListener("offline", handleOffline)

        return () => {
            window.removeEventListener("online", handleOnline)
            window.removeEventListener("offline", handleOffline)
        }
    }, [])

    const toggleDarkMode = () => {
        const html = document.documentElement
        if (isDark) {
            html.classList.remove("dark")
            setIsDark(false)
        } else {
            html.classList.add("dark")
            setIsDark(true)
        }
    }

    const formatTime = (date: Date) => {
        return date.toLocaleTimeString("en-US", {
            hour: "2-digit",
            minute: "2-digit",
            second: "2-digit",
            hour12: true,
        })
    }

    const formatDate = (date: Date) => {
        return date.toLocaleDateString("en-US", {
            weekday: "long",
            year: "numeric",
            month: "long",
            day: "numeric",
        })
    }

    return (
        <header className="h-16 border-b bg-card/80 backdrop-blur-xl sticky top-0 z-30">
            <div className="h-full px-6 flex items-center justify-between">
                {/* Left: Hospital Name & Status */}
                <div className="flex items-center gap-4">
                    <div>
                        <h1 className="text-lg font-semibold">{hospitalName}</h1>
                        <p className="text-xs text-muted-foreground">Command Center</p>
                    </div>

                    {/* Connection Status */}
                    <motion.div
                        initial={{ opacity: 0, scale: 0.8 }}
                        animate={{ opacity: 1, scale: 1 }}
                        className={`flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium ${isOnline
                                ? "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400"
                                : "bg-red-500/10 text-red-600 dark:text-red-400"
                            }`}
                    >
                        {isOnline ? (
                            <>
                                <Wifi className="h-3 w-3" />
                                <span>Online</span>
                            </>
                        ) : (
                            <>
                                <WifiOff className="h-3 w-3" />
                                <span>Offline</span>
                            </>
                        )}
                    </motion.div>
                </div>

                {/* Center: Date & Time */}
                <div className="hidden md:flex flex-col items-center">
                    <motion.div
                        key={formatTime(currentTime)}
                        initial={{ opacity: 0, y: -5 }}
                        animate={{ opacity: 1, y: 0 }}
                        className="text-2xl font-bold tracking-tight tabular-nums"
                    >
                        {formatTime(currentTime)}
                    </motion.div>
                    <p className="text-xs text-muted-foreground">{formatDate(currentTime)}</p>
                </div>

                {/* Right: Actions */}
                <div className="flex items-center gap-2">
                    <Button variant="ghost" size="icon" className="hidden sm:flex">
                        <Search className="h-5 w-5" />
                    </Button>

                    <Button variant="ghost" size="icon" className="relative">
                        <Bell className="h-5 w-5" />
                        <span className="absolute top-1.5 right-1.5 h-2 w-2 rounded-full bg-red-500" />
                    </Button>

                    <Button
                        variant="ghost"
                        size="icon"
                        onClick={toggleDarkMode}
                    >
                        <motion.div
                            initial={false}
                            animate={{ rotate: isDark ? 180 : 0 }}
                            transition={{ duration: 0.3 }}
                        >
                            {isDark ? (
                                <Sun className="h-5 w-5" />
                            ) : (
                                <Moon className="h-5 w-5" />
                            )}
                        </motion.div>
                    </Button>
                </div>
            </div>
        </header>
    )
}
