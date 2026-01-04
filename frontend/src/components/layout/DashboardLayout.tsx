import { Sidebar } from "./Sidebar"
import { Topbar } from "./Topbar"
import { motion } from "framer-motion"
import { useState, useEffect } from "react"

interface DashboardLayoutProps {
    children: React.ReactNode
    onLogout: () => void
}

export function DashboardLayout({ children, onLogout }: DashboardLayoutProps) {
    const [sidebarWidth, setSidebarWidth] = useState(260)

    useEffect(() => {
        // Listen for sidebar collapse changes via CSS variable or resize observer
        const checkWidth = () => {
            const sidebar = document.querySelector("aside")
            if (sidebar) {
                setSidebarWidth(sidebar.offsetWidth)
            }
        }

        const observer = new ResizeObserver(checkWidth)
        const sidebar = document.querySelector("aside")
        if (sidebar) {
            observer.observe(sidebar)
        }

        return () => observer.disconnect()
    }, [])

    return (
        <div className="min-h-screen bg-background">
            <Sidebar onLogout={onLogout} />

            <motion.div
                initial={false}
                animate={{ marginLeft: sidebarWidth }}
                transition={{ duration: 0.3, ease: "easeInOut" }}
                className="flex flex-col min-h-screen"
            >
                <Topbar />

                <main className="flex-1 p-6">
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.4, ease: "easeOut" }}
                    >
                        {children}
                    </motion.div>
                </main>
            </motion.div>
        </div>
    )
}
