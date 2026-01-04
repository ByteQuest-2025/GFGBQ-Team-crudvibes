import { cn } from "@/lib/utils"
import { Logo } from "@/components/Logo"
import { Button } from "@/components/ui/button"
import { Link, useLocation } from "react-router-dom"
import { motion, AnimatePresence } from "framer-motion"
import {
    LayoutDashboard,
    Bot,
    BarChart3,
    Settings,
    LogOut,
    ChevronLeft,
    ChevronRight,
    Activity,
    Users,
    Bed,
} from "lucide-react"
import { useState } from "react"

interface NavItem {
    label: string
    href: string
    icon: React.ElementType
}

const mainNavItems: NavItem[] = [
    { label: "Dashboard", href: "/dashboard", icon: LayoutDashboard },
    { label: "AI Agents", href: "/agents", icon: Bot },
    { label: "Analytics", href: "/analytics", icon: BarChart3 },
]

const quickStats = [
    { label: "Emergency", value: "24", icon: Activity, color: "text-red-500" },
    { label: "ICU Beds", value: "18", icon: Bed, color: "text-amber-500" },
    { label: "Staff", value: "142", icon: Users, color: "text-emerald-500" },
]

interface SidebarProps {
    onLogout: () => void
}

export function Sidebar({ onLogout }: SidebarProps) {
    const [collapsed, setCollapsed] = useState(false)
    const location = useLocation()

    return (
        <motion.aside
            initial={false}
            animate={{ width: collapsed ? 80 : 260 }}
            transition={{ duration: 0.3, ease: "easeInOut" }}
            className={cn(
                "fixed left-0 top-0 z-40 h-screen border-r bg-card flex flex-col",
                "transition-shadow duration-300"
            )}
        >
            {/* Logo */}
            <div className="flex h-16 items-center justify-between px-4 border-b">
                <AnimatePresence mode="wait">
                    {!collapsed && (
                        <motion.div
                            initial={{ opacity: 0 }}
                            animate={{ opacity: 1 }}
                            exit={{ opacity: 0 }}
                            transition={{ duration: 0.2 }}
                        >
                            <Link to="/dashboard">
                                <Logo size="md" />
                            </Link>
                        </motion.div>
                    )}
                </AnimatePresence>
                {collapsed && (
                    <Link to="/dashboard" className="mx-auto">
                        <Logo size="sm" showText={false} />
                    </Link>
                )}
            </div>

            {/* Navigation */}
            <nav className="flex-1 px-3 py-4 space-y-1">
                {mainNavItems.map((item) => {
                    const isActive = location.pathname === item.href
                    return (
                        <Link key={item.href} to={item.href}>
                            <motion.div
                                whileHover={{ x: 4 }}
                                whileTap={{ scale: 0.98 }}
                                className={cn(
                                    "flex items-center gap-3 px-3 py-2.5 rounded-lg transition-colors",
                                    isActive
                                        ? "bg-primary text-primary-foreground"
                                        : "text-muted-foreground hover:bg-muted hover:text-foreground"
                                )}
                            >
                                <item.icon className="h-5 w-5 flex-shrink-0" />
                                <AnimatePresence mode="wait">
                                    {!collapsed && (
                                        <motion.span
                                            initial={{ opacity: 0, width: 0 }}
                                            animate={{ opacity: 1, width: "auto" }}
                                            exit={{ opacity: 0, width: 0 }}
                                            className="text-sm font-medium whitespace-nowrap overflow-hidden"
                                        >
                                            {item.label}
                                        </motion.span>
                                    )}
                                </AnimatePresence>
                            </motion.div>
                        </Link>
                    )
                })}
            </nav>

            {/* Quick Stats */}
            <AnimatePresence mode="wait">
                {!collapsed && (
                    <motion.div
                        initial={{ opacity: 0, height: 0 }}
                        animate={{ opacity: 1, height: "auto" }}
                        exit={{ opacity: 0, height: 0 }}
                        className="px-3 py-4 border-t"
                    >
                        <p className="text-xs font-semibold text-muted-foreground uppercase tracking-wider mb-3 px-3">
                            Quick Stats
                        </p>
                        <div className="space-y-2">
                            {quickStats.map((stat) => (
                                <div
                                    key={stat.label}
                                    className="flex items-center gap-3 px-3 py-2 rounded-lg bg-muted/50"
                                >
                                    <stat.icon className={cn("h-4 w-4", stat.color)} />
                                    <span className="text-sm text-muted-foreground flex-1">
                                        {stat.label}
                                    </span>
                                    <span className="text-sm font-semibold">{stat.value}</span>
                                </div>
                            ))}
                        </div>
                    </motion.div>
                )}
            </AnimatePresence>

            {/* Footer */}
            <div className="border-t p-3 space-y-1">
                <Button
                    variant="ghost"
                    size={collapsed ? "icon" : "default"}
                    className={cn(
                        "w-full",
                        !collapsed && "justify-start gap-3"
                    )}
                    onClick={onLogout}
                >
                    <LogOut className="h-5 w-5" />
                    {!collapsed && <span>Logout</span>}
                </Button>

                <Button
                    variant="ghost"
                    size="icon"
                    className="w-full"
                    onClick={() => setCollapsed(!collapsed)}
                >
                    {collapsed ? (
                        <ChevronRight className="h-5 w-5" />
                    ) : (
                        <ChevronLeft className="h-5 w-5" />
                    )}
                </Button>
            </div>
        </motion.aside>
    )
}
