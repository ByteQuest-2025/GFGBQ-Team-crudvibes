import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Skeleton } from "@/components/ui/skeleton"
import { motion } from "framer-motion"
import { useState, useEffect } from "react"
import {
    LineChart,
    Line,
    AreaChart,
    Area,
    BarChart,
    Bar,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    Legend,
} from "recharts"
import {
    Activity,
    Bed,
    Users,
    AlertTriangle,
    TrendingUp,
    TrendingDown,
    ArrowUpRight,
    ArrowDownRight,
} from "lucide-react"

// Animated counter hook
function useAnimatedCounter(end: number, duration: number = 1000) {
    const [count, setCount] = useState(0)

    useEffect(() => {
        let startTime: number
        let animationFrame: number

        const animate = (timestamp: number) => {
            if (!startTime) startTime = timestamp
            const progress = Math.min((timestamp - startTime) / duration, 1)
            setCount(Math.floor(progress * end))

            if (progress < 1) {
                animationFrame = requestAnimationFrame(animate)
            }
        }

        animationFrame = requestAnimationFrame(animate)
        return () => cancelAnimationFrame(animationFrame)
    }, [end, duration])

    return count
}

// Sample data
const emergencyData = [
    { hour: "6AM", patients: 12, predicted: 14 },
    { hour: "8AM", patients: 28, predicted: 25 },
    { hour: "10AM", patients: 35, predicted: 32 },
    { hour: "12PM", patients: 42, predicted: 45 },
    { hour: "2PM", patients: 38, predicted: 40 },
    { hour: "4PM", patients: 45, predicted: 42 },
    { hour: "6PM", patients: 52, predicted: 55 },
    { hour: "8PM", patients: 48, predicted: 50 },
    { hour: "10PM", patients: 32, predicted: 35 },
]

const icuData = [
    { day: "Mon", occupied: 18, available: 7 },
    { day: "Tue", occupied: 20, available: 5 },
    { day: "Wed", occupied: 22, available: 3 },
    { day: "Thu", occupied: 19, available: 6 },
    { day: "Fri", occupied: 21, available: 4 },
    { day: "Sat", occupied: 17, available: 8 },
    { day: "Sun", occupied: 15, available: 10 },
]

const staffData = [
    { shift: "Morning", nurses: 45, doctors: 12, support: 28 },
    { shift: "Afternoon", nurses: 42, doctors: 10, support: 25 },
    { shift: "Night", nurses: 30, doctors: 6, support: 18 },
]

interface KPICardProps {
    title: string
    value: number
    unit?: string
    change: number
    icon: React.ElementType
    color: string
    delay?: number
}

function KPICard({ title, value, unit = "", change, icon: Icon, color, delay = 0 }: KPICardProps) {
    const animatedValue = useAnimatedCounter(value, 1500)
    const isPositive = change >= 0

    return (
        <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay, duration: 0.4 }}
        >
            <Card className="card-hover">
                <CardContent className="p-6">
                    <div className="flex items-start justify-between">
                        <div>
                            <p className="text-sm font-medium text-muted-foreground">{title}</p>
                            <div className="flex items-baseline gap-1 mt-2">
                                <span className="text-3xl font-bold">{animatedValue}</span>
                                {unit && <span className="text-muted-foreground">{unit}</span>}
                            </div>
                            <div className={`flex items-center gap-1 mt-2 text-sm ${isPositive ? "text-emerald-600" : "text-red-600"
                                }`}>
                                {isPositive ? (
                                    <ArrowUpRight className="h-4 w-4" />
                                ) : (
                                    <ArrowDownRight className="h-4 w-4" />
                                )}
                                <span>{Math.abs(change)}% from yesterday</span>
                            </div>
                        </div>
                        <div className={`p-3 rounded-xl ${color}`}>
                            <Icon className="h-6 w-6 text-white" />
                        </div>
                    </div>
                </CardContent>
            </Card>
        </motion.div>
    )
}

export function Dashboard() {
    const [isLoading, setIsLoading] = useState(true)

    useEffect(() => {
        const timer = setTimeout(() => setIsLoading(false), 1000)
        return () => clearTimeout(timer)
    }, [])

    if (isLoading) {
        return (
            <div className="space-y-6">
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                    {[...Array(4)].map((_, i) => (
                        <Card key={i}>
                            <CardContent className="p-6">
                                <Skeleton className="h-4 w-24 mb-4" />
                                <Skeleton className="h-8 w-16 mb-2" />
                                <Skeleton className="h-4 w-32" />
                            </CardContent>
                        </Card>
                    ))}
                </div>
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                    <Card>
                        <CardContent className="p-6">
                            <Skeleton className="h-[300px]" />
                        </CardContent>
                    </Card>
                    <Card>
                        <CardContent className="p-6">
                            <Skeleton className="h-[300px]" />
                        </CardContent>
                    </Card>
                </div>
            </div>
        )
    }

    return (
        <div className="space-y-6">
            {/* Page Header */}
            <div>
                <h1 className="text-2xl font-bold">Dashboard</h1>
                <p className="text-muted-foreground">Hospital operations overview</p>
            </div>

            {/* KPI Cards */}
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                <KPICard
                    title="Emergency Load"
                    value={48}
                    unit="patients"
                    change={12}
                    icon={Activity}
                    color="bg-red-500"
                    delay={0}
                />
                <KPICard
                    title="ICU Occupancy"
                    value={88}
                    unit="%"
                    change={-5}
                    icon={Bed}
                    color="bg-amber-500"
                    delay={0.1}
                />
                <KPICard
                    title="Staff On Duty"
                    value={142}
                    change={8}
                    icon={Users}
                    color="bg-emerald-500"
                    delay={0.2}
                />
                <KPICard
                    title="Risk Level"
                    value={3}
                    unit="/ 5"
                    change={0}
                    icon={AlertTriangle}
                    color="bg-primary"
                    delay={0.3}
                />
            </div>

            {/* Charts Row 1 */}
            <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                {/* Emergency Admissions Trend */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.4 }}
                >
                    <Card>
                        <CardHeader>
                            <CardTitle className="flex items-center gap-2">
                                <TrendingUp className="h-5 w-5 text-primary" />
                                Emergency Admissions
                            </CardTitle>
                        </CardHeader>
                        <CardContent>
                            <ResponsiveContainer width="100%" height={300}>
                                <AreaChart data={emergencyData}>
                                    <defs>
                                        <linearGradient id="colorPatients" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="hsl(221, 83%, 53%)" stopOpacity={0.3} />
                                            <stop offset="95%" stopColor="hsl(221, 83%, 53%)" stopOpacity={0} />
                                        </linearGradient>
                                        <linearGradient id="colorPredicted" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="hsl(174, 72%, 48%)" stopOpacity={0.3} />
                                            <stop offset="95%" stopColor="hsl(174, 72%, 48%)" stopOpacity={0} />
                                        </linearGradient>
                                    </defs>
                                    <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                                    <XAxis dataKey="hour" className="text-xs" />
                                    <YAxis className="text-xs" />
                                    <Tooltip
                                        contentStyle={{
                                            backgroundColor: "hsl(var(--card))",
                                            border: "1px solid hsl(var(--border))",
                                            borderRadius: "8px",
                                        }}
                                    />
                                    <Legend />
                                    <Area
                                        type="monotone"
                                        dataKey="patients"
                                        stroke="hsl(221, 83%, 53%)"
                                        fillOpacity={1}
                                        fill="url(#colorPatients)"
                                        name="Actual"
                                    />
                                    <Area
                                        type="monotone"
                                        dataKey="predicted"
                                        stroke="hsl(174, 72%, 48%)"
                                        fillOpacity={1}
                                        fill="url(#colorPredicted)"
                                        name="Predicted"
                                    />
                                </AreaChart>
                            </ResponsiveContainer>
                        </CardContent>
                    </Card>
                </motion.div>

                {/* ICU Utilization */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.5 }}
                >
                    <Card>
                        <CardHeader>
                            <CardTitle className="flex items-center gap-2">
                                <Bed className="h-5 w-5 text-amber-500" />
                                ICU Utilization
                            </CardTitle>
                        </CardHeader>
                        <CardContent>
                            <ResponsiveContainer width="100%" height={300}>
                                <BarChart data={icuData}>
                                    <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                                    <XAxis dataKey="day" className="text-xs" />
                                    <YAxis className="text-xs" />
                                    <Tooltip
                                        contentStyle={{
                                            backgroundColor: "hsl(var(--card))",
                                            border: "1px solid hsl(var(--border))",
                                            borderRadius: "8px",
                                        }}
                                    />
                                    <Legend />
                                    <Bar dataKey="occupied" fill="hsl(221, 83%, 53%)" name="Occupied" radius={[4, 4, 0, 0]} />
                                    <Bar dataKey="available" fill="hsl(174, 72%, 48%)" name="Available" radius={[4, 4, 0, 0]} />
                                </BarChart>
                            </ResponsiveContainer>
                        </CardContent>
                    </Card>
                </motion.div>
            </div>

            {/* Charts Row 2 */}
            <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.6 }}
            >
                <Card>
                    <CardHeader>
                        <CardTitle className="flex items-center gap-2">
                            <Users className="h-5 w-5 text-emerald-500" />
                            Staff Distribution by Shift
                        </CardTitle>
                    </CardHeader>
                    <CardContent>
                        <ResponsiveContainer width="100%" height={300}>
                            <BarChart data={staffData} layout="vertical">
                                <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                                <XAxis type="number" className="text-xs" />
                                <YAxis dataKey="shift" type="category" className="text-xs" width={80} />
                                <Tooltip
                                    contentStyle={{
                                        backgroundColor: "hsl(var(--card))",
                                        border: "1px solid hsl(var(--border))",
                                        borderRadius: "8px",
                                    }}
                                />
                                <Legend />
                                <Bar dataKey="nurses" fill="hsl(221, 83%, 53%)" name="Nurses" radius={[0, 4, 4, 0]} />
                                <Bar dataKey="doctors" fill="hsl(174, 72%, 48%)" name="Doctors" radius={[0, 4, 4, 0]} />
                                <Bar dataKey="support" fill="hsl(262, 83%, 58%)" name="Support" radius={[0, 4, 4, 0]} />
                            </BarChart>
                        </ResponsiveContainer>
                    </CardContent>
                </Card>
            </motion.div>
        </div>
    )
}
