import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { motion } from "framer-motion"
import { useState } from "react"
import {
    LineChart,
    Line,
    AreaChart,
    Area,
    BarChart,
    Bar,
    PieChart,
    Pie,
    Cell,
    XAxis,
    YAxis,
    CartesianGrid,
    Tooltip,
    ResponsiveContainer,
    Legend,
} from "recharts"
import {
    Calendar,
    Filter,
    Download,
    TrendingUp,
    Users,
    Activity,
    Bed,
} from "lucide-react"

// Sample data
const monthlyTrends = [
    { month: "Jan", emergency: 320, icu: 145, outpatient: 890 },
    { month: "Feb", emergency: 280, icu: 120, outpatient: 850 },
    { month: "Mar", emergency: 350, icu: 160, outpatient: 920 },
    { month: "Apr", emergency: 420, icu: 180, outpatient: 1050 },
    { month: "May", emergency: 380, icu: 155, outpatient: 980 },
    { month: "Jun", emergency: 450, icu: 190, outpatient: 1100 },
    { month: "Jul", emergency: 520, icu: 210, outpatient: 1200 },
    { month: "Aug", emergency: 480, icu: 195, outpatient: 1150 },
    { month: "Sep", emergency: 420, icu: 175, outpatient: 1080 },
    { month: "Oct", emergency: 380, icu: 160, outpatient: 950 },
    { month: "Nov", emergency: 350, icu: 140, outpatient: 880 },
    { month: "Dec", emergency: 400, icu: 165, outpatient: 920 },
]

const departmentData = [
    { name: "Cardiology", patients: 420, color: "#3b82f6" },
    { name: "Neurology", patients: 280, color: "#14b8a6" },
    { name: "Orthopedics", patients: 350, color: "#f59e0b" },
    { name: "Pediatrics", patients: 290, color: "#10b981" },
    { name: "General", patients: 580, color: "#8b5cf6" },
]

const staffEfficiency = [
    { department: "Emergency", efficiency: 92 },
    { department: "ICU", efficiency: 88 },
    { department: "Surgery", efficiency: 95 },
    { department: "Outpatient", efficiency: 78 },
    { department: "Radiology", efficiency: 85 },
]

const dailyPatientFlow = Array.from({ length: 24 }, (_, i) => ({
    hour: `${i.toString().padStart(2, "0")}:00`,
    admissions: Math.floor(Math.random() * 20) + 5,
    discharges: Math.floor(Math.random() * 15) + 3,
}))

const COLORS = ["#3b82f6", "#14b8a6", "#f59e0b", "#10b981", "#8b5cf6"]

export function Analytics() {
    const [dateRange, setDateRange] = useState("This Month")
    const [department, setDepartment] = useState("All Departments")

    return (
        <div className="space-y-6">
            {/* Page Header */}
            <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
                <div>
                    <h1 className="text-2xl font-bold">Analytics</h1>
                    <p className="text-muted-foreground">
                        Comprehensive hospital performance metrics
                    </p>
                </div>

                <div className="flex items-center gap-2">
                    <Button variant="outline" size="sm" className="gap-2">
                        <Calendar className="h-4 w-4" />
                        {dateRange}
                    </Button>
                    <Button variant="outline" size="sm" className="gap-2">
                        <Filter className="h-4 w-4" />
                        {department}
                    </Button>
                    <Button variant="outline" size="sm" className="gap-2">
                        <Download className="h-4 w-4" />
                        Export
                    </Button>
                </div>
            </div>

            {/* Summary Cards */}
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                {[
                    { title: "Total Patients", value: "12,847", change: "+12%", icon: Users, color: "text-blue-500" },
                    { title: "Emergency Cases", value: "4,920", change: "+8%", icon: Activity, color: "text-red-500" },
                    { title: "ICU Admissions", value: "1,895", change: "-3%", icon: Bed, color: "text-amber-500" },
                    { title: "Avg. Stay", value: "4.2 days", change: "-5%", icon: TrendingUp, color: "text-emerald-500" },
                ].map((stat, i) => (
                    <motion.div
                        key={stat.title}
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ delay: i * 0.1 }}
                    >
                        <Card>
                            <CardContent className="p-4">
                                <div className="flex items-center gap-4">
                                    <div className={`p-2 rounded-lg bg-muted ${stat.color}`}>
                                        <stat.icon className="h-5 w-5" />
                                    </div>
                                    <div>
                                        <p className="text-sm text-muted-foreground">{stat.title}</p>
                                        <div className="flex items-baseline gap-2">
                                            <span className="text-xl font-bold">{stat.value}</span>
                                            <span className={`text-xs ${stat.change.startsWith("+") ? "text-emerald-500" : "text-red-500"
                                                }`}>
                                                {stat.change}
                                            </span>
                                        </div>
                                    </div>
                                </div>
                            </CardContent>
                        </Card>
                    </motion.div>
                ))}
            </div>

            {/* Charts Grid */}
            <div className="grid lg:grid-cols-2 gap-6">
                {/* Monthly Trends */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.2 }}
                >
                    <Card>
                        <CardHeader>
                            <CardTitle>Monthly Patient Trends</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <ResponsiveContainer width="100%" height={300}>
                                <AreaChart data={monthlyTrends}>
                                    <defs>
                                        <linearGradient id="emergency" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#ef4444" stopOpacity={0.3} />
                                            <stop offset="95%" stopColor="#ef4444" stopOpacity={0} />
                                        </linearGradient>
                                        <linearGradient id="icu" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#f59e0b" stopOpacity={0.3} />
                                            <stop offset="95%" stopColor="#f59e0b" stopOpacity={0} />
                                        </linearGradient>
                                        <linearGradient id="outpatient" x1="0" y1="0" x2="0" y2="1">
                                            <stop offset="5%" stopColor="#10b981" stopOpacity={0.3} />
                                            <stop offset="95%" stopColor="#10b981" stopOpacity={0} />
                                        </linearGradient>
                                    </defs>
                                    <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                                    <XAxis dataKey="month" className="text-xs" />
                                    <YAxis className="text-xs" />
                                    <Tooltip
                                        contentStyle={{
                                            backgroundColor: "hsl(var(--card))",
                                            border: "1px solid hsl(var(--border))",
                                            borderRadius: "8px",
                                        }}
                                    />
                                    <Legend />
                                    <Area type="monotone" dataKey="emergency" stroke="#ef4444" fill="url(#emergency)" name="Emergency" />
                                    <Area type="monotone" dataKey="icu" stroke="#f59e0b" fill="url(#icu)" name="ICU" />
                                    <Area type="monotone" dataKey="outpatient" stroke="#10b981" fill="url(#outpatient)" name="Outpatient" />
                                </AreaChart>
                            </ResponsiveContainer>
                        </CardContent>
                    </Card>
                </motion.div>

                {/* Department Distribution */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.3 }}
                >
                    <Card>
                        <CardHeader>
                            <CardTitle>Department Distribution</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <ResponsiveContainer width="100%" height={300}>
                                <PieChart>
                                    <Pie
                                        data={departmentData}
                                        cx="50%"
                                        cy="50%"
                                        innerRadius={60}
                                        outerRadius={100}
                                        paddingAngle={5}
                                        dataKey="patients"
                                    >
                                        {departmentData.map((entry, index) => (
                                            <Cell key={`cell-${index}`} fill={entry.color} />
                                        ))}
                                    </Pie>
                                    <Tooltip
                                        contentStyle={{
                                            backgroundColor: "hsl(var(--card))",
                                            border: "1px solid hsl(var(--border))",
                                            borderRadius: "8px",
                                        }}
                                    />
                                    <Legend />
                                </PieChart>
                            </ResponsiveContainer>
                        </CardContent>
                    </Card>
                </motion.div>

                {/* Staff Efficiency */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.4 }}
                >
                    <Card>
                        <CardHeader>
                            <CardTitle>Staff Efficiency by Department</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <ResponsiveContainer width="100%" height={300}>
                                <BarChart data={staffEfficiency} layout="vertical">
                                    <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                                    <XAxis type="number" domain={[0, 100]} className="text-xs" />
                                    <YAxis dataKey="department" type="category" width={80} className="text-xs" />
                                    <Tooltip
                                        contentStyle={{
                                            backgroundColor: "hsl(var(--card))",
                                            border: "1px solid hsl(var(--border))",
                                            borderRadius: "8px",
                                        }}
                                        formatter={(value) => [`${value}%`, "Efficiency"]}
                                    />
                                    <Bar
                                        dataKey="efficiency"
                                        fill="hsl(221, 83%, 53%)"
                                        radius={[0, 4, 4, 0]}
                                    />
                                </BarChart>
                            </ResponsiveContainer>
                        </CardContent>
                    </Card>
                </motion.div>

                {/* Daily Patient Flow */}
                <motion.div
                    initial={{ opacity: 0, y: 20 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ delay: 0.5 }}
                >
                    <Card>
                        <CardHeader>
                            <CardTitle>24h Patient Flow</CardTitle>
                        </CardHeader>
                        <CardContent>
                            <ResponsiveContainer width="100%" height={300}>
                                <LineChart data={dailyPatientFlow}>
                                    <CartesianGrid strokeDasharray="3 3" className="stroke-muted" />
                                    <XAxis dataKey="hour" className="text-xs" interval={3} />
                                    <YAxis className="text-xs" />
                                    <Tooltip
                                        contentStyle={{
                                            backgroundColor: "hsl(var(--card))",
                                            border: "1px solid hsl(var(--border))",
                                            borderRadius: "8px",
                                        }}
                                    />
                                    <Legend />
                                    <Line
                                        type="monotone"
                                        dataKey="admissions"
                                        stroke="#3b82f6"
                                        strokeWidth={2}
                                        dot={false}
                                        name="Admissions"
                                    />
                                    <Line
                                        type="monotone"
                                        dataKey="discharges"
                                        stroke="#10b981"
                                        strokeWidth={2}
                                        dot={false}
                                        name="Discharges"
                                    />
                                </LineChart>
                            </ResponsiveContainer>
                        </CardContent>
                    </Card>
                </motion.div>
            </div>
        </div>
    )
}
