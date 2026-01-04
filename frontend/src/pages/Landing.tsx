import { Button } from "@/components/ui/button"
import { Logo } from "@/components/Logo"
import { Link } from "react-router-dom"
import { motion } from "framer-motion"
import {
    Activity,
    Brain,
    Shield,
    Zap,
    ArrowRight,
    BarChart3,
    Bot,
    Clock
} from "lucide-react"

const features = [
    {
        icon: Brain,
        title: "AI-Powered Analysis",
        description: "Llama 3.3 70B analyzes health patterns specific to your region and season."
    },
    {
        icon: Activity,
        title: "Emergency Prediction",
        description: "Forecast patient load, identify peak hours, and optimize resource allocation."
    },
    {
        icon: Shield,
        title: "Risk Assessment",
        description: "Endemic disease tracking, outbreak alerts, and hospital preparedness protocols."
    },
    {
        icon: Zap,
        title: "Real-time Insights",
        description: "Live dashboards with ICU occupancy, staff workload, and critical alerts."
    }
]

const stats = [
    { value: "99.9%", label: "Uptime" },
    { value: "50ms", label: "Response Time" },
    { value: "4", label: "AI Agents" },
    { value: "24/7", label: "Monitoring" }
]

export function Landing() {
    return (
        <div className="min-h-screen bg-background relative overflow-hidden">
            {/* Animated mesh gradient background */}
            <div className="absolute inset-0 gradient-mesh" />

            {/* Grid pattern overlay */}
            <div
                className="absolute inset-0 opacity-[0.02] dark:opacity-[0.05]"
                style={{
                    backgroundImage: `url("data:image/svg+xml,%3Csvg width='60' height='60' viewBox='0 0 60 60' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' fill-rule='evenodd'%3E%3Cg fill='%23000' fill-opacity='1'%3E%3Cpath d='M36 34v-4h-2v4h-4v2h4v4h2v-4h4v-2h-4zm0-30V0h-2v4h-4v2h4v4h2V6h4V4h-4zM6 34v-4H4v4H0v2h4v4h2v-4h4v-2H6zM6 4V0H4v4H0v2h4v4h2V6h4V4H6z'/%3E%3C/g%3E%3C/g%3E%3C/svg%3E")`
                }}
            />

            {/* Navigation */}
            <nav className="relative z-10 flex items-center justify-between px-6 py-4 lg:px-12">
                <Logo size="lg" />

                <div className="flex items-center gap-4">
                    <Link to="/login">
                        <Button variant="ghost">Sign In</Button>
                    </Link>
                    <Link to="/register">
                        <Button variant="medical">Get Started</Button>
                    </Link>
                </div>
            </nav>

            {/* Hero Section */}
            <section className="relative z-10 max-w-7xl mx-auto px-6 py-20 lg:py-32">
                <div className="text-center max-w-4xl mx-auto">
                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.6 }}
                    >
                        <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary/10 text-primary text-sm font-medium mb-6">
                            <Bot className="h-4 w-4" />
                            Powered by Llama 3.3 70B
                        </div>
                    </motion.div>

                    <motion.h1
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.6, delay: 0.1 }}
                        className="text-4xl sm:text-5xl lg:text-7xl font-bold tracking-tight mb-6"
                    >
                        <span className="text-foreground">Hospital Intelligence</span>
                        <br />
                        <span className="bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
                            Reimagined
                        </span>
                    </motion.h1>

                    <motion.p
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.6, delay: 0.2 }}
                        className="text-xl text-muted-foreground mb-8 max-w-2xl mx-auto"
                    >
                        Predict. Prepare. Protect.
                        <br />
                        AI-powered hospital command center for emergency prediction,
                        resource optimization, and outbreak prevention.
                    </motion.p>

                    <motion.div
                        initial={{ opacity: 0, y: 20 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.6, delay: 0.3 }}
                        className="flex flex-col sm:flex-row items-center justify-center gap-4"
                    >
                        <Link to="/dashboard">
                            <Button size="xl" variant="medical" className="group">
                                View Dashboard
                                <ArrowRight className="ml-2 h-4 w-4 transition-transform group-hover:translate-x-1" />
                            </Button>
                        </Link>
                        <Link to="/agents">
                            <Button size="xl" variant="outline">
                                Run AI Agents
                            </Button>
                        </Link>
                    </motion.div>
                </div>

                {/* Animated pulse line */}
                <motion.div
                    initial={{ opacity: 0 }}
                    animate={{ opacity: 1 }}
                    transition={{ delay: 0.5 }}
                    className="mt-16 flex justify-center"
                >
                    <div className="w-64 h-0.5 bg-gradient-to-r from-transparent via-primary to-transparent animate-pulse-line" />
                </motion.div>
            </section>

            {/* Stats */}
            <section className="relative z-10 max-w-7xl mx-auto px-6 py-12">
                <motion.div
                    initial={{ opacity: 0, y: 40 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.6, delay: 0.4 }}
                    className="grid grid-cols-2 md:grid-cols-4 gap-8"
                >
                    {stats.map((stat, i) => (
                        <motion.div
                            key={stat.label}
                            initial={{ opacity: 0, y: 20 }}
                            animate={{ opacity: 1, y: 0 }}
                            transition={{ delay: 0.5 + i * 0.1 }}
                            className="text-center"
                        >
                            <div className="text-3xl lg:text-4xl font-bold bg-gradient-to-r from-primary to-accent bg-clip-text text-transparent">
                                {stat.value}
                            </div>
                            <div className="text-sm text-muted-foreground mt-1">{stat.label}</div>
                        </motion.div>
                    ))}
                </motion.div>
            </section>

            {/* Features */}
            <section className="relative z-10 max-w-7xl mx-auto px-6 py-20">
                <motion.div
                    initial={{ opacity: 0 }}
                    whileInView={{ opacity: 1 }}
                    viewport={{ once: true }}
                    className="text-center mb-16"
                >
                    <h2 className="text-3xl font-bold mb-4">Built for Healthcare Excellence</h2>
                    <p className="text-muted-foreground max-w-2xl mx-auto">
                        Four specialized AI agents working together to transform hospital operations
                    </p>
                </motion.div>

                <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
                    {features.map((feature, i) => (
                        <motion.div
                            key={feature.title}
                            initial={{ opacity: 0, y: 20 }}
                            whileInView={{ opacity: 1, y: 0 }}
                            viewport={{ once: true }}
                            transition={{ delay: i * 0.1 }}
                            className="group p-6 rounded-2xl border bg-card/50 backdrop-blur-sm hover:bg-card hover:shadow-lg transition-all duration-300"
                        >
                            <div className="h-12 w-12 rounded-xl bg-primary/10 flex items-center justify-center mb-4 group-hover:bg-primary/20 transition-colors">
                                <feature.icon className="h-6 w-6 text-primary" />
                            </div>
                            <h3 className="font-semibold mb-2">{feature.title}</h3>
                            <p className="text-sm text-muted-foreground">{feature.description}</p>
                        </motion.div>
                    ))}
                </div>
            </section>

            {/* CTA */}
            <section className="relative z-10 max-w-7xl mx-auto px-6 py-20">
                <motion.div
                    initial={{ opacity: 0, scale: 0.95 }}
                    whileInView={{ opacity: 1, scale: 1 }}
                    viewport={{ once: true }}
                    className="rounded-3xl bg-gradient-to-r from-primary to-accent p-12 text-center relative overflow-hidden"
                >
                    <div className="absolute inset-0 opacity-10">
                        <div className="absolute inset-0" style={{
                            backgroundImage: 'radial-gradient(circle at 2px 2px, white 1px, transparent 0)',
                            backgroundSize: '32px 32px'
                        }} />
                    </div>

                    <div className="relative z-10">
                        <Clock className="h-12 w-12 text-white/80 mx-auto mb-4" />
                        <h2 className="text-3xl font-bold text-white mb-4">
                            Ready to Transform Your Hospital?
                        </h2>
                        <p className="text-white/80 mb-8 max-w-xl mx-auto">
                            Join leading hospitals using MedicOS to predict emergencies,
                            optimize resources, and save lives.
                        </p>
                        <Link to="/register">
                            <Button size="xl" variant="secondary" className="bg-white text-primary hover:bg-white/90">
                                Start Free Trial
                                <ArrowRight className="ml-2 h-4 w-4" />
                            </Button>
                        </Link>
                    </div>
                </motion.div>
            </section>

            {/* Footer */}
            <footer className="relative z-10 border-t py-8 px-6">
                <div className="max-w-7xl mx-auto flex flex-col md:flex-row items-center justify-between gap-4">
                    <Logo size="sm" />
                    <p className="text-sm text-muted-foreground">
                        © 2024 MedicOS. Hospital Intelligence Platform.
                    </p>
                </div>
            </footer>
        </div>
    )
}
