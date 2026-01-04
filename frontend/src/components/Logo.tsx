import { cn } from "@/lib/utils"

interface LogoProps {
    className?: string
    size?: "sm" | "md" | "lg"
    showText?: boolean
}

export function Logo({ className, size = "md", showText = true }: LogoProps) {
    const sizes = {
        sm: { icon: 24, text: "text-lg" },
        md: { icon: 32, text: "text-xl" },
        lg: { icon: 48, text: "text-3xl" },
    }

    const { icon, text } = sizes[size]

    return (
        <div className={cn("flex items-center gap-2", className)}>
            <svg
                width={icon}
                height={icon}
                viewBox="0 0 48 48"
                fill="none"
                xmlns="http://www.w3.org/2000/svg"
                className="flex-shrink-0"
            >
                {/* Background circle with gradient */}
                <defs>
                    <linearGradient id="logoGradient" x1="0%" y1="0%" x2="100%" y2="100%">
                        <stop offset="0%" stopColor="hsl(221, 83%, 53%)" />
                        <stop offset="100%" stopColor="hsl(174, 72%, 48%)" />
                    </linearGradient>
                    <linearGradient id="pulseGradient" x1="0%" y1="50%" x2="100%" y2="50%">
                        <stop offset="0%" stopColor="white" stopOpacity="0.9" />
                        <stop offset="50%" stopColor="white" stopOpacity="1" />
                        <stop offset="100%" stopColor="white" stopOpacity="0.9" />
                    </linearGradient>
                </defs>

                {/* Main circle */}
                <circle cx="24" cy="24" r="22" fill="url(#logoGradient)" />

                {/* Heartbeat/pulse line */}
                <path
                    d="M8 24 L14 24 L17 18 L21 30 L25 20 L28 26 L31 24 L40 24"
                    stroke="url(#pulseGradient)"
                    strokeWidth="2.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    fill="none"
                />

                {/* Cross overlay (subtle) */}
                <path
                    d="M24 12 L24 16 M24 32 L24 36 M12 24 L16 24 M32 24 L36 24"
                    stroke="white"
                    strokeWidth="1.5"
                    strokeLinecap="round"
                    strokeOpacity="0.4"
                />
            </svg>

            {showText && (
                <span className={cn("font-semibold tracking-tight", text)}>
                    <span className="text-primary">Medic</span>
                    <span className="text-accent">OS</span>
                </span>
            )}
        </div>
    )
}
