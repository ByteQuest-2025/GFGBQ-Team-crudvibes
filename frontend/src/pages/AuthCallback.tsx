import { useEffect } from "react"
import { useNavigate, useSearchParams } from "react-router-dom"
import { useAuth } from "@/context/AuthContext"
import { Loader2 } from "lucide-react"

export function AuthCallback() {
    const navigate = useNavigate()
    const [searchParams] = useSearchParams()
    const { setToken } = useAuth()

    useEffect(() => {
        const token = searchParams.get("token")
        if (token) {
            setToken(token)
            navigate("/dashboard")
        } else {
            navigate("/login")
        }
    }, [searchParams, setToken, navigate])

    return (
        <div className="min-h-screen flex items-center justify-center">
            <div className="text-center">
                <Loader2 className="h-8 w-8 animate-spin text-primary mx-auto mb-4" />
                <p className="text-muted-foreground">Completing sign in...</p>
            </div>
        </div>
    )
}
