import { createContext, useContext, useState, useEffect, ReactNode } from "react"
import { authAPI, User } from "@/lib/api"

interface AuthContextType {
    user: User | null
    token: string | null
    isLoading: boolean
    login: (email: string, password: string) => Promise<void>
    register: (email: string, name: string, password: string) => Promise<void>
    logout: () => void
    setToken: (token: string) => void
}

const AuthContext = createContext<AuthContextType | null>(null)

export function AuthProvider({ children }: { children: ReactNode }) {
    const [user, setUser] = useState<User | null>(null)
    const [token, setTokenState] = useState<string | null>(
        localStorage.getItem("token")
    )
    const [isLoading, setIsLoading] = useState(true)

    useEffect(() => {
        const loadUser = async () => {
            if (token) {
                try {
                    const response = await authAPI.me()
                    setUser(response.data)
                } catch (error) {
                    localStorage.removeItem("token")
                    setTokenState(null)
                }
            }
            setIsLoading(false)
        }
        loadUser()
    }, [token])

    const setToken = (newToken: string) => {
        localStorage.setItem("token", newToken)
        setTokenState(newToken)
    }

    const login = async (email: string, password: string) => {
        const response = await authAPI.login(email, password)
        setToken(response.data.token)
        setUser(response.data.user)
    }

    const register = async (email: string, name: string, password: string) => {
        const response = await authAPI.register(email, name, password)
        setToken(response.data.token)
        setUser(response.data.user)
    }

    const logout = () => {
        localStorage.removeItem("token")
        setTokenState(null)
        setUser(null)
    }

    return (
        <AuthContext.Provider
            value={{ user, token, isLoading, login, register, logout, setToken }}
        >
            {children}
        </AuthContext.Provider>
    )
}

export function useAuth() {
    const context = useContext(AuthContext)
    if (!context) {
        throw new Error("useAuth must be used within an AuthProvider")
    }
    return context
}
