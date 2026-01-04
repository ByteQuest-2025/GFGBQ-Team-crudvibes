import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom"
import { AuthProvider, useAuth } from "@/context/AuthContext"
import { DashboardLayout } from "@/components/layout/DashboardLayout"

// Pages
import { Landing } from "@/pages/Landing"
import { Login } from "@/pages/Login"
import { Register } from "@/pages/Register"
import { AuthCallback } from "@/pages/AuthCallback"
import { Dashboard } from "@/pages/Dashboard"
import { Agents } from "@/pages/Agents"
import { Analytics } from "@/pages/Analytics"

// Protected Route Component
function ProtectedRoute({ children }: { children: React.ReactNode }) {
    const { token, isLoading } = useAuth()

    if (isLoading) {
        return (
            <div className="min-h-screen flex items-center justify-center">
                <div className="animate-spin h-8 w-8 border-4 border-primary border-t-transparent rounded-full" />
            </div>
        )
    }

    if (!token) {
        return <Navigate to="/login" replace />
    }

    return <>{children}</>
}

// App Routes
function AppRoutes() {
    const { token, logout } = useAuth()

    return (
        <Routes>
            {/* Public Routes */}
            <Route path="/" element={token ? <Navigate to="/dashboard" replace /> : <Landing />} />
            <Route path="/login" element={token ? <Navigate to="/dashboard" replace /> : <Login />} />
            <Route path="/register" element={token ? <Navigate to="/dashboard" replace /> : <Register />} />
            <Route path="/auth/callback" element={<AuthCallback />} />

            {/* Protected Routes */}
            <Route
                path="/dashboard"
                element={
                    <ProtectedRoute>
                        <DashboardLayout onLogout={logout}>
                            <Dashboard />
                        </DashboardLayout>
                    </ProtectedRoute>
                }
            />
            <Route
                path="/agents"
                element={
                    <ProtectedRoute>
                        <DashboardLayout onLogout={logout}>
                            <Agents />
                        </DashboardLayout>
                    </ProtectedRoute>
                }
            />
            <Route
                path="/analytics"
                element={
                    <ProtectedRoute>
                        <DashboardLayout onLogout={logout}>
                            <Analytics />
                        </DashboardLayout>
                    </ProtectedRoute>
                }
            />

            {/* Fallback */}
            <Route path="*" element={<Navigate to="/" replace />} />
        </Routes>
    )
}

function App() {
    return (
        <BrowserRouter>
            <AuthProvider>
                <AppRoutes />
            </AuthProvider>
        </BrowserRouter>
    )
}

export default App
