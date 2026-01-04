import axios from "axios"

const API_URL = "http://localhost:8000/api"

export const api = axios.create({
    baseURL: API_URL,
    headers: {
        "Content-Type": "application/json",
    },
})

// Add token to requests
api.interceptors.request.use((config) => {
    const token = localStorage.getItem("token")
    if (token) {
        config.headers.Authorization = `Bearer ${token}`
    }
    return config
})

// Auth endpoints
export const authAPI = {
    login: (email: string, password: string) =>
        api.post("/login", { email, password }),

    register: (email: string, name: string, password: string) =>
        api.post("/register", { email, name, password }),

    me: () => api.get("/me"),

    googleLogin: () => `${API_URL}/auth/google`,
}

// Agent endpoints
export const agentAPI = {
    context: (formData: FormData) =>
        api.post("/agents/context", formData, {
            headers: { "Content-Type": "multipart/form-data" },
        }),

    emergency: (formData: FormData) =>
        api.post("/agents/emergency", formData, {
            headers: { "Content-Type": "multipart/form-data" },
        }),

    icu: (formData: FormData) =>
        api.post("/agents/icu", formData, {
            headers: { "Content-Type": "multipart/form-data" },
        }),

    staff: (formData: FormData) =>
        api.post("/agents/staff", formData, {
            headers: { "Content-Type": "multipart/form-data" },
        }),
}

export type User = {
    id: number
    email: string
    name: string
}
