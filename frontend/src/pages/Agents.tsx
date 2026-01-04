import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { motion, AnimatePresence } from "framer-motion"
import { useState, useRef } from "react"
import { agentAPI } from "@/lib/api"
import {
    Brain,
    AlertTriangle,
    Upload,
    FileText,
    Loader2,
    MapPin,
    Thermometer,
    CloudRain,
    Sun,
    Snowflake,
    Leaf,
    Wind,
    Bed,
    Users,
    CheckCircle,
    X,
} from "lucide-react"

const seasons = [
    { value: "Summer", icon: Sun, color: "text-amber-500" },
    { value: "Monsoon", icon: CloudRain, color: "text-blue-500" },
    { value: "Winter", icon: Snowflake, color: "text-cyan-500" },
    { value: "Spring", icon: Leaf, color: "text-emerald-500" },
    { value: "Autumn", icon: Wind, color: "text-orange-500" },
]

interface FileUploadProps {
    onFileSelect: (file: File) => void
    file: File | null
    onClear: () => void
}

function FileUpload({ onFileSelect, file, onClear }: FileUploadProps) {
    const inputRef = useRef<HTMLInputElement>(null)
    const [isDragging, setIsDragging] = useState(false)

    const handleDrag = (e: React.DragEvent) => {
        e.preventDefault()
        e.stopPropagation()
    }

    const handleDragIn = (e: React.DragEvent) => {
        e.preventDefault()
        e.stopPropagation()
        setIsDragging(true)
    }

    const handleDragOut = (e: React.DragEvent) => {
        e.preventDefault()
        e.stopPropagation()
        setIsDragging(false)
    }

    const handleDrop = (e: React.DragEvent) => {
        e.preventDefault()
        e.stopPropagation()
        setIsDragging(false)

        if (e.dataTransfer.files && e.dataTransfer.files[0]) {
            onFileSelect(e.dataTransfer.files[0])
        }
    }

    return (
        <div
            className={`relative border-2 border-dashed rounded-xl p-8 transition-all ${isDragging
                ? "border-primary bg-primary/5"
                : file
                    ? "border-emerald-500 bg-emerald-500/5"
                    : "border-muted-foreground/25 hover:border-muted-foreground/50"
                }`}
            onDragEnter={handleDragIn}
            onDragLeave={handleDragOut}
            onDragOver={handleDrag}
            onDrop={handleDrop}
        >
            <input
                ref={inputRef}
                type="file"
                accept=".csv,.xlsx,.xls"
                className="hidden"
                onChange={(e) => e.target.files?.[0] && onFileSelect(e.target.files[0])}
            />

            {file ? (
                <div className="flex items-center justify-between">
                    <div className="flex items-center gap-3">
                        <div className="p-2 rounded-lg bg-emerald-500/10">
                            <CheckCircle className="h-5 w-5 text-emerald-500" />
                        </div>
                        <div>
                            <p className="font-medium">{file.name}</p>
                            <p className="text-sm text-muted-foreground">
                                {(file.size / 1024).toFixed(1)} KB
                            </p>
                        </div>
                    </div>
                    <Button variant="ghost" size="icon" onClick={onClear}>
                        <X className="h-4 w-4" />
                    </Button>
                </div>
            ) : (
                <div className="text-center">
                    <Upload className="h-10 w-10 text-muted-foreground mx-auto mb-4" />
                    <p className="font-medium mb-1">Drop your file here</p>
                    <p className="text-sm text-muted-foreground mb-4">
                        CSV or Excel files supported
                    </p>
                    <Button variant="outline" onClick={() => inputRef.current?.click()}>
                        Browse Files
                    </Button>
                </div>
            )}
        </div>
    )
}

// Custom component import
import { StreamingOutput } from "@/components/ui/streaming-output"

interface AgentOutputProps {
    output: string
    isLoading: boolean
    generatedBy?: string
}

function AgentOutput({ output, isLoading, generatedBy }: AgentOutputProps) {
    if (isLoading) {
        return (
            <div className="flex items-center justify-center py-12">
                <div className="text-center">
                    <Loader2 className="h-8 w-8 animate-spin text-primary mx-auto mb-4" />
                    <p className="text-muted-foreground">Analyzing with AI...</p>
                </div>
            </div>
        )
    }

    if (!output) {
        return (
            <div className="text-center py-12 text-muted-foreground">
                <FileText className="h-12 w-12 mx-auto mb-4 opacity-50" />
                <p>Run an analysis to see results here</p>
            </div>
        )
    }

    return (
        <div className="space-y-4">
            {generatedBy && (
                <div className="flex items-center gap-2 text-sm">
                    <Brain className="h-4 w-4 text-primary" />
                    <span className="text-muted-foreground">Generated by:</span>
                    <span className="font-medium text-primary">{generatedBy}</span>
                </div>
            )}
            <div className="bg-muted/50 p-6 rounded-lg overflow-auto max-h-[500px] border border-border/50 shadow-inner">
                <StreamingOutput content={output} speed={10} />
            </div>
        </div>
    )
}

export function Agents() {
    // Context Agent State
    const [location, setLocation] = useState("")
    const [season, setSeason] = useState("Winter")
    const [contextOutput, setContextOutput] = useState("")
    const [contextLoading, setContextLoading] = useState(false)
    const [contextGeneratedBy, setContextGeneratedBy] = useState("")

    // Emergency Agent State
    const [emergencyFile, setEmergencyFile] = useState<File | null>(null)
    const [emergencyOutput, setEmergencyOutput] = useState("")
    const [emergencyLoading, setEmergencyLoading] = useState(false)

    // ICU Agent State
    const [icuFile, setIcuFile] = useState<File | null>(null)
    const [icuOutput, setIcuOutput] = useState("")
    const [icuLoading, setIcuLoading] = useState(false)

    // Staff Agent State
    const [staffFile, setStaffFile] = useState<File | null>(null)
    const [staffOutput, setStaffOutput] = useState("")
    const [staffLoading, setStaffLoading] = useState(false)

    const handleContextAnalysis = async () => {
        if (!location.trim()) return

        setContextLoading(true)
        setContextOutput("")

        try {
            const formData = new FormData()
            formData.append("location", location)
            formData.append("season", season)

            const response = await agentAPI.context(formData)
            setContextOutput(response.data.analysis || JSON.stringify(response.data, null, 2))
            setContextGeneratedBy(response.data.generated_by || "AI")
        } catch (error: any) {
            setContextOutput(`Error: ${error.response?.data?.detail || error.message}`)
        } finally {
            setContextLoading(false)
        }
    }

    const handleEmergencyAnalysis = async () => {
        if (!emergencyFile) return

        setEmergencyLoading(true)
        setEmergencyOutput("")

        try {
            const formData = new FormData()
            formData.append("file", emergencyFile)

            const response = await agentAPI.emergency(formData)
            setEmergencyOutput(formatAgentOutput(response.data))
        } catch (error: any) {
            setEmergencyOutput(`Error: ${error.response?.data?.detail || error.message}`)
        } finally {
            setEmergencyLoading(false)
        }
    }

    const handleIcuAnalysis = async () => {
        if (!icuFile) return

        setIcuLoading(true)
        setIcuOutput("")

        try {
            const formData = new FormData()
            formData.append("file", icuFile)

            const response = await agentAPI.icu(formData)
            setIcuOutput(formatAgentOutput(response.data))
        } catch (error: any) {
            setIcuOutput(`Error: ${error.response?.data?.detail || error.message}`)
        } finally {
            setIcuLoading(false)
        }
    }

    const handleStaffAnalysis = async () => {
        if (!staffFile) return

        setStaffLoading(true)
        setStaffOutput("")

        try {
            const formData = new FormData()
            formData.append("file", staffFile)

            const response = await agentAPI.staff(formData)
            setStaffOutput(formatAgentOutput(response.data))
        } catch (error: any) {
            setStaffOutput(`Error: ${error.response?.data?.detail || error.message}`)
        } finally {
            setStaffLoading(false)
        }
    }

    const formatAgentOutput = (data: any): string => {
        if (typeof data === "string") return data
        return JSON.stringify(data, null, 2)
    }

    const requestGeolocation = () => {
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(
                async (position) => {
                    try {
                        const response = await fetch(
                            `https://nominatim.openstreetmap.org/reverse?lat=${position.coords.latitude}&lon=${position.coords.longitude}&format=json`
                        )
                        const data = await response.json()
                        const city = data.address?.city || data.address?.town || data.address?.village || ""
                        if (city) setLocation(city)
                    } catch (error) {
                        console.error("Geocoding error:", error)
                    }
                },
                (error) => console.error("Geolocation error:", error)
            )
        }
    }

    return (
        <div className="space-y-6">
            {/* Page Header */}
            <div>
                <h1 className="text-2xl font-bold">AI Agents</h1>
                <p className="text-muted-foreground">
                    Run specialized AI analysis for hospital operations
                </p>
            </div>

            <Tabs defaultValue="context" className="space-y-6">
                <TabsList className="grid w-full grid-cols-4 lg:w-auto lg:inline-flex">
                    <TabsTrigger value="context" className="gap-2">
                        <Brain className="h-4 w-4" />
                        <span className="hidden sm:inline">Context</span>
                    </TabsTrigger>
                    <TabsTrigger value="emergency" className="gap-2">
                        <AlertTriangle className="h-4 w-4" />
                        <span className="hidden sm:inline">Emergency</span>
                    </TabsTrigger>
                    <TabsTrigger value="icu" className="gap-2">
                        <Bed className="h-4 w-4" />
                        <span className="hidden sm:inline">ICU</span>
                    </TabsTrigger>
                    <TabsTrigger value="staff" className="gap-2">
                        <Users className="h-4 w-4" />
                        <span className="hidden sm:inline">Staff</span>
                    </TabsTrigger>
                </TabsList>

                {/* Context Agent */}
                <TabsContent value="context">
                    <div className="grid lg:grid-cols-2 gap-6">
                        <Card>
                            <CardHeader>
                                <CardTitle className="flex items-center gap-2">
                                    <Brain className="h-5 w-5 text-primary" />
                                    Context Analyst
                                </CardTitle>
                                <CardDescription>
                                    AI-powered health analysis based on location and season
                                </CardDescription>
                            </CardHeader>
                            <CardContent className="space-y-4">
                                <div className="space-y-2">
                                    <label className="text-sm font-medium">Location</label>
                                    <div className="flex gap-2">
                                        <Input
                                            placeholder="Enter city name..."
                                            value={location}
                                            onChange={(e) => setLocation(e.target.value)}
                                        />
                                        <Button
                                            variant="outline"
                                            size="icon"
                                            onClick={requestGeolocation}
                                            title="Use my location"
                                        >
                                            <MapPin className="h-4 w-4" />
                                        </Button>
                                    </div>
                                </div>

                                <div className="space-y-2">
                                    <label className="text-sm font-medium">Season</label>
                                    <div className="flex flex-wrap gap-2">
                                        {seasons.map((s) => (
                                            <Button
                                                key={s.value}
                                                variant={season === s.value ? "default" : "outline"}
                                                size="sm"
                                                onClick={() => setSeason(s.value)}
                                                className="gap-2"
                                            >
                                                <s.icon className={`h-4 w-4 ${season === s.value ? "" : s.color}`} />
                                                {s.value}
                                            </Button>
                                        ))}
                                    </div>
                                </div>

                                <Button
                                    className="w-full"
                                    variant="medical"
                                    onClick={handleContextAnalysis}
                                    disabled={!location.trim() || contextLoading}
                                >
                                    {contextLoading ? (
                                        <>
                                            <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                                            Analyzing...
                                        </>
                                    ) : (
                                        <>
                                            <Brain className="h-4 w-4 mr-2" />
                                            Run Analysis
                                        </>
                                    )}
                                </Button>
                            </CardContent>
                        </Card>

                        <Card>
                            <CardHeader>
                                <CardTitle>Analysis Results</CardTitle>
                            </CardHeader>
                            <CardContent>
                                <AgentOutput
                                    output={contextOutput}
                                    isLoading={contextLoading}
                                    generatedBy={contextGeneratedBy}
                                />
                            </CardContent>
                        </Card>
                    </div>
                </TabsContent>

                {/* Emergency Agent */}
                <TabsContent value="emergency">
                    <div className="grid lg:grid-cols-2 gap-6">
                        <Card>
                            <CardHeader>
                                <CardTitle className="flex items-center gap-2">
                                    <AlertTriangle className="h-5 w-5 text-red-500" />
                                    Emergency Predictor
                                </CardTitle>
                                <CardDescription>
                                    Predict emergency load and identify peak hours
                                </CardDescription>
                            </CardHeader>
                            <CardContent className="space-y-4">
                                <FileUpload
                                    file={emergencyFile}
                                    onFileSelect={setEmergencyFile}
                                    onClear={() => setEmergencyFile(null)}
                                />
                                <Button
                                    className="w-full"
                                    variant="medical"
                                    onClick={handleEmergencyAnalysis}
                                    disabled={!emergencyFile || emergencyLoading}
                                >
                                    {emergencyLoading ? (
                                        <>
                                            <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                                            Analyzing...
                                        </>
                                    ) : (
                                        <>
                                            <AlertTriangle className="h-4 w-4 mr-2" />
                                            Predict Emergency Load
                                        </>
                                    )}
                                </Button>
                            </CardContent>
                        </Card>

                        <Card>
                            <CardHeader>
                                <CardTitle>Prediction Results</CardTitle>
                            </CardHeader>
                            <CardContent>
                                <AgentOutput output={emergencyOutput} isLoading={emergencyLoading} />
                            </CardContent>
                        </Card>
                    </div>
                </TabsContent>

                {/* ICU Agent */}
                <TabsContent value="icu">
                    <div className="grid lg:grid-cols-2 gap-6">
                        <Card>
                            <CardHeader>
                                <CardTitle className="flex items-center gap-2">
                                    <Bed className="h-5 w-5 text-amber-500" />
                                    ICU Optimizer
                                </CardTitle>
                                <CardDescription>
                                    Analyze ICU capacity and resource allocation
                                </CardDescription>
                            </CardHeader>
                            <CardContent className="space-y-4">
                                <FileUpload
                                    file={icuFile}
                                    onFileSelect={setIcuFile}
                                    onClear={() => setIcuFile(null)}
                                />
                                <Button
                                    className="w-full"
                                    variant="medical"
                                    onClick={handleIcuAnalysis}
                                    disabled={!icuFile || icuLoading}
                                >
                                    {icuLoading ? (
                                        <>
                                            <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                                            Analyzing...
                                        </>
                                    ) : (
                                        <>
                                            <Bed className="h-4 w-4 mr-2" />
                                            Analyze ICU Data
                                        </>
                                    )}
                                </Button>
                            </CardContent>
                        </Card>

                        <Card>
                            <CardHeader>
                                <CardTitle>ICU Analysis Results</CardTitle>
                            </CardHeader>
                            <CardContent>
                                <AgentOutput output={icuOutput} isLoading={icuLoading} />
                            </CardContent>
                        </Card>
                    </div>
                </TabsContent>

                {/* Staff Agent */}
                <TabsContent value="staff">
                    <div className="grid lg:grid-cols-2 gap-6">
                        <Card>
                            <CardHeader>
                                <CardTitle className="flex items-center gap-2">
                                    <Users className="h-5 w-5 text-emerald-500" />
                                    Staff Manager
                                </CardTitle>
                                <CardDescription>
                                    Analyze staff workload and identify gaps
                                </CardDescription>
                            </CardHeader>
                            <CardContent className="space-y-4">
                                <FileUpload
                                    file={staffFile}
                                    onFileSelect={setStaffFile}
                                    onClear={() => setStaffFile(null)}
                                />
                                <Button
                                    className="w-full"
                                    variant="medical"
                                    onClick={handleStaffAnalysis}
                                    disabled={!staffFile || staffLoading}
                                >
                                    {staffLoading ? (
                                        <>
                                            <Loader2 className="h-4 w-4 mr-2 animate-spin" />
                                            Analyzing...
                                        </>
                                    ) : (
                                        <>
                                            <Users className="h-4 w-4 mr-2" />
                                            Analyze Staff Data
                                        </>
                                    )}
                                </Button>
                            </CardContent>
                        </Card>

                        <Card>
                            <CardHeader>
                                <CardTitle>Staff Analysis Results</CardTitle>
                            </CardHeader>
                            <CardContent>
                                <AgentOutput output={staffOutput} isLoading={staffLoading} />
                            </CardContent>
                        </Card>
                    </div>
                </TabsContent>
            </Tabs>
        </div>
    )
}
