import { useState, useEffect, useCallback } from "react"
import { Sparkle, Cpu, Monitor } from "@phosphor-icons/react"
import { Button } from "@/components/ui/button"
import { Separator } from "@/components/ui/separator"
import { Badge } from "@/components/ui/badge"
import { ThemeToggle } from "@/components/ThemeToggle"
import { PromptManual } from "@/components/PromptManual"
import { PromptBuilder } from "@/components/PromptBuilder"
import { ImageDisplay } from "@/components/ImageDisplay"

interface HardwareStatus {
  profile: string
  gpu_name: string
  vram_gb: number
  resolution: number
  model_loaded: boolean
}

interface GenerationMeta {
  seed: number
  duration: string
  resolution: number
}

type Mode = "manual" | "builder"

const PROFILE_COLORS: Record<string, string> = {
  high: "text-green-600 dark:text-green-400",
  mid: "text-blue-600 dark:text-blue-400",
  low: "text-amber-600 dark:text-amber-400",
  very_low: "text-orange-600 dark:text-orange-400",
  cpu: "text-red-600 dark:text-red-400",
}

export default function App() {
  const [mode, setMode] = useState<Mode>("manual")
  const [manualPrompt, setManualPrompt] = useState("")
  const [builderPrompt, setBuilderPrompt] = useState("")
  const [generating, setGenerating] = useState(false)
  const [imageUrl, setImageUrl] = useState<string | null>(null)
  const [meta, setMeta] = useState<GenerationMeta | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [hwStatus, setHwStatus] = useState<HardwareStatus | null>(null)

  useEffect(() => {
    fetch("/api/status")
      .then((r) => (r.ok ? r.json() : null))
      .then((data: HardwareStatus | null) => data && setHwStatus(data))
      .catch(() => {})
  }, [])

  const handleBuilderPrompt = useCallback((p: string) => {
    setBuilderPrompt(p)
  }, [])

  const activePrompt = mode === "manual" ? manualPrompt : builderPrompt

  const handleGenerate = async () => {
    if (!activePrompt.trim() || generating) return
    setGenerating(true)
    setError(null)

    const startTime = Date.now()

    try {
      const res = await fetch("/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: activePrompt.trim() }),
      })

      if (!res.ok) throw new Error(await res.text())

      const blob = await res.blob()
      if (imageUrl) URL.revokeObjectURL(imageUrl)

      const seedHeader = res.headers.get("X-Seed")
      setImageUrl(URL.createObjectURL(blob))
      setMeta({
        seed: seedHeader ? parseInt(seedHeader, 10) : 0,
        duration: ((Date.now() - startTime) / 1000).toFixed(1),
        resolution: hwStatus?.resolution ?? 1024,
      })
    } catch (e) {
      setError(e instanceof Error ? e.message : "Generation failed. Please try again.")
    } finally {
      setGenerating(false)
    }
  }

  const isCpu = hwStatus?.profile === "cpu"

  return (
    <div className="flex min-h-screen flex-col bg-background">
      {/* Header */}
      <header className="sticky top-0 z-10 border-b border-border bg-background/80 backdrop-blur-sm">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-3">
          {/* Logo */}
          <div className="flex items-center gap-3">
            <div className="flex size-8 items-center justify-center rounded-lg bg-primary/10">
              <Sparkle size={16} className="text-primary" weight="fill" />
            </div>
            <div>
              <h1 className="text-sm font-semibold leading-none">Portrait Studio</h1>
              <p className="mt-0.5 text-xs text-muted-foreground">
                SDXL · LoRA CelebA · Local
              </p>
            </div>
          </div>

          {/* Hardware badge + theme toggle */}
          <div className="flex items-center gap-3">
            {hwStatus && (
              <div className="hidden items-center gap-2 sm:flex">
                {isCpu ? (
                  <Cpu size={14} className="text-muted-foreground" />
                ) : (
                  <Monitor size={14} className="text-muted-foreground" />
                )}
                <span className="text-xs text-muted-foreground">
                  {isCpu ? "CPU mode" : hwStatus.gpu_name}
                </span>
                {hwStatus.vram_gb > 0 && (
                  <Badge
                    variant="outline"
                    className={PROFILE_COLORS[hwStatus.profile] ?? ""}
                  >
                    {hwStatus.vram_gb} GB
                  </Badge>
                )}
              </div>
            )}
            <ThemeToggle />
          </div>
        </div>
      </header>

      {/* Main layout */}
      <main className="mx-auto grid w-full max-w-7xl flex-1 grid-cols-1 gap-6 p-6 md:grid-cols-[440px_1fr]">
        {/* Left panel — controls */}
        <div className="flex flex-col gap-5">
          {/* Mode switcher */}
          <div className="flex gap-1 rounded-lg border border-border bg-muted p-1">
            {(["manual", "builder"] as Mode[]).map((m) => (
              <button
                key={m}
                onClick={() => setMode(m)}
                className={`flex-1 rounded-md px-3 py-1.5 text-sm font-medium transition-all ${
                  mode === m
                    ? "bg-primary text-primary-foreground shadow-sm"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                {m === "manual" ? "Manual input" : "Prompt builder"}
              </button>
            ))}
          </div>

          {/* Active prompt mode */}
          <div className="flex-1">
            {mode === "manual" ? (
              <PromptManual prompt={manualPrompt} onChange={setManualPrompt} />
            ) : (
              <PromptBuilder onPromptChange={handleBuilderPrompt} />
            )}
          </div>

          <Separator />

          {/* Generate */}
          <Button
            size="lg"
            className="w-full"
            onClick={handleGenerate}
            disabled={generating || !activePrompt.trim()}
          >
            {generating ? (
              <>
                <span className="size-4 animate-spin rounded-full border-2 border-primary-foreground/30 border-t-primary-foreground" />
                Generating…
              </>
            ) : (
              <>
                <Sparkle size={16} weight="fill" />
                Generate Portrait
              </>
            )}
          </Button>

          {/* Hardware info card */}
          {hwStatus && (
            <div className="rounded-lg border border-border bg-muted/20 px-4 py-3">
              <div className="grid grid-cols-2 gap-y-1.5 text-xs">
                <span className="text-muted-foreground">Device</span>
                <span className="text-right font-medium">
                  {isCpu ? "CPU only" : hwStatus.gpu_name}
                </span>

                {hwStatus.vram_gb > 0 && (
                  <>
                    <span className="text-muted-foreground">VRAM</span>
                    <span className="text-right font-medium">{hwStatus.vram_gb} GB</span>
                  </>
                )}

                <span className="text-muted-foreground">Resolution</span>
                <span className="text-right font-medium">
                  {hwStatus.resolution}×{hwStatus.resolution}
                </span>

                <span className="text-muted-foreground">Profile</span>
                <span
                  className={`text-right font-medium capitalize ${PROFILE_COLORS[hwStatus.profile] ?? ""}`}
                >
                  {hwStatus.profile.replace("_", " ")}
                </span>
              </div>
            </div>
          )}
        </div>

        {/* Right panel — image output */}
        <ImageDisplay
          imageUrl={imageUrl}
          meta={meta}
          generating={generating}
          error={error}
        />
      </main>
    </div>
  )
}
