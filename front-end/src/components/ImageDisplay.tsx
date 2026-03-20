import { useRef } from "react"
import { DownloadSimple, ImageSquare, Warning } from "@phosphor-icons/react"
import { Button } from "@/components/ui/button"
import { Separator } from "@/components/ui/separator"

interface GenerationMeta {
  seed: number
  duration: string
  resolution: number
}

interface Props {
  imageUrl: string | null
  meta: GenerationMeta | null
  generating: boolean
  error: string | null
}

export function ImageDisplay({ imageUrl, meta, generating, error }: Props) {
  const downloadRef = useRef<HTMLAnchorElement>(null)

  const handleDownload = () => {
    if (!imageUrl || !downloadRef.current) return
    downloadRef.current.href = imageUrl
    downloadRef.current.download = `portrait_${meta?.seed ?? Date.now()}.png`
    downloadRef.current.click()
  }

  return (
    <div className="flex h-full flex-col gap-4">
      {/* Canvas */}
      <div className="relative flex aspect-square w-full items-center justify-center overflow-hidden rounded-xl border border-border bg-muted/20">
        {generating && (
          <div className="absolute inset-0 z-10 flex flex-col items-center justify-center gap-3 bg-background/80 backdrop-blur-sm">
            <div className="size-8 animate-spin rounded-full border-2 border-border border-t-primary" />
            <p className="text-xs uppercase tracking-widest text-muted-foreground">
              Generating portrait…
            </p>
            <p className="text-xs text-muted-foreground/60">30 – 60 seconds</p>
          </div>
        )}

        {error && !generating && (
          <div className="flex flex-col items-center gap-2 px-8 text-center">
            <Warning size={32} className="text-destructive" />
            <p className="text-sm font-medium text-destructive">Generation failed</p>
            <p className="text-xs text-muted-foreground">{error}</p>
          </div>
        )}

        {!imageUrl && !generating && !error && (
          <div className="flex flex-col items-center gap-3 px-8 text-center">
            <ImageSquare size={48} className="text-muted-foreground/20" weight="thin" />
            <p className="text-xs uppercase tracking-widest text-muted-foreground/40">
              Your portrait will appear here
            </p>
          </div>
        )}

        {imageUrl && !generating && (
          <img
            src={imageUrl}
            alt="Generated portrait"
            className="h-full w-full object-cover"
          />
        )}
      </div>

      {/* Metadata + download */}
      {meta && imageUrl && !generating && (
        <>
          <Separator />
          <div className="flex items-center justify-between">
            <div className="flex gap-4 text-xs text-muted-foreground">
              <span>
                <span className="font-medium text-foreground">
                  {meta.resolution}×{meta.resolution}
                </span>{" "}
                px
              </span>
              <span>
                Seed{" "}
                <span className="font-mono font-medium text-foreground">{meta.seed}</span>
              </span>
              <span>
                <span className="font-medium text-foreground">{meta.duration}s</span>
              </span>
            </div>
            <Button variant="outline" size="sm" onClick={handleDownload}>
              <DownloadSimple size={14} />
              Download
            </Button>
          </div>
        </>
      )}

      {/* Hidden anchor for programmatic download */}
      <a ref={downloadRef} className="hidden" />
    </div>
  )
}
