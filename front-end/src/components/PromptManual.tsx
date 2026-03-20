import { useState } from "react"
import { CaretDown, CaretUp, ClipboardText } from "@phosphor-icons/react"
import { Textarea } from "@/components/ui/textarea"
import { Label } from "@/components/ui/label"
import { Separator } from "@/components/ui/separator"

const EXAMPLES = [
  "a hyperrealistic portrait photo of a young woman with brown hair, soft studio lighting, detailed skin texture, 8k resolution, sharp focus, neutral background, subtle makeup",
  "a cinematic close-up of a middle-aged man with a short beard, dramatic side lighting, high contrast, film grain, ultra detailed face, shallow depth of field, 4k resolution",
  "a portrait of an elderly woman with deep wrinkles and kind eyes, warm indoor lighting, high detail, natural skin tones, sharp focus, soft blurred background, 8k",
  "a studio portrait of a young man with curly dark hair, confident expression, soft key light and rim light, highly detailed eyes, photorealistic, 4k ultra sharp",
]

const HINT_CATEGORIES = [
  {
    label: "Shot type",
    tags: ["portrait photo", "cinematic close-up", "studio portrait", "close-up portrait", "editorial portrait"],
  },
  {
    label: "Subject",
    tags: ["a young woman", "a young man", "a middle-aged woman", "an elderly man", "a teenage girl"],
  },
  {
    label: "Hair",
    tags: ["brown hair", "blonde hair", "black hair", "curly hair", "wavy hair", "short hair"],
  },
  {
    label: "Expression",
    tags: ["warm smile", "serious expression", "laughing", "neutral expression", "kind eyes"],
  },
  {
    label: "Lighting",
    tags: ["soft studio lighting", "dramatic side lighting", "natural daylight", "golden hour", "warm indoor lighting"],
  },
  {
    label: "Quality",
    tags: ["8k resolution", "4k resolution", "sharp focus", "highly detailed", "shallow depth of field", "photorealistic"],
  },
]

interface Props {
  prompt: string
  onChange: (value: string) => void
}

export function PromptManual({ prompt, onChange }: Props) {
  const [hintsOpen, setHintsOpen] = useState(false)
  const [appliedIdx, setAppliedIdx] = useState<number | null>(null)

  const appendTag = (tag: string) => {
    const trimmed = prompt.trim()
    onChange(trimmed ? `${trimmed}, ${tag}` : tag)
  }

  const useExample = (example: string, idx: number) => {
    onChange(example)
    setAppliedIdx(idx)
    setTimeout(() => setAppliedIdx(null), 1500)
  }

  return (
    <div className="flex flex-col gap-4">
      {/* Prompt textarea */}
      <div className="flex flex-col gap-2">
        <Label htmlFor="prompt-input">Describe your portrait</Label>
        <Textarea
          id="prompt-input"
          value={prompt}
          onChange={(e) => onChange(e.target.value)}
          placeholder="a hyperrealistic portrait photo of a young woman with brown hair, soft studio lighting, 8k resolution, sharp focus…"
          className="min-h-[120px] font-mono text-xs leading-relaxed"
        />
        <p className="text-xs text-muted-foreground">
          Structure:{" "}
          <span className="text-foreground">
            [shot type] of [subject] with [hair], [expression], [lighting], [quality]
          </span>
        </p>
      </div>

      {/* Hints toggle */}
      <button
        type="button"
        onClick={() => setHintsOpen((v) => !v)}
        className="flex w-full items-center justify-between rounded-lg border border-border px-3 py-2 text-xs font-medium text-muted-foreground transition-colors hover:bg-muted/50 hover:text-foreground"
      >
        <span>Keyword hints — click to add to prompt</span>
        {hintsOpen ? <CaretUp size={14} /> : <CaretDown size={14} />}
      </button>

      {hintsOpen && (
        <div className="flex flex-col gap-3 rounded-lg border border-border bg-muted/20 p-3">
          {HINT_CATEGORIES.map((cat) => (
            <div key={cat.label} className="flex flex-col gap-1.5">
              <span className="text-xs font-medium uppercase tracking-widest text-muted-foreground">
                {cat.label}
              </span>
              <div className="flex flex-wrap gap-1.5">
                {cat.tags.map((tag) => (
                  <button
                    key={tag}
                    type="button"
                    onClick={() => appendTag(tag)}
                    className="cursor-pointer rounded-full border border-border px-2.5 py-0.5 text-xs text-muted-foreground transition-colors hover:border-primary/50 hover:bg-primary/10 hover:text-primary"
                  >
                    {tag}
                  </button>
                ))}
              </div>
            </div>
          ))}
        </div>
      )}

      <Separator />

      {/* Example prompts */}
      <div className="flex flex-col gap-2">
        <Label>Example prompts</Label>
        <div className="flex flex-col gap-2">
          {EXAMPLES.map((example, i) => (
            <button
              key={i}
              type="button"
              onClick={() => useExample(example, i)}
              className="group flex w-full items-start gap-2 rounded-lg border border-border px-3 py-2.5 text-left text-xs text-muted-foreground transition-colors hover:border-primary/40 hover:bg-primary/5 hover:text-foreground"
            >
              <ClipboardText
                size={13}
                className="mt-0.5 shrink-0 text-muted-foreground/50 group-hover:text-primary"
              />
              <span className="line-clamp-2 leading-relaxed">
                {appliedIdx === i ? "✓ Applied" : example}
              </span>
            </button>
          ))}
        </div>
      </div>
    </div>
  )
}
