import { useState, useEffect } from "react"
import { Label } from "@/components/ui/label"
import { Separator } from "@/components/ui/separator"

const ATTRIBUTES = {
  shotType: {
    label: "Shot type",
    options: [
      { value: "portrait photo", label: "Portrait photo" },
      { value: "cinematic close-up", label: "Cinematic close-up" },
      { value: "studio portrait", label: "Studio portrait" },
      { value: "close-up portrait", label: "Close-up portrait" },
      { value: "editorial portrait", label: "Editorial portrait" },
    ],
  },
  subject: {
    label: "Subject",
    options: [
      { value: "a young woman", label: "Young woman" },
      { value: "a young man", label: "Young man" },
      { value: "a middle-aged woman", label: "Middle-aged woman" },
      { value: "a middle-aged man", label: "Middle-aged man" },
      { value: "an elderly woman", label: "Elderly woman" },
      { value: "an elderly man", label: "Elderly man" },
      { value: "a teenage girl", label: "Teenage girl" },
      { value: "a teenage boy", label: "Teenage boy" },
    ],
  },
  hairColor: {
    label: "Hair color",
    options: [
      { value: "brown hair", label: "Brown" },
      { value: "blonde hair", label: "Blonde" },
      { value: "black hair", label: "Black" },
      { value: "red hair", label: "Red" },
      { value: "gray hair", label: "Gray" },
      { value: "white hair", label: "White" },
      { value: "silver hair", label: "Silver" },
      { value: "auburn hair", label: "Auburn" },
    ],
  },
  hairStyle: {
    label: "Hair style",
    options: [
      { value: "straight hair", label: "Straight" },
      { value: "curly hair", label: "Curly" },
      { value: "wavy hair", label: "Wavy" },
      { value: "short hair", label: "Short" },
      { value: "long hair", label: "Long" },
      { value: "braided hair", label: "Braided" },
    ],
  },
  expression: {
    label: "Expression",
    options: [
      { value: "warm smile", label: "Warm smile" },
      { value: "smiling", label: "Smiling" },
      { value: "serious expression", label: "Serious" },
      { value: "neutral expression", label: "Neutral" },
      { value: "laughing", label: "Laughing" },
      { value: "kind eyes", label: "Kind eyes" },
      { value: "thoughtful expression", label: "Thoughtful" },
    ],
  },
  extras: {
    label: "Features",
    options: [
      { value: "", label: "None" },
      { value: "wearing glasses", label: "Glasses" },
      { value: "with freckles", label: "Freckles" },
      { value: "with a short beard", label: "Short beard" },
      { value: "subtle makeup", label: "Subtle makeup" },
      { value: "wearing earrings", label: "Earrings" },
      { value: "with deep wrinkles", label: "Wrinkles" },
    ],
  },
  lighting: {
    label: "Lighting",
    options: [
      { value: "soft studio lighting", label: "Soft studio" },
      { value: "dramatic side lighting", label: "Dramatic side" },
      { value: "natural daylight", label: "Natural daylight" },
      { value: "golden hour lighting", label: "Golden hour" },
      { value: "warm indoor lighting", label: "Warm indoor" },
      { value: "soft window light", label: "Window light" },
      { value: "high contrast lighting", label: "High contrast" },
    ],
  },
  style: {
    label: "Style",
    options: [
      { value: "hyperrealistic", label: "Hyperrealistic" },
      { value: "cinematic", label: "Cinematic" },
      { value: "film grain", label: "Film grain" },
      { value: "professional photography", label: "Professional" },
      { value: "editorial photography style", label: "Editorial" },
    ],
  },
  quality: {
    label: "Quality",
    options: [
      { value: "8k resolution, sharp focus, highly detailed", label: "8K Ultra" },
      { value: "4k resolution, sharp focus", label: "4K High" },
      { value: "ultra detailed face, shallow depth of field", label: "Ultra detail" },
      { value: "detailed skin texture, sharp focus", label: "Detailed skin" },
    ],
  },
} as const

type AttributeKey = keyof typeof ATTRIBUTES
type Selections = Record<AttributeKey, string>

const DEFAULTS: Selections = {
  shotType: "portrait photo",
  subject: "a young woman",
  hairColor: "brown hair",
  hairStyle: "straight hair",
  expression: "warm smile",
  extras: "",
  lighting: "soft studio lighting",
  style: "hyperrealistic",
  quality: "8k resolution, sharp focus, highly detailed",
}

function buildPrompt(sel: Selections): string {
  return [
    `a ${sel.style} ${sel.shotType}`,
    `of ${sel.subject}`,
    `with ${sel.hairColor}, ${sel.hairStyle}`,
    sel.expression,
    sel.extras,
    sel.lighting,
    sel.quality,
  ]
    .filter(Boolean)
    .join(", ")
}

interface Props {
  onPromptChange: (prompt: string) => void
}

export function PromptBuilder({ onPromptChange }: Props) {
  const [selections, setSelections] = useState<Selections>(DEFAULTS)

  useEffect(() => {
    onPromptChange(buildPrompt(selections))
  }, [selections, onPromptChange])

  const set = (key: AttributeKey, value: string) =>
    setSelections((prev) => ({ ...prev, [key]: value }))

  const selectClass =
    "w-full rounded-lg border border-border bg-background px-3 py-2 text-sm text-foreground focus:outline-none focus:ring-2 focus:ring-ring transition-colors cursor-pointer appearance-none"

  return (
    <div className="flex flex-col gap-4">
      <div className="grid grid-cols-2 gap-3">
        {(Object.keys(ATTRIBUTES) as AttributeKey[]).map((key) => {
          const attr = ATTRIBUTES[key]
          return (
            <div key={key} className="flex flex-col gap-1.5">
              <Label htmlFor={`sel-${key}`}>{attr.label}</Label>
              <div className="relative">
                <select
                  id={`sel-${key}`}
                  value={selections[key]}
                  onChange={(e) => set(key, e.target.value)}
                  className={selectClass}
                >
                  {attr.options.map((opt) => (
                    <option key={opt.value} value={opt.value}>
                      {opt.label}
                    </option>
                  ))}
                </select>
                <div className="pointer-events-none absolute inset-y-0 right-2.5 flex items-center">
                  <svg
                    className="size-3.5 text-muted-foreground"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth={2}
                  >
                    <polyline points="6 9 12 15 18 9" />
                  </svg>
                </div>
              </div>
            </div>
          )
        })}
      </div>

      <Separator />

      {/* Live prompt preview */}
      <div className="flex flex-col gap-2">
        <Label>Generated prompt preview</Label>
        <div className="rounded-lg border border-border bg-muted/20 p-3 font-mono text-xs leading-relaxed text-muted-foreground">
          {buildPrompt(selections)}
        </div>
      </div>
    </div>
  )
}
