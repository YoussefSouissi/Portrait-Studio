import type { HTMLAttributes } from "react"
import { cn } from "@/lib/utils"

interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  variant?: "default" | "secondary" | "outline"
}

function Badge({ className, variant = "default", ...props }: BadgeProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium transition-colors",
        variant === "default" && "bg-primary/10 text-primary",
        variant === "secondary" && "bg-secondary/10 text-secondary-foreground",
        variant === "outline" && "border border-border text-muted-foreground",
        className
      )}
      {...props}
    />
  )
}

export { Badge }
