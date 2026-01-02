import { cn } from "@/lib/utils"

type Status = "training" | "completed" | "idle" | "error"

export function StatusBadge({ status, children }: { status: Status; children: React.ReactNode }) {
  const variants = {
    training: "border-primary text-primary bg-primary/10",
    completed: "border-emerald-500 text-emerald-700 bg-emerald-50 dark:bg-emerald-950/30 dark:text-emerald-400",
    idle: "border-border text-muted-foreground bg-muted",
    error: "border-red-500 text-red-700 bg-red-50 dark:bg-red-950/30 dark:text-red-400"
  }

  return (
    <span className={cn(
      "inline-block px-3 py-1 rounded-full text-xs font-medium border",
      variants[status]
    )}>
      {children}
    </span>
  )
}
