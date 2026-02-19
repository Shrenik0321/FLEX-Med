"use client";

import { Loader2 } from "lucide-react";
import { cn } from "@/lib/utils";

interface LoadingProps {
  className?: string;
  size?: "sm" | "md" | "lg" | "xl";
  text?: string;
  fullScreen?: boolean;
}

export function Loading({
  className,
  size = "md",
  text,
  fullScreen = false,
}: LoadingProps) {
  const sizeClasses = {
    sm: "h-4 w-4 border-2",
    md: "h-8 w-8 border-2",
    lg: "h-12 w-12 border-3",
    xl: "h-16 w-16 border-4",
  };

  const loader = (
    <div
      className={cn(
        "flex flex-col items-center justify-center gap-3",
        className,
      )}
    >
      <div className="relative">
        <Loader2
          className={cn(
            "animate-spin text-primary/80",
            size === "sm" && "h-4 w-4",
            size === "md" && "h-8 w-8",
            size === "lg" && "h-12 w-12",
            size === "xl" && "h-16 w-16",
          )}
        />
        {/* Subtle pulse background */}
        <div
          className={cn(
            "absolute inset-0 rounded-full bg-primary/10 animate-pulse",
            size === "sm" && "scale-150",
            size === "md" && "scale-125",
            size === "lg" && "scale-110",
            size === "xl" && "scale-105",
          )}
        />
      </div>
      {text && (
        <span className="text-sm font-medium text-muted-foreground animate-pulse">
          {text}
        </span>
      )}
    </div>
  );

  if (fullScreen) {
    return (
      <div className="fixed inset-0 z-[100] flex items-center justify-center bg-background/60 backdrop-blur-sm transition-all duration-300">
        {loader}
      </div>
    );
  }

  return loader;
}
