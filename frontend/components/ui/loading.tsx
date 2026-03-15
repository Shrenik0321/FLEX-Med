"use client";

import { cn } from "@/lib/utils";

interface LoadingProps {
  className?: string;
  size?: "sm" | "md" | "lg" | "xl";
  text?: string;
  fullScreen?: boolean;
}

const sizeMap = {
  sm: "h-5 w-5 border-2",
  md: "h-8 w-8 border-2",
  lg: "h-10 w-10 border-2",
  xl: "h-14 w-14 border-[3px]",
};

export function Loading({
  className,
  size = "md",
  text,
  fullScreen = false,
}: LoadingProps) {
  const loader = (
    <div
      className={cn(
        "flex flex-col items-center justify-center gap-3",
        className,
      )}
    >
      <div
        className={cn(
          "rounded-full border-gray-100 border-t-[#b80028] animate-spin",
          sizeMap[size],
        )}
      />
      {text && (
        <span className="text-sm font-medium text-gray-400">{text}</span>
      )}
    </div>
  );

  if (fullScreen) {
    return (
      <div className="fixed inset-0 z-[100] flex items-center justify-center bg-white/70 backdrop-blur-sm">
        {loader}
      </div>
    );
  }

  return loader;
}
