import React from "react";

interface EvalCardProps {
  title: string;
  description: string;
  content: React.ReactNode;
}

export default function EvalCard({
  title,
  description,
  content,
}: EvalCardProps) {
  return (
    <div className="bg-white border border-slate-200 rounded-3xl p-8 shadow-sm hover:shadow-xl transition-all duration-300">
      <div className="mb-6">
        <h3 className="text-xl font-bold text-slate-900 tracking-tight">
          {title}
        </h3>
        <p className="text-slate-500 text-sm mt-1">{description}</p>
      </div>

      <div className="h-[300px] w-full">{content}</div>
    </div>
  );
}
