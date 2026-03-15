"use client";

import { Plus, Play } from "lucide-react";
import { useRouter, usePathname } from "next/navigation";

interface ActionInfo {
  label: string;
  onClick: (router: any) => void;
}

interface HeaderInfo {
  title: string;
  description?: string;
  action?: ActionInfo;
}

const routeTitleMap: Record<string, HeaderInfo> = {
  "/clients": {
    title: "All Clients",
    description: "Registered healthcare institutions",
    action: {
      label: "Add Client",
      onClick: (router) => router.push("/clients/add"),
    },
  },
  "/clients/add": {
    title: "Add Client",
    description: "Register a new client for federated learning",
  },
  "/federated": {
    title: "FL Simulations",
    description: "History of federated learning runs",
    action: {
      label: "Start Simulation",
      onClick: (router) => router.push("/federated/start"),
    },
  },
  "/federated/start": {
    title: "Start Simulation",
    description: "Initialize a new federated learning run",
  },
  "/settings": {
    title: "Settings",
    description: "Training parameters and system configuration",
  },
};

export default function Header() {
  const router = useRouter();
  const pathname = usePathname();

  const getHeaderInfo = (path: string): HeaderInfo => {
    if (routeTitleMap[path]) return routeTitleMap[path];

    if (path.startsWith("/federated/")) {
      return { title: "Simulation Details" };
    }
    if (path.startsWith("/clients/")) {
      return { title: "Client Details" };
    }

    const segments = path.split("/").filter(Boolean);
    if (segments.length === 0) return { title: "FLEX-Med" };
    const last = segments[segments.length - 1];
    return {
      title: last.charAt(0).toUpperCase() + last.slice(1).replace(/-/g, " "),
    };
  };

  const info = getHeaderInfo(pathname);

  return (
    <header className="h-14 sticky top-0 z-40 w-full border-b border-gray-100 bg-white/90 backdrop-blur flex items-center px-6 gap-4">
      <div className="flex flex-col flex-1">
        <span className="text-base font-semibold text-gray-900 leading-tight">
          {info.title}
        </span>
        {info.description && (
          <span className="text-xs text-gray-400 leading-tight mt-0.5">
            {info.description}
          </span>
        )}
      </div>

      {info.action && (
        <button
          onClick={() => info.action?.onClick(router)}
          className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#b80028] text-white text-xs font-medium hover:bg-[#9b0022] transition-colors"
        >
          {info.title === "All Clients" ? (
            <Plus size={12} />
          ) : (
            <Play size={12} className="fill-current" />
          )}
          {info.action.label}
        </button>
      )}
    </header>
  );
}
