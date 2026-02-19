"use client";

import { ArrowLeft, Plus, Play } from "lucide-react";
import { useRouter, usePathname } from "next/navigation";
import { Button } from "@/components/ui/button";

interface ActionInfo {
  label: string;
  icon: React.ReactNode;
  onClick: (router: any) => void;
}

interface HeaderInfo {
  title: string;
  description?: string;
  action?: ActionInfo;
}

const routeTitleMap: Record<string, HeaderInfo> = {
  "/": {
    title: "Dashboard Overview",
    description: "Monitor and manage your federated learning network",
  },
  "/clients": {
    title: "All Clients",
    description: "View and manage registered healthcare institutions",
    action: {
      label: "Add New Client",
      icon: <Plus className="h-4 w-4" />,
      onClick: (router) => router.push("/clients/add"),
    },
  },
  "/clients/add": {
    title: "Add New Client",
    description:
      "Register and configure a new healthcare client for federated learning",
  },
  "/federated": {
    title: "FL Simulation History",
    description:
      "View results and progress of previous federated learning runs",
    action: {
      label: "Start New Simulation",
      icon: <Play className="h-4 w-4 fill-current" />,
      onClick: (router) => router.push("/federated/start"),
    },
  },
  "/federated/start": {
    title: "Start FL Simulation",
    description:
      "Initialize a new federated learning process across active clients",
  },
  "/settings": {
    title: "Settings",
    description: "Configure application preferences and system parameters",
  },
};

export default function Header() {
  const router = useRouter();
  const pathname = usePathname();

  const getHeaderInfo = (path: string): HeaderInfo => {
    if (routeTitleMap[path]) return routeTitleMap[path];

    // Handle dynamic routes
    if (path.startsWith("/federated/simulation/")) {
      return {
        title: "Simulation Details",
        description:
          "Review detailed metrics and client performance for this simulation",
      };
    }

    if (path.startsWith("/clients/")) {
      return {
        title: "Client Details",
        description:
          "View specific information and history for this institution",
      };
    }

    // Default fallback
    const segments = path.split("/").filter(Boolean);
    if (segments.length === 0) return routeTitleMap["/"];
    const lastSegment = segments[segments.length - 1];
    return {
      title:
        lastSegment.charAt(0).toUpperCase() +
        lastSegment.slice(1).replace(/-/g, " "),
    };
  };

  const info = getHeaderInfo(pathname);

  return (
    <header className="h-20 sticky top-0 z-40 w-full border-b border-border bg-background/80 backdrop-blur-md flex items-center px-6 gap-4">
      <div className="flex items-center gap-4 flex-1">
        <Button
          variant="ghost"
          size="icon"
          onClick={() => router.back()}
          className="hover:bg-muted"
          title="Go Back"
        >
          <ArrowLeft className="h-4 w-4" />
        </Button>

        <div className="h-8 w-[1px] bg-border mx-2" />

        <div className="flex flex-col">
          <h2 className="text-lg font-semibold text-foreground leading-tight">
            {info.title}
          </h2>
          {info.description && (
            <p className="text-xs text-muted-foreground leading-tight mt-0.5">
              {info.description}
            </p>
          )}
        </div>
      </div>

      {info.action && (
        <Button onClick={() => info.action?.onClick(router)} className="gap-2">
          {info.action.icon}
          {info.action.label}
        </Button>
      )}
    </header>
  );
}
