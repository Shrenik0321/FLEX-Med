"use client";

import {
  Home,
  Users,
  Plus,
  Brain,
  History,
  Play,
  UsersRound,
} from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  Accordion,
  AccordionContent,
  AccordionItem,
  AccordionTrigger,
} from "@/components/ui/accordion";

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <div className="w-64 bg-card border-r border-border flex flex-col">
      {/* Logo */}
      <div className="h-20 flex items-center px-6 border-b border-border">
        <Link href="/" className="flex items-center gap-2">
          <div className="w-8 h-8 bg-primary rounded-lg flex items-center justify-center text-white">
            <span className="font-bold">FM</span>
          </div>
          <h1 className="text-lg font-semibold text-foreground">FLEX-Med</h1>
        </Link>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-4 py-6 space-y-2">
        {/* Home Link */}
        <Link
          href="/"
          className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all ${
            pathname === "/"
              ? "text-primary bg-primary/10 flex-red-border-left"
              : "text-muted-foreground hover:text-foreground hover:bg-muted"
          }`}
        >
          <Home size={20} />
          <span>Home</span>
        </Link>

        {/* Clients Accordion */}
        <Accordion type="single" collapsible className="w-full">
          <AccordionItem value="clients" className="border-none">
            <AccordionTrigger className="w-full flex items-center gap-3 px-4 py-0 rounded-lg text-sm font-medium text-muted-foreground hover:text-foreground hover:bg-muted hover:no-underline">
              <div className="flex items-center gap-3">
                <Users size={20} />
                <span>Clients</span>
              </div>
            </AccordionTrigger>
            <AccordionContent className="pl-4 space-y-1">
              <Link
                href="/clients"
                className={`w-full flex items-center gap-3 px-4 py-2 rounded-lg text-sm transition-all ${
                  pathname === "/clients"
                    ? "text-primary bg-primary/10"
                    : "text-muted-foreground hover:text-foreground hover:bg-muted"
                }`}
              >
                <UsersRound size={16} />
                All Clients
              </Link>
              <Link
                href="/clients/add"
                className={`w-full flex items-center gap-3 px-4 py-2 rounded-lg text-sm transition-all ${
                  pathname === "/clients/add"
                    ? "text-primary bg-primary/10"
                    : "text-muted-foreground hover:text-foreground hover:bg-muted"
                }`}
              >
                <Plus size={16} />
                Add Client
              </Link>
            </AccordionContent>
          </AccordionItem>
        </Accordion>

        {/* Federated Learning Accordion */}
        <Accordion type="single" collapsible className="w-full">
          <AccordionItem value="federated" className="border-none">
            <AccordionTrigger className="w-full flex items-center gap-3 px-4 py-0 rounded-lg text-sm font-medium text-muted-foreground hover:text-foreground hover:bg-muted hover:no-underline">
              <div className="flex items-center gap-3">
                <Brain size={20} />
                <span>Federated Learning</span>
              </div>
            </AccordionTrigger>
            <AccordionContent className="pl-4 space-y-1">
              <Link
                href="/federated"
                className={`w-full flex items-center gap-3 px-4 py-2 rounded-lg text-sm transition-all ${
                  pathname === "/federated"
                    ? "text-primary bg-primary/10"
                    : "text-muted-foreground hover:text-foreground hover:bg-muted"
                }`}
              >
                <History size={16} />
                FL Simulation History
              </Link>
              <Link
                href="/federated/start"
                className={`w-full flex items-center gap-3 px-4 py-2 rounded-lg text-sm transition-all ${
                  pathname === "/federated/start"
                    ? "text-primary bg-primary/10"
                    : "text-muted-foreground hover:text-foreground hover:bg-muted"
                }`}
              >
                <Play size={16} />
                Start FL Simulation
              </Link>
            </AccordionContent>
          </AccordionItem>
        </Accordion>
      </nav>
    </div>
  );
}
