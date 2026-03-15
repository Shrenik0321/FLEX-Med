"use client";

import {
  Users,
  Plus,
  Brain,
  History,
  Play,
  UsersRound,
  Settings,
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

  const linkCls = (active: boolean) =>
    `w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
      active
        ? "text-[#b80028] bg-red-50"
        : "text-gray-600 hover:text-gray-900 hover:bg-gray-50"
    }`;

  return (
    <div className="w-58 bg-white border-r border-gray-300 flex flex-col">
      {/* Logo */}
      <div className="h-14 flex items-center px-4 border-b border-gray-300">
        <Link href="/clients" className="flex items-center gap-2">
          <div className="w-6 h-6 bg-[#b80028] rounded-md flex items-center justify-center">
            <span className="text-[10px] font-bold text-white">FM</span>
          </div>
          <span className="text-sm font-semibold text-gray-900 tracking-tight">
            FLEX-Med
          </span>
        </Link>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-3 py-4 space-y-0.5">
        {/* Clients */}
        <Accordion type="single" collapsible className="w-full">
          <AccordionItem value="clients" className="border-none">
            <AccordionTrigger className="w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm font-medium text-gray-600 hover:text-gray-900 hover:bg-gray-50 hover:no-underline [&>svg]:text-gray-500">
              <div className="flex items-center gap-2.5">
                <Users size={15} />
                <span>Clients</span>
              </div>
            </AccordionTrigger>
            <AccordionContent className="pl-3 pt-0.5 space-y-0.5">
              <Link
                href="/clients"
                className={linkCls(pathname === "/clients")}
              >
                <UsersRound size={14} />
                All Clients
              </Link>
              <Link
                href="/clients/add"
                className={linkCls(pathname === "/clients/add")}
              >
                <Plus size={14} />
                Add Client
              </Link>
            </AccordionContent>
          </AccordionItem>
        </Accordion>

        {/* Federated Learning */}
        <Accordion type="single" collapsible className="w-full">
          <AccordionItem value="federated" className="border-none">
            <AccordionTrigger className="w-full flex items-center gap-2.5 px-3 py-2 rounded-lg text-sm font-medium text-gray-600 hover:text-gray-900 hover:bg-gray-50 hover:no-underline [&>svg]:text-gray-500">
              <div className="flex items-center gap-2.5">
                <Brain size={15} />
                <span>Federated Learning</span>
              </div>
            </AccordionTrigger>
            <AccordionContent className="pl-3 pt-0.5 space-y-0.5">
              <Link
                href="/federated"
                className={linkCls(pathname === "/federated")}
              >
                <History size={14} />
                Simulation History
              </Link>
              <Link
                href="/federated/start"
                className={linkCls(pathname === "/federated/start")}
              >
                <Play size={14} />
                Start Simulation
              </Link>
            </AccordionContent>
          </AccordionItem>
        </Accordion>

        {/* Settings */}
        <Link href="/settings" className={linkCls(pathname === "/settings")}>
          <Settings size={15} />
          <span>Settings</span>
        </Link>
      </nav>
    </div>
  );
}
