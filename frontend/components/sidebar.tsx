"use client";

import { Home, Users, Plus, Brain, Settings, Database } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";

export default function Sidebar() {
  const pathname = usePathname();

  const navItems = [
    { id: "/", label: "Home", icon: Home },
    { id: "/clients", label: "Clients", icon: Users },
    { id: "/federated", label: "Fed Learning", icon: Brain },
    { id: "/public-dataset", label: "Public Dataset", icon: Database },
    { id: "/settings", label: "Settings", icon: Settings },
  ];

  return (
    <div className="w-64 bg-white border-r border-[#E2E8F0] flex flex-col">
      {/* Logo */}
      <div className="p-6 border-b border-[#E2E8F0]">
        <Link href="/" className="flex items-center gap-2">
          <div className="w-8 h-8 bg-[#B80028] rounded-lg flex items-center justify-center">
            <span className="text-white font-bold">FM</span>
          </div>
          <h1 className="text-lg font-semibold text-gray-900">FLEX-Med</h1>
        </Link>
      </div>

      {/* Navigation */}
      <nav className="flex-1 px-4 py-6 space-y-2">
        {navItems.map((item) => {
          const Icon = item.icon;
          const isActive = pathname === item.id;
          return (
            <Link
              key={item.id}
              href={item.id}
              className={`w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-all ${
                isActive
                  ? "text-[#B80028] bg-[rgba(184,0,40,0.08)] flex-red-border-left"
                  : "text-[#718096] hover:text-gray-900 hover:bg-[#F7F8FA]"
              }`}
            >
              <Icon size={20} />
              <span>{item.label}</span>
            </Link>
          );
        })}
      </nav>

      {/* Footer */}
      <div className="p-4 border-t border-[#E2E8F0]">
        <Link
          href="/clients/add"
          className="w-full flex items-center justify-center gap-2 px-4 py-2 text-[#B80028] border border-[#B80028] rounded-lg hover:bg-[rgba(184,0,40,0.08)] transition-colors text-sm font-medium"
        >
          <Plus size={18} />
          Add Client
        </Link>
      </div>
    </div>
  );
}
