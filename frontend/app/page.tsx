"use client";

import { useState } from "react";
import { Suspense } from "react";
import Sidebar from "@/components/sidebar";
import HomePage from "@/components/pages/home-page";
import ClientsListPage from "@/components/pages/clients-list-page";
import AddClientPage from "@/components/pages/add-client-page";
import ClientDetailsPage from "@/components/pages/client-details-page";
import SettingsPage from "@/components/pages/settings-page";
import FederatedPage from "@/components/pages/federated-page";
import { AuthButton } from "@/components/auth-button";
import { EnvVarWarning } from "@/components/env-var-warning";
import { ThemeSwitcher } from "@/components/theme-switcher";
import { hasEnvVars } from "@/lib/utils";

type Client = {
  id: number;
  client_name: string;
  client_email: string;
  status: string;
  model_type: number | string;
  created_at: string;
};

export default function Dashboard() {
  const [currentPage, setCurrentPage] = useState("home");
  const [selectedClient, setSelectedClient] = useState<Client | null>(null);

  const renderPage = () => {
    switch (currentPage) {
      case "home":
        return <HomePage />;
      case "clients":
        return (
          <ClientsListPage
            onAddClick={() => setCurrentPage("add-client")}
            onSelectClient={(client) => {
              setSelectedClient(client);
              setCurrentPage("client-details");
            }}
          />
        );
      case "add-client":
        return <AddClientPage onBack={() => setCurrentPage("clients")} />;
      case "client-details":
        return selectedClient ? (
          <ClientDetailsPage
            client={selectedClient}
            onBack={() => setCurrentPage("clients")} 
          />
        ) : (
          <HomePage />
        );
      case "federated":
        return <FederatedPage />;
      case "settings":
        return <SettingsPage />;
      default:
        return <HomePage />;
    }
  };

  return (
    <div className="flex flex-col h-screen bg-[#F7F8FA]">
      {/* Top Navigation Bar with Authentication */}
      <nav className="w-full flex justify-center border-b border-b-foreground/10 bg-white">
        <div className="w-full flex justify-between items-center p-3 px-5 text-sm">
          <div className="flex gap-5 items-center font-semibold">
            <span>FlexMed Dashboard</span>
          </div>
          <div className="flex items-center gap-4">
            <ThemeSwitcher />
            {!hasEnvVars ? (
              <EnvVarWarning />
            ) : (
              <Suspense>
                <AuthButton />
              </Suspense>
            )}
          </div>
        </div>
      </nav>

      {/* Main Dashboard Content */}
      <div className="flex flex-1 overflow-hidden">
        <Sidebar currentPage={currentPage} onPageChange={setCurrentPage} />
        <div className="flex-1 overflow-auto">{renderPage()}</div>
      </div>
    </div>
  );
}
