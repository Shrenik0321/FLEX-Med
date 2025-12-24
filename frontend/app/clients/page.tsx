"use client";

import ClientsListPage from "@/components/pages/clients-list-page";
import { useRouter } from "next/navigation";

export default function ClientsPage() {
  const router = useRouter();

  return (
    <ClientsListPage
      onAddClick={() => router.push("/clients/add")}
      onSelectClient={(client) => {
        // Store client data in localStorage or use route params
        localStorage.setItem("selectedClient", JSON.stringify(client));
        router.push(`/clients/${client.id}`);
      }}
    />
  );
}
