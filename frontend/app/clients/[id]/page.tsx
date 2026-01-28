"use client";

import { useEffect, useState } from "react";
import { useRouter, useParams } from "next/navigation";
import ClientDetailsPage from "@/components/pages/client-details-page";

type Client = {
  id: number;
  client_name: string;
  client_email: string;
  status: string;
  model_type: number | string;
  created_at: string;
};

export default function ClientDetails() {
  const router = useRouter();
  const params = useParams();
  const [client, setClient] = useState<Client | null>(null);

  useEffect(() => {
    // Try to get client data from localStorage
    const storedClient = localStorage.getItem("selectedClient");
    if (storedClient) {
      const clientData = JSON.parse(storedClient);
      // Verify it's the right client
      if (clientData.id === parseInt(params.id as string)) {
        setClient(clientData);
      }
    }
  }, [params.id]);

  if (!client) {
    return (
      <div className="p-8">
        <p>Loading client details...</p>
      </div>
    );
  }

  return <ClientDetailsPage client={client} />;
}
