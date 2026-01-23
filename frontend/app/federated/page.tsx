"use client";

import FLHistoryPage from "@/components/pages/fl-history-page";
import { useRouter } from "next/navigation";

export default function FederatedLearningPage() {
  const router = useRouter();

  return (
    <FLHistoryPage
      onStartClick={() => router.push("/federated/start")}
      onSelectSimulation={(simulation) => {
        // Store simulation data in localStorage for the details page
        localStorage.setItem("selectedSimulation", JSON.stringify(simulation));
        router.push(`/federated/${simulation.id}`);
      }}
    />
  );
}
