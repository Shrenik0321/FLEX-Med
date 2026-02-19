"use client";

import FLSimulationListPage from "@/components/pages/fl-simulations-list-page";
import { useRouter } from "next/navigation";

export default function FederatedLearningPage() {
  const router = useRouter();

  return (
    <FLSimulationListPage
      onStartClick={() => router.push("/federated/start")}
      onSelectSimulation={(simulation) => {
        // Store simulation data in localStorage for the details page
        localStorage.setItem("selectedSimulation", JSON.stringify(simulation));
        router.push(`/federated/${simulation.id}`);
      }}
    />
  );
}
