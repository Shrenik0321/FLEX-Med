"use client";

import FLSimulationDetailsPage from "@/components/pages/fl-simulation-details-page";
import { useRouter, useParams } from "next/navigation";
import { useEffect, useState } from "react";

export default function SimulationDetailsPage() {
  const router = useRouter();
  const params = useParams();
  const simulationId = parseInt(params.id as string);
  const [simulationName, setSimulationName] = useState(
    `FL Simulation #${simulationId}`
  );

  useEffect(() => {
    // Try to get the simulation name from localStorage
    const storedData = localStorage.getItem("selectedSimulation");
    if (storedData) {
      const simulation = JSON.parse(storedData);
      if (simulation.id === simulationId) {
        setSimulationName(simulation.run_name);
      }
    }
  }, [simulationId]);

  return (
    <FLSimulationDetailsPage
      simulationId={simulationId}
      simulationName={simulationName}
      onBack={() => router.push("/federated")}
    />
  );
}
